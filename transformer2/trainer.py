"""
训练器模块 - 重构版本
负责模型训练、验证和保存
详细的数据流转注释和shape说明
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
import json
import time
import math
from typing import Dict, Optional

from config import TRANSFORMER_CONFIG, TRAINING_CONFIG, setup_logging
from model import Transformer
from data_loader import DataManager
from transformer import create_padding_mask, create_combined_mask

logger = logging.getLogger("Transformer2")


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数

    减少模型过拟合，提高泛化能力

    数据流转：
    logits: [batch_size, seq_len, vocab_size]
    targets: [batch_size, seq_len]
    -> 计算平滑标签 -> 计算KL散度损失
    """

    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = 0):
        """
        初始化标签平滑损失

        Args:
            vocab_size: 词汇表大小
            smoothing: 平滑系数
            ignore_index: 忽略的索引(通常是padding token)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

        logger.debug(f"标签平滑损失初始化: vocab_size={vocab_size}, smoothing={smoothing}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            logits: 模型输出 [batch_size, seq_len, vocab_size]
            targets: 目标标签 [batch_size, seq_len]

        Returns:
            损失值 [1]
        """
        batch_size, seq_len, vocab_size = logits.shape

        # 重塑为二维
        logits = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
        targets = targets.view(-1)  # [batch_size * seq_len]

        # 计算log概率
        log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size * seq_len, vocab_size]

        # 创建平滑标签
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (vocab_size - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        # 忽略padding token
        mask = (targets != self.ignore_index).float()
        true_dist = true_dist * mask.unsqueeze(1)

        # 计算KL散度
        loss = -torch.sum(true_dist * log_probs, dim=-1)  # [batch_size * seq_len]
        loss = loss * mask  # 应用掩码

        return loss.sum() / mask.sum()


class WarmupScheduler:
    """学习率预热调度器

    实现Transformer论文中的学习率调度策略
    """

    def __init__(self, optimizer: optim.Optimizer, d_model: int, warmup_steps: int):
        """
        初始化调度器

        Args:
            optimizer: 优化器
            d_model: 模型维度
            warmup_steps: 预热步数
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

        logger.debug(f"学习率调度器初始化: d_model={d_model}, warmup_steps={warmup_steps}")

    def step(self) -> float:
        """更新学习率"""
        self.step_num += 1

        # 计算学习率
        lr = (self.d_model**-0.5) * min(self.step_num**-0.5, self.step_num * (self.warmup_steps**-1.5))

        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


class Trainer:
    """Transformer训练器

    负责模型训练的完整流程，包括：
    - 模型初始化
    - 数据准备
    - 训练循环
    - 验证
    - 模型保存
    """

    def __init__(self):
        """初始化训练器"""
        # 设置日志
        setup_logging()

        # 设置设备
        self.device = TRAINING_CONFIG.device
        logger.info(f"使用设备: {self.device}")

        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.data_manager = DataManager()

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        # 模型保存目录
        self.model_save_dir = TRAINING_CONFIG.model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)

        logger.info("训练器初始化完成")

    def setup_model(self):
        """设置模型和优化器"""
        logger.info("正在设置模型...")

        # 创建模型
        self.model = Transformer().to(self.device)

        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"模型参数数量: 总计={total_params:,}, 可训练={trainable_params:,}")

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG.learning_rate,
            betas=(TRAINING_CONFIG.adam_beta1, TRAINING_CONFIG.adam_beta2),
            eps=TRAINING_CONFIG.adam_eps,
            weight_decay=TRAINING_CONFIG.weight_decay,
        )

        # 学习率调度器
        self.scheduler = WarmupScheduler(self.optimizer, TRANSFORMER_CONFIG.d_model, TRAINING_CONFIG.warmup_steps)

        # 损失函数
        self.criterion = LabelSmoothingLoss(
            vocab_size=TRANSFORMER_CONFIG.vocab_size_tgt, smoothing=0.1, ignore_index=TRANSFORMER_CONFIG.pad_token_id
        )

        logger.info("模型设置完成")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            训练指标字典

        数据流转：
        批次数据 -> 模型前向传播 -> 计算损失 -> 反向传播 -> 参数更新
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # 移动数据到设备
            # src: [batch_size, src_seq_len]
            # decoder_input: [batch_size, tgt_seq_len-1]
            # decoder_target: [batch_size, tgt_seq_len-1]
            src = batch["src"].to(self.device)
            decoder_input = batch["decoder_input"].to(self.device)
            decoder_target = batch["decoder_target"].to(self.device)

            batch_size, src_seq_len = src.shape
            _, tgt_seq_len = decoder_input.shape

            # 创建掩码
            # 目标序列组合掩码 [batch_size, tgt_seq_len, tgt_seq_len]
            tgt_mask = create_combined_mask(decoder_input, TRANSFORMER_CONFIG.pad_token_id).to(self.device)

            # 调试信息
            logger.debug(f"src shape: {src.shape}, decoder_input shape: {decoder_input.shape}")
            logger.debug(f"tgt_mask shape: {tgt_mask.shape}")

            # 前向传播
            self.optimizer.zero_grad()

            # logits: [batch_size, tgt_seq_len, vocab_size]
            logits = self.model(src, decoder_input, None, tgt_mask)

            # 计算损失
            loss = self.criterion(logits, decoder_target)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAINING_CONFIG.max_grad_norm)

            # 更新参数
            self.optimizer.step()
            lr = self.scheduler.step()

            # 统计
            total_loss += loss.item()
            total_tokens += (decoder_target != TRANSFORMER_CONFIG.pad_token_id).sum().item()
            self.global_step += 1

            # 记录日志
            if batch_idx % TRAINING_CONFIG.logging_steps == 0:
                avg_loss = total_loss / (batch_idx + 1)
                perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")
                elapsed = time.time() - start_time

                logger.info(
                    f"Epoch {self.current_epoch+1}, Step {batch_idx+1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, "
                    f"PPL: {perplexity:.2f}, LR: {lr:.2e}, "
                    f"Time: {elapsed:.1f}s"
                )

        # 计算平均指标
        avg_loss = total_loss / len(train_loader)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

        return {"loss": avg_loss, "perplexity": perplexity, "tokens": total_tokens}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                # 移动数据到设备
                src = batch["src"].to(self.device)
                decoder_input = batch["decoder_input"].to(self.device)
                decoder_target = batch["decoder_target"].to(self.device)

                # 创建掩码
                tgt_mask = create_combined_mask(decoder_input, TRANSFORMER_CONFIG.pad_token_id).to(self.device)

                # 前向传播
                logits = self.model(src, decoder_input, None, tgt_mask)

                # 计算损失
                loss = self.criterion(logits, decoder_target)

                # 统计
                total_loss += loss.item()
                total_tokens += (decoder_target != TRANSFORMER_CONFIG.pad_token_id).sum().item()

        # 计算平均指标
        avg_loss = total_loss / len(val_loader)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

        return {"loss": avg_loss, "perplexity": perplexity, "tokens": total_tokens}

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """
        保存检查点

        Args:
            epoch: 当前epoch
            loss: 当前损失
            is_best: 是否是最佳模型
        """
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": {"transformer": TRANSFORMER_CONFIG.model_dump(), "training": TRAINING_CONFIG.model_dump()},
        }

        # 保存当前检查点
        checkpoint_path = os.path.join(self.model_save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.model_save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")

        logger.info(f"保存检查点: {checkpoint_path}")

    def train(self) -> Dict:
        """
        主训练循环

        Returns:
            训练历史字典
        """
        logger.info("开始训练...")

        # 准备数据
        train_loader, val_loader = self.data_manager.prepare_data()

        # 设置模型
        self.setup_model()

        # 训练历史
        history = {"train_loss": [], "val_loss": [], "train_perplexity": [], "val_perplexity": []}

        # 训练循环
        for epoch in range(TRAINING_CONFIG.num_epochs):
            self.current_epoch = epoch

            logger.info(f"开始第 {epoch + 1}/{TRAINING_CONFIG.num_epochs} 轮训练")

            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证
            val_metrics = self.validate(val_loader)

            # 记录历史
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_perplexity"].append(train_metrics["perplexity"])
            history["val_perplexity"].append(val_metrics["perplexity"])

            # 日志
            logger.info(
                f"Epoch {epoch + 1} 完成 - "
                f"训练损失: {train_metrics['loss']:.4f}, "
                f"训练困惑度: {train_metrics['perplexity']:.2f}, "
                f"验证损失: {val_metrics['loss']:.4f}, "
                f"验证困惑度: {val_metrics['perplexity']:.2f}"
            )

            # 保存模型
            is_best = val_metrics["loss"] < self.best_loss
            if is_best:
                self.best_loss = val_metrics["loss"]

            self.save_checkpoint(epoch, val_metrics["loss"], is_best)

        # 保存训练历史
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"训练完成! 最佳验证损失: {self.best_loss:.4f}")
        return history

    def get_tokenizer(self):
        """获取分词器"""
        return self.data_manager.get_tokenizer()
