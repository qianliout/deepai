"""
训练器模块 - BERT预训练训练器
包含完整的训练循环、优化器、学习率调度器等
重点关注训练过程的数据流转，包含详细的shape注释
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import time
from pathlib import Path
import numpy as np

from config import BERT_CONFIG, TRAINING_CONFIG, get_device, setup_logging
from model import BertForPreTraining
from data_loader import create_pretraining_dataloader

logger = logging.getLogger("BERT")


class BertTrainer:
    """
    BERT预训练训练器

    负责完整的预训练流程：
    1. 模型初始化
    2. 数据加载
    3. 优化器和调度器设置
    4. 训练循环
    5. 模型保存和评估
    """

    def __init__(self):
        """初始化训练器"""
        # 设置日志
        setup_logging()

        # 设备配置
        self.device = get_device()
        logger.info(f"使用设备: {self.device}")

        # 创建模型保存目录
        self.model_save_dir = Path(TRAINING_CONFIG.model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化模型
        self.model = self._create_model()

        # 初始化数据加载器
        self.train_dataloader, self.tokenizer = self._create_dataloader()

        # 初始化优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # 训练历史
        self.training_history = {
            "train_loss": [],
            "mlm_loss": [],
            "nsp_loss": [],
            "learning_rate": [],
            "steps": [],
        }

        logger.info("训练器初始化完成")

    def _create_model(self) -> BertForPreTraining:
        """
        创建BERT预训练模型

        Returns:
            BertForPreTraining: 预训练模型
        """
        logger.info("创建BERT预训练模型...")

        # 更新词汇表大小（根据实际tokenizer）
        model = BertForPreTraining()
        model.to(self.device)

        logger.info(f"模型参数数量: {model.count_parameters():,}")
        logger.info(f"模型大小: {model.count_parameters() * 4 / 1024 / 1024:.2f} MB")

        return model

    def _create_dataloader(self) -> Tuple[torch.utils.data.DataLoader, Any]:
        """
        创建数据加载器

        Returns:
            (数据加载器, tokenizer)
        """
        logger.info("创建预训练数据加载器...")

        dataloader, tokenizer = create_pretraining_dataloader()

        # 更新模型的词汇表大小
        actual_vocab_size = len(tokenizer)
        if actual_vocab_size != BERT_CONFIG.vocab_size:
            logger.info(f"更新词汇表大小: {BERT_CONFIG.vocab_size} -> {actual_vocab_size}")
            BERT_CONFIG.vocab_size = actual_vocab_size

            # 重新创建模型以匹配词汇表大小
            self.model = BertForPreTraining()
            self.model.to(self.device)

        logger.info(f"数据加载器创建完成，批次数量: {len(dataloader)}")

        return dataloader, tokenizer

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        创建优化器

        Returns:
            AdamW优化器
        """
        logger.info("创建优化器...")

        # 分离权重衰减参数
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": TRAINING_CONFIG.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=TRAINING_CONFIG.learning_rate,
            eps=TRAINING_CONFIG.adam_epsilon,
            betas=(TRAINING_CONFIG.adam_beta1, TRAINING_CONFIG.adam_beta2),
        )

        logger.info(f"AdamW优化器创建完成，学习率: {TRAINING_CONFIG.learning_rate}")

        return optimizer

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """
        创建学习率调度器

        Returns:
            学习率调度器
        """
        total_steps = len(self.train_dataloader) * TRAINING_CONFIG.num_epochs

        # 线性衰减调度器
        scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

        logger.info(f"学习率调度器创建完成，总步数: {total_steps}")

        return scheduler

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        Returns:
            epoch指标字典
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_mlm_loss = 0.0
        epoch_nsp_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"预训练 Epoch {self.epoch + 1}/{TRAINING_CONFIG.num_epochs}",
            leave=False,
        )

        for step, batch in enumerate(progress_bar):
            # 移动数据到设备
            # batch包含：
            # - input_ids: (batch_size, seq_len) 掩码后的token ids
            # - token_type_ids: (batch_size, seq_len) 句子类型ids
            # - attention_mask: (batch_size, seq_len) 注意力掩码
            # - labels: (batch_size, seq_len) MLM标签
            # - next_sentence_label: (batch_size,) NSP标签
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播
            outputs = self.model(**batch)

            # 获取损失
            total_loss = outputs["loss"]  # MLM损失 + NSP损失

            # 计算单独的损失（用于监控）
            # 上面self.model(**batch)里可能也会计算mlm_loss和nsp_loss
            # 上面 model和这里的计算方式是一致的，两处都计算了，只是工程问题，不影响理解代码
            mlm_loss = self._compute_mlm_loss(outputs, batch)
            nsp_loss = self._compute_nsp_loss(outputs, batch)

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAINING_CONFIG.max_grad_norm)

            # 优化器步骤
            # #self.optimizer.step()
            # 优化器（如 Adam、SGD 等）根据反向传播计算出的梯度（通过 loss.backward() 得到），对模型参数进行 实际更新 。
            # 这一步是模型学习的关键——通过调整参数使损失函数逐步降低。
            # 2. self.scheduler.step()
            # 学习率调度器（如余弦退火调度器、线性衰减调度器等）根据当前训练状态（如步数、epoch 数或损失值）调整优化器的 学习率。
            # 学习率是训练的核心超参数，合理调整可加速收敛或避免过拟合。
            # 执行顺序说明
            # 通常先执行 optimizer.step() 更新参数，再执行 scheduler.step() 更新学习率。这是因为学习率调度器可能依赖当前步数（如热身策略在初始阶段逐渐增加学习率），而参数更新后才需要调整下一次更新的学习率。
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # 更新统计
            epoch_loss += total_loss.item()
            epoch_mlm_loss += mlm_loss.item()
            epoch_nsp_loss += nsp_loss.item()
            num_batches += 1

            self.global_step += 1

            # 更新进度条
            progress_bar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "mlm": f"{mlm_loss.item():.4f}",
                    "nsp": f"{nsp_loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

            # 记录训练日志
            if self.global_step % TRAINING_CONFIG.logging_steps == 0:
                self._log_training_step(total_loss.item(), mlm_loss.item(), nsp_loss.item())

            # 保存检查点
            if self.global_step % TRAINING_CONFIG.save_steps == 0:
                self._save_checkpoint()

        # 计算epoch平均指标
        avg_loss = epoch_loss / num_batches
        avg_mlm_loss = epoch_mlm_loss / num_batches
        avg_nsp_loss = epoch_nsp_loss / num_batches

        return {
            "train_loss": avg_loss,
            "mlm_loss": avg_mlm_loss,
            "nsp_loss": avg_nsp_loss,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

    def _compute_mlm_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算MLM损失

        Args:
            outputs: 模型输出
            batch: 批次数据

        Returns:
            MLM损失
        """
        prediction_logits = outputs["prediction_logits"]  # (batch_size, seq_len, vocab_size)
        labels = batch["labels"]  # (batch_size, seq_len)

        loss_fct = nn.CrossEntropyLoss()
        mlm_loss = loss_fct(
            prediction_logits.view(-1, BERT_CONFIG.vocab_size),  # (batch_size * seq_len, vocab_size)
            labels.view(-1),  # (batch_size * seq_len,)
        )

        return mlm_loss

    def _compute_nsp_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算NSP损失

        Args:
            outputs: 模型输出
            batch: 批次数据

        Returns:
            NSP损失
        """
        seq_relationship_logits = outputs["seq_relationship_logits"]  # (batch_size, 2)
        next_sentence_labels = batch["next_sentence_label"]  # (batch_size,)

        loss_fct = nn.CrossEntropyLoss()
        nsp_loss = loss_fct(
            seq_relationship_logits.view(-1, 2),  # (batch_size, 2)
            next_sentence_labels.view(-1),  # (batch_size,)
        )

        return nsp_loss

    def _log_training_step(self, total_loss: float, mlm_loss: float, nsp_loss: float):
        """记录训练步骤"""
        logger.info(
            f"Step {self.global_step}: "
            f"total_loss={total_loss:.4f}, "
            f"mlm_loss={mlm_loss:.4f}, "
            f"nsp_loss={nsp_loss:.4f}, "
            f"lr={self.scheduler.get_last_lr()[0]:.2e}"
        )

    def _save_checkpoint(self):
        """保存训练检查点"""
        checkpoint_dir = self.model_save_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(exist_ok=True)

        # 保存模型状态
        torch.save(self.model.state_dict(), checkpoint_dir / "pytorch_model.bin")

        # 保存配置
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(BERT_CONFIG.model_dump(), f, indent=2)

        # 保存训练状态
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }

        torch.save(training_state, checkpoint_dir / "training_state.bin")

        logger.info(f"保存检查点: {checkpoint_dir}")

    def train(self) -> Dict[str, Any]:
        """
        完整训练流程

        Returns:
            训练历史
        """
        logger.info("开始BERT预训练")
        logger.info(f"训练轮数: {TRAINING_CONFIG.num_epochs}")
        logger.info(f"批次大小: {TRAINING_CONFIG.batch_size}")
        logger.info(f"学习率: {TRAINING_CONFIG.learning_rate}")

        training_start_time = time.time()

        for epoch in range(TRAINING_CONFIG.num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            # 训练一个epoch
            epoch_metrics = self.train_epoch()

            # 记录历史
            self.training_history["train_loss"].append(epoch_metrics["train_loss"])
            self.training_history["mlm_loss"].append(epoch_metrics["mlm_loss"])
            self.training_history["nsp_loss"].append(epoch_metrics["nsp_loss"])
            self.training_history["learning_rate"].append(epoch_metrics["learning_rate"])
            self.training_history["steps"].append(self.global_step)

            # 保存最佳模型
            if epoch_metrics["train_loss"] < self.best_loss:
                self.best_loss = epoch_metrics["train_loss"]
                self._save_best_model()

            # 记录epoch总结
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} 完成")
            logger.info(f"训练损失: {epoch_metrics['train_loss']:.4f}")
            logger.info(f"MLM损失: {epoch_metrics['mlm_loss']:.4f}")
            logger.info(f"NSP损失: {epoch_metrics['nsp_loss']:.4f}")
            logger.info(f"学习率: {epoch_metrics['learning_rate']:.2e}")
            logger.info(f"用时: {epoch_time:.2f}秒")

        # 训练完成
        total_time = time.time() - training_start_time
        logger.info(f"预训练完成，总用时: {total_time:.2f}秒")
        logger.info(f"最佳训练损失: {self.best_loss:.4f}")

        # 保存最终模型
        self._save_final_model()

        # 保存训练历史
        self._save_training_history()

        return self.training_history

    def _save_best_model(self):
        """保存最佳模型"""
        best_model_dir = self.model_save_dir / "best_model"
        best_model_dir.mkdir(exist_ok=True)

        torch.save(self.model.state_dict(), best_model_dir / "pytorch_model.bin")

        with open(best_model_dir / "config.json", "w") as f:
            json.dump(BERT_CONFIG.model_dump(), f, indent=2)

        logger.info(f"保存最佳模型，损失: {self.best_loss:.4f}")

    def _save_final_model(self):
        """保存最终模型"""
        final_model_dir = self.model_save_dir / "final_model"
        final_model_dir.mkdir(exist_ok=True)

        torch.save(self.model.state_dict(), final_model_dir / "pytorch_model.bin")

        with open(final_model_dir / "config.json", "w") as f:
            json.dump(BERT_CONFIG.model_dump(), f, indent=2)

        logger.info("保存最终模型")

    def _save_training_history(self):
        """保存训练历史"""
        with open(self.model_save_dir / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

        logger.info("保存训练历史")


def main():
    """主训练函数"""
    trainer = BertTrainer()
    history = trainer.train()

    print("\n🎉 预训练完成！")
    print(f"最佳损失: {trainer.best_loss:.4f}")
    print(f"模型保存目录: {trainer.model_save_dir}")


if __name__ == "__main__":
    main()
