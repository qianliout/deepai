"""
训练器 - 负责模型训练、验证和保存
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import math
from typing import Dict, Optional
from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, get_device
from model import Transformer
from data_loader import DataManager
from utils import setup_logging, save_model, count_parameters, format_time


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数 - 防止模型过度确信，提高泛化能力

    核心思想：
    1. 传统交叉熵：正确答案概率为1.0，其他为0（硬标签）
    2. 标签平滑：正确答案概率为0.9，其他类别平均分配0.1（软标签）
    3. 优势：防止过拟合，提高训练稳定性，增强泛化能力
    """
    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = 0):
        """
        初始化标签平滑损失

        Args:
            vocab_size: 词汇表大小
            smoothing: 平滑参数（如0.1表示将0.1的概率分配给错误标签）
            ignore_index: 忽略的索引（通常是PAD token，不参与损失计算）
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing  # 标签平滑参数，决定“软标签”分配给非目标类别的概率总和（如0.1）
        self.ignore_index = ignore_index  # 忽略的标签索引（通常是PAD token的ID），这些位置不参与损失计算
        # 正确标签的置信度 = 1.0 - smoothing
        # 例如 smoothing=0.1 时，正确类别的概率为0.9，其余类别平分0.1
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算标签平滑损失

        Args:
            pred: 预测logits [batch_size, seq_len, vocab_size]
            target: 目标标签 [batch_size, seq_len]

        Returns:
            损失值（标量）
        """
        batch_size, seq_len, vocab_size = pred.size()

        # 1. 重塑张量为二维，方便后续处理
        # pred: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        # target: [batch_size, seq_len] -> [batch_size * seq_len]
        pred = pred.view(-1, vocab_size)
        target = target.view(-1)

        # 2. 创建平滑标签分布
        # 初始化所有类别的概率为 smoothing / (vocab_size - 2)
        # -2 是因为要排除PAD token和目标token本身
        true_dist = torch.zeros_like(pred)
        smooth_prob = self.smoothing / (vocab_size - 2)
        true_dist.fill_(smooth_prob)

        # 3. 设置正确标签的置信度（如0.9）
        # scatter_ 在每一行的 target 指定的位置填入 self.confidence
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # 4. 将PAD token的概率设为0（不参与损失计算）
        true_dist[:, self.ignore_index] = 0

        # 5. 创建掩码，忽略PAD token
        # mask: 1表示有效token，0表示PAD token
        mask = (target != self.ignore_index).float()

        # 6. 计算KL散度
        # KL(P||Q) = Σ P * (log(P) - log(Q))
        # 这里P为true_dist（平滑标签），Q为softmax(pred)（模型预测概率）
        # torch.log_softmax(pred, dim=1)：对预测logits做softmax再取对数
        # torch.log(true_dist + 1e-12)：对平滑标签取对数（加1e-12防止log(0)）
        kl_div = torch.sum(
            true_dist * (torch.log(true_dist + 1e-12) - torch.log_softmax(pred, dim=1)),
            dim=1,  # 在词汇表维度上求和
        )

        # 7. 应用掩码并计算平均损失
        # 只统计非PAD token的损失
        loss = torch.sum(kl_div * mask) / torch.sum(mask)

        return loss


class WarmupScheduler:
    """
    学习率预热调度器
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

    def step(self):
        """更新学习率"""
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5)
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


class Trainer:
    """
    训练器类
    """

    def __init__(self):
        """
        初始化训练器
        """
        self.device = get_device()
        self.logger = setup_logging(TRAINING_CONFIG.log_dir)

        # 数据管理器
        self.data_manager = DataManager()

        # 模型
        self.model: Optional[Transformer] = None
        self.optimizer: Optional[optim.Adam] = None
        self.scheduler: Optional[WarmupScheduler] = None
        self.criterion: Optional[LabelSmoothingLoss] = None

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        self.logger.info(f"使用设备: {self.device}")

    def setup_model(self):
        """设置模型和优化器"""
        self.logger.info("正在设置模型...")

        # 创建模型
        self.model = Transformer(MODEL_CONFIG).to(self.device)

        # 计算参数数量
        param_count = count_parameters(self.model)
        self.logger.info(f"模型参数数量: {param_count:,}")

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # 学习率调度器
        self.scheduler = WarmupScheduler(
            self.optimizer, MODEL_CONFIG.d_model, TRAINING_CONFIG.warmup_steps
        )

        # 损失函数
        self.criterion = LabelSmoothingLoss(
            vocab_size=MODEL_CONFIG.vocab_size_it,
            smoothing=0.1,
            ignore_index=0,  # PAD token
        )

        self.logger.info("模型设置完成")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # 移动数据到设备
            # encoder_input: [batch_size, src_seq_len] - 编码器输入token序列
            # decoder_input: [batch_size, tgt_seq_len-1] - 解码器输入序列
            # decoder_target: [batch_size, tgt_seq_len-1] - 解码器目标序列
            encoder_input = batch["encoder_input"].to(self.device)
            decoder_input = batch["decoder_input"].to(self.device)
            decoder_target = batch["decoder_target"].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            # output: [batch_size, tgt_seq_len-1, vocab_size_it] - 模型输出logits
            output = self.model(encoder_input, decoder_input)

            # 计算损失
            # output: [batch_size, tgt_seq_len-1, vocab_size_it]
            # decoder_target: [batch_size, tgt_seq_len-1]
            # loss: scalar - 交叉熵损失
            loss = self.criterion(output, decoder_target)

            # 反向传播
            loss.backward()

            # 梯度裁剪 - 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()
            lr = self.scheduler.step()

            # 统计
            # 统计当前批次中有效token的数量（排除填充token）
            # decoder_target: [batch_size, tgt_seq_len-1] -> batch_tokens: scalar
            # 示例：若decoder_target为[[1, 2, 0, 0], [3, 4, 5, 0]]（0是PAD），则结果为5
            batch_tokens = (decoder_target != 0).sum().item()
            # loss.item(): 当前批次的平均损失（每个token的损失均值）
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            self.global_step += 1

            # 日志
            if batch_idx % TRAINING_CONFIG.log_interval == 0:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, LR: {lr:.2e}, "
                    f"Time: {format_time(elapsed)}"
                )

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}

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
                encoder_input = batch["encoder_input"].to(self.device)
                decoder_input = batch["decoder_input"].to(self.device)
                decoder_target = batch["decoder_target"].to(self.device)

                # 前向传播
                output = self.model(encoder_input, decoder_input)

                # 计算损失
                loss = self.criterion(output, decoder_target)

                # 统计
                batch_tokens = (decoder_target != 0).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}

    def train(self):
        """主训练循环"""
        self.logger.info("开始训练...")

        # 准备数据
        train_loader, val_loader = self.data_manager.prepare_data()

        # 设置模型
        self.setup_model()

        # 创建保存目录
        os.makedirs(TRAINING_CONFIG.pretrain_checkpoints_dir, exist_ok=True)
        os.makedirs(TRAINING_CONFIG.pretrain_best_dir, exist_ok=True)
        os.makedirs(TRAINING_CONFIG.pretrain_final_dir, exist_ok=True)

        # 训练循环
        for epoch in range(TRAINING_CONFIG.num_epochs):
            self.current_epoch = epoch

            self.logger.info(
                f"开始第 {epoch + 1}/{TRAINING_CONFIG.num_epochs} 轮训练"
            )

            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证
            val_metrics = self.validate(val_loader)

            # 日志
            self.logger.info(
                f"Epoch {epoch + 1} 完成 - "
                f"训练损失: {train_metrics['loss']:.4f}, "
                f"训练困惑度: {train_metrics['perplexity']:.2f}, "
                f"验证损失: {val_metrics['loss']:.4f}, "
                f"验证困惑度: {val_metrics['perplexity']:.2f}"
            )

            # 保存最佳模型
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                model_path = os.path.join(
                    TRAINING_CONFIG.pretrain_best_dir,
                    f"best_model_epoch_{epoch + 1}.pt",
                )
                save_model(
                    self.model, self.optimizer, epoch, val_metrics["loss"], model_path
                )
                self.logger.info(f"保存最佳模型: {model_path}")

            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    TRAINING_CONFIG.pretrain_checkpoints_dir,
                    f"checkpoint_epoch_{epoch + 1}.pt",
                )
                save_model(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics["loss"],
                    checkpoint_path,
                )

        # 保存最终模型
        final_model_path = os.path.join(
            TRAINING_CONFIG.pretrain_final_dir,
            "final_model.pt",
        )
        save_model(
            self.model, self.optimizer, TRAINING_CONFIG.num_epochs - 1, self.best_val_loss, final_model_path
        )
        self.logger.info(f"保存最终模型: {final_model_path}")

        self.logger.info("训练完成!")

    def load_model(self, model_path: str):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 先加载词汇表以获取正确的词汇表大小
        vocab_path = TRAINING_CONFIG.pretrain_vocab_dir
        en_vocab_file = os.path.join(vocab_path, "en_vocab.json")
        it_vocab_file = os.path.join(vocab_path, "it_vocab.json")

        if os.path.exists(en_vocab_file) and os.path.exists(it_vocab_file):
            self.logger.info("加载词汇表...")
            self.data_manager.tokenizer.load_vocabs(vocab_path)

            # 更新配置中的词汇表大小
            MODEL_CONFIG.vocab_size_en = len(self.data_manager.tokenizer.vocab_en)
            MODEL_CONFIG.vocab_size_it = len(self.data_manager.tokenizer.vocab_it)

            self.logger.info(f"英语词汇表大小: {MODEL_CONFIG.vocab_size_en}")
            self.logger.info(f"意大利语词汇表大小: {MODEL_CONFIG.vocab_size_it}")
        else:
            self.logger.warning("未找到词汇表文件，使用默认配置")

        # 如果模型还没有初始化，先初始化
        if self.model is None:
            self.setup_model()

        # 加载模型状态
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.logger.info(f"成功加载模型: {model_path}")

    def translate(self, text: str, max_length: int = 50) -> str:
        """
        翻译单个句子 - 使用贪心搜索进行推理

        Args:
            text: 输入的英语文本
            max_length: 最大生成长度

        Returns:
            翻译后的意大利语文本
        """
        if self.model is None:
            raise RuntimeError("模型未初始化，请先训练或加载模型")

        self.model.eval()
        tokenizer = self.data_manager.get_tokenizer()

        with torch.no_grad():
            # 编码输入文本
            # src_tokens: [src_seq_len] - 英语token ID序列
            src_tokens = tokenizer.encode(text, 'en', MODEL_CONFIG.max_seq_len)
            # src: [1, src_seq_len] - 添加batch维度
            src = torch.tensor([src_tokens], device=self.device)

            # 编码阶段
            # encoder_output: [1, src_seq_len, d_model] - 编码器输出表示
            encoder_output = self.model.encode(src)

            # 解码阶段 - 贪心搜索
            # tgt: [1, 1] - 初始化为BOS token
            tgt = torch.tensor([[tokenizer.bos_id]], device=self.device)

            for _ in range(max_length):
                # 解码一步
                # output: [1, current_tgt_len, vocab_size_it] - 当前步的输出logits
                output = self.model.decode_step(tgt, encoder_output)

                # 贪心选择下一个token
                # output[:, -1, :]: [1, vocab_size_it] - 最后一个位置的logits
                # next_token: [1, 1] - 概率最大的token ID
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

                # 拼接新token
                # tgt: [1, current_tgt_len] -> [1, current_tgt_len+1]
                tgt = torch.cat([tgt, next_token], dim=1)

                # 如果生成了EOS token，停止生成
                if next_token.item() == tokenizer.eos_id:
                    break

            # 解码输出为文本
            # tgt_tokens: [tgt_seq_len] - 生成的token ID序列
            tgt_tokens = tgt[0].cpu().tolist()
            translation = tokenizer.decode(tgt_tokens, 'it')

            return translation

    def get_tokenizer(self):
        """获取分词器"""
        return self.data_manager.get_tokenizer()
