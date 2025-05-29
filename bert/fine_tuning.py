"""
微调模块 - BERT分类任务微调
支持从预训练模型加载权重，进行分类任务微调
重点关注微调过程的数据流转，包含详细的shape注释
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import BERT_CONFIG, TRAINING_CONFIG, DATA_CONFIG, get_device, setup_logging
from model import BertForSequenceClassification
from data_loader import load_imdb_dataset, create_classification_dataloader
from transformers import AutoTokenizer

logger = logging.getLogger("BERT")


class BertFineTuner:
    """
    BERT微调器

    负责完整的微调流程：
    1. 加载预训练模型
    2. 准备分类数据
    3. 微调训练
    4. 模型评估
    5. 模型保存
    """

    def __init__(self, pretrained_model_path: Optional[str] = None, num_labels: int = 2):
        """
        初始化微调器

        Args:
            pretrained_model_path: 预训练模型路径，如果为None则使用配置中的默认路径
            num_labels: 分类标签数量
        """
        # 设置日志
        setup_logging()

        # 设备配置
        self.device = get_device()
        logger.info(f"使用设备: {self.device}")

        # 模型配置 - 使用统一的配置管理
        if pretrained_model_path is None:
            self.pretrained_model_path = Path(TRAINING_CONFIG.pretrained_model_path)
        else:
            self.pretrained_model_path = Path(pretrained_model_path)
        self.num_labels = num_labels

        # 创建微调模型保存目录 - 使用统一的配置管理
        self.fine_tuning_save_dir = Path(TRAINING_CONFIG.fine_tuning_save_dir)
        self.fine_tuning_save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化tokenizer
        self.tokenizer = self._load_tokenizer()

        # 初始化模型
        self.model = self._create_and_load_model()

        # 初始化数据
        self.train_dataloader, self.val_dataloader = self._create_dataloaders()

        # 初始化优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_accuracy = 0.0

        # 训练历史
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "learning_rate": [],
            "epochs": [],
        }

        logger.info("微调器初始化完成")

    def _load_tokenizer(self) -> AutoTokenizer:
        """加载tokenizer"""
        logger.info(f"加载tokenizer: {DATA_CONFIG.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            DATA_CONFIG.tokenizer_name,
            cache_dir=TRAINING_CONFIG.cache_dir
        )

        # 确保有必要的特殊token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _create_and_load_model(self) -> BertForSequenceClassification:
        """
        创建并加载预训练模型

        Returns:
            BertForSequenceClassification: 分类模型
        """
        logger.info("创建分类模型...")

        # 创建分类模型
        model = BertForSequenceClassification(num_labels=self.num_labels)

        # 加载预训练权重
        if self.pretrained_model_path.exists():
            logger.info(f"加载预训练权重: {self.pretrained_model_path}")

            # 加载预训练模型的权重
            pretrained_state_dict = torch.load(self.pretrained_model_path / "pytorch_model.bin", map_location=self.device)

            # 过滤掉分类头的权重（因为预训练模型没有分类头）
            model_state_dict = model.state_dict()
            filtered_state_dict = {}

            for key, value in pretrained_state_dict.items():
                if key in model_state_dict and model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    logger.warning(f"跳过权重: {key} (shape不匹配或不存在)")

            # 加载过滤后的权重
            model.load_state_dict(filtered_state_dict, strict=False)
            logger.info(f"成功加载 {len(filtered_state_dict)} 个预训练权重")
        else:
            logger.warning(f"预训练模型路径不存在: {self.pretrained_model_path}")
            logger.warning("使用随机初始化的权重")

        model.to(self.device)

        logger.info(f"分类模型参数数量: {model.count_parameters():,}")

        return model

    def _create_dataloaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        创建训练和验证数据加载器

        Returns:
            (训练数据加载器, 验证数据加载器)
        """
        logger.info("创建分类数据加载器...")

        # 加载IMDB数据集
        texts, labels = load_imdb_dataset(TRAINING_CONFIG.max_samples)

        # 分割训练和验证集
        split_idx = int(len(texts) * 0.8)
        train_texts, val_texts = texts[:split_idx], texts[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]

        # 创建数据加载器
        train_dataloader = create_classification_dataloader(train_texts, train_labels, self.tokenizer, shuffle=True)
        val_dataloader = create_classification_dataloader(val_texts, val_labels, self.tokenizer, shuffle=False)

        logger.info(f"训练样本: {len(train_texts)}, 验证样本: {len(val_texts)}")
        logger.info(f"训练批次: {len(train_dataloader)}, 验证批次: {len(val_dataloader)}")

        return train_dataloader, val_dataloader

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        logger.info("创建微调优化器...")

        # 使用较小的学习率进行微调
        fine_tune_lr = TRAINING_CONFIG.learning_rate * 0.1  # 通常微调使用更小的学习率

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
            lr=fine_tune_lr,
            eps=TRAINING_CONFIG.adam_epsilon,
            betas=(TRAINING_CONFIG.adam_beta1, TRAINING_CONFIG.adam_beta2),
        )

        logger.info(f"微调学习率: {fine_tune_lr}")

        return optimizer

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """创建学习率调度器"""
        total_steps = len(self.train_dataloader) * TRAINING_CONFIG.num_epochs

        scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

        logger.info(f"微调调度器创建完成，总步数: {total_steps}")

        return scheduler

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        Returns:
            epoch指标字典
        """
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"微调 Epoch {self.epoch + 1}/{TRAINING_CONFIG.num_epochs}",
            leave=False,
        )

        for step, batch in enumerate(progress_bar):
            # 移动数据到设备
            # batch包含：
            # - input_ids: (batch_size, seq_len) token ids
            # - token_type_ids: (batch_size, seq_len) 句子类型ids
            # - attention_mask: (batch_size, seq_len) 注意力掩码
            # - labels: (batch_size,) 分类标签
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播
            outputs = self.model(**batch)

            # 获取损失
            loss = outputs["loss"]

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAINING_CONFIG.max_grad_norm)

            # 优化器步骤
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # 更新统计
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # 更新进度条
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

        # 计算epoch平均指标
        avg_loss = epoch_loss / num_batches

        return {
            "train_loss": avg_loss,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

    def evaluate(self) -> Dict[str, float]:
        """
        评估模型

        Returns:
            评估指标字典
        """
        self.model.eval()

        eval_loss = 0.0
        predictions = []
        true_labels = []
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="评估", leave=False):
                # 移动数据到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 前向传播
                outputs = self.model(**batch)

                # 获取损失和预测
                loss = outputs["loss"]
                logits = outputs["logits"]  # (batch_size, num_labels)

                # 计算预测标签
                preds = torch.argmax(logits, dim=-1)  # (batch_size,)

                # 收集结果
                eval_loss += loss.item()
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch["labels"].cpu().numpy())
                num_batches += 1

        # 计算指标
        avg_loss = eval_loss / num_batches
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted")

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
        }

    def fine_tune(self) -> Dict[str, Any]:
        """
        完整微调流程

        Returns:
            训练历史
        """
        logger.info("开始BERT微调")
        logger.info(f"微调轮数: {TRAINING_CONFIG.num_epochs}")
        logger.info(f"分类标签数: {self.num_labels}")

        training_start_time = time.time()

        for epoch in range(TRAINING_CONFIG.num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            # 训练一个epoch
            train_metrics = self.train_epoch()

            # 评估模型
            eval_metrics = self.evaluate()

            # 记录历史
            self.training_history["train_loss"].append(train_metrics["train_loss"])
            self.training_history["val_loss"].append(eval_metrics["val_loss"])
            self.training_history["val_accuracy"].append(eval_metrics["val_accuracy"])
            self.training_history["val_precision"].append(eval_metrics["val_precision"])
            self.training_history["val_recall"].append(eval_metrics["val_recall"])
            self.training_history["val_f1"].append(eval_metrics["val_f1"])
            self.training_history["learning_rate"].append(train_metrics["learning_rate"])
            self.training_history["epochs"].append(epoch + 1)

            # 保存最佳模型
            if eval_metrics["val_accuracy"] > self.best_accuracy:
                self.best_accuracy = eval_metrics["val_accuracy"]
                self._save_best_model()

            # 记录epoch总结
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} 完成")
            logger.info(f"训练损失: {train_metrics['train_loss']:.4f}")
            logger.info(f"验证损失: {eval_metrics['val_loss']:.4f}")
            logger.info(f"验证准确率: {eval_metrics['val_accuracy']:.4f}")
            logger.info(f"验证F1: {eval_metrics['val_f1']:.4f}")
            logger.info(f"学习率: {train_metrics['learning_rate']:.2e}")
            logger.info(f"用时: {epoch_time:.2f}秒")

        # 微调完成
        total_time = time.time() - training_start_time
        logger.info(f"微调完成，总用时: {total_time:.2f}秒")
        logger.info(f"最佳验证准确率: {self.best_accuracy:.4f}")

        # 保存最终模型
        self._save_final_model()

        # 保存训练历史
        self._save_training_history()

        return self.training_history

    def _save_best_model(self):
        """保存最佳模型"""
        best_model_dir = self.fine_tuning_save_dir / "best_model"
        best_model_dir.mkdir(exist_ok=True)

        torch.save(self.model.state_dict(), best_model_dir / "pytorch_model.bin")

        with open(best_model_dir / "config.json", "w") as f:
            config_dict = BERT_CONFIG.model_dump()
            config_dict["num_labels"] = self.num_labels
            json.dump(config_dict, f, indent=2)

        logger.info(f"保存最佳模型，准确率: {self.best_accuracy:.4f}")

    def _save_final_model(self):
        """保存最终模型"""
        final_model_dir = self.fine_tuning_save_dir / "final_model"
        final_model_dir.mkdir(exist_ok=True)

        torch.save(self.model.state_dict(), final_model_dir / "pytorch_model.bin")

        with open(final_model_dir / "config.json", "w") as f:
            config_dict = BERT_CONFIG.model_dump()
            config_dict["num_labels"] = self.num_labels
            json.dump(config_dict, f, indent=2)

        logger.info("保存最终模型")

    def _save_training_history(self):
        """保存训练历史"""
        with open(self.fine_tuning_save_dir / "fine_tuning_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

        logger.info("保存微调历史")


def fine_tune_bert(pretrained_model_path: Optional[str] = None, num_labels: int = 2) -> Dict[str, Any]:
    """
    便捷的微调函数

    Args:
        pretrained_model_path: 预训练模型路径，如果为None则使用配置中的默认路径
        num_labels: 分类标签数量

    Returns:
        训练历史
    """
    fine_tuner = BertFineTuner(pretrained_model_path, num_labels)
    history = fine_tuner.fine_tune()

    print("\n🎉 微调完成！")
    print(f"最佳准确率: {fine_tuner.best_accuracy:.4f}")
    print(f"微调模型保存目录: {fine_tuner.fine_tuning_save_dir}")

    return history


def main():
    """主微调函数"""
    import sys

    # 支持可选的预训练模型路径参数
    if len(sys.argv) > 1:
        pretrained_model_path = sys.argv[1]
        print(f"使用指定的预训练模型路径: {pretrained_model_path}")
    else:
        pretrained_model_path = None
        print(f"使用配置中的默认预训练模型路径: {TRAINING_CONFIG.pretrained_model_path}")

    history = fine_tune_bert(pretrained_model_path)
    print(f"训练历史已保存")


if __name__ == "__main__":
    main()
