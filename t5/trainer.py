"""
T5训练器模块
负责模型训练、验证和保存
重点关注训练过程中的数据流转和性能监控
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import os
import json
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
from datetime import datetime
import numpy as np

from config import T5_CONFIG, TRAINING_CONFIG, get_device, create_directories
from model import T5ForConditionalGeneration
from data_loader import create_data_loader

logger = logging.getLogger("T5")


class T5Trainer:
    """
    T5训练器
    
    负责完整的训练流程：
    1. 模型初始化
    2. 数据加载
    3. 优化器和调度器设置
    4. 训练循环
    5. 验证和保存
    """
    
    def __init__(self):
        """初始化训练器"""
        logger.info("初始化T5训练器...")
        
        # 创建必要的目录
        create_directories()
        
        # 设备设置
        self.device = get_device()
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        logger.info("初始化T5模型...")
        self.model = T5ForConditionalGeneration()
        self.model.to(self.device)
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # 目录设置
        self.checkpoints_dir = TRAINING_CONFIG.pretrain_checkpoints_dir
        self.best_model_dir = TRAINING_CONFIG.pretrain_best_dir
        self.final_model_dir = TRAINING_CONFIG.pretrain_final_dir
        
        logger.info("T5训练器初始化完成")
    
    def setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        设置数据加载器
        
        Returns:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        logger.info("设置数据加载器...")
        
        # 训练数据加载器
        train_loader = create_data_loader(
            dataset_name=TRAINING_CONFIG.dataset_name,
            split="train",
            batch_size=TRAINING_CONFIG.batch_size,
            max_samples=TRAINING_CONFIG.max_samples,
            shuffle=True
        )
        
        # 验证数据加载器
        val_loader = create_data_loader(
            dataset_name=TRAINING_CONFIG.dataset_name,
            split="validation",
            batch_size=TRAINING_CONFIG.batch_size,
            max_samples=TRAINING_CONFIG.max_samples // 5 if TRAINING_CONFIG.max_samples else None,
            shuffle=False
        )
        
        logger.info(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
        return train_loader, val_loader
    
    def setup_optimizer_and_scheduler(self, train_loader: DataLoader) -> Tuple[AdamW, object]:
        """
        设置优化器和学习率调度器
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            optimizer: 优化器
            scheduler: 学习率调度器
        """
        logger.info("设置优化器和调度器...")
        
        # 优化器
        optimizer = AdamW(
            self.model.parameters(),
            lr=TRAINING_CONFIG.learning_rate,
            betas=(TRAINING_CONFIG.adam_beta1, TRAINING_CONFIG.adam_beta2),
            eps=TRAINING_CONFIG.adam_epsilon,
            weight_decay=TRAINING_CONFIG.weight_decay
        )
        
        # 计算总训练步数
        total_steps = len(train_loader) * TRAINING_CONFIG.num_epochs
        
        # 学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=TRAINING_CONFIG.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"总训练步数: {total_steps}, 预热步数: {TRAINING_CONFIG.warmup_steps}")
        return optimizer, scheduler
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: AdamW, 
        scheduler: object
    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            
        Returns:
            epoch_metrics: epoch指标
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}/{TRAINING_CONFIG.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(
                input_ids=batch["input_ids"],  # (batch_size, encoder_seq_len)
                attention_mask=batch["attention_mask"],  # (batch_size, encoder_seq_len)
                decoder_input_ids=batch["decoder_input_ids"],  # (batch_size, decoder_seq_len)
                decoder_attention_mask=batch["decoder_attention_mask"],  # (batch_size, decoder_seq_len)
                labels=batch["labels"]  # (batch_size, decoder_seq_len)
            )
            
            loss = outputs["loss"]
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAINING_CONFIG.max_grad_norm)
            
            # 优化器步进
            optimizer.step()
            scheduler.step()
            
            # 更新统计信息
            total_loss += loss.item()
            self.global_step += 1
            
            # 更新进度条
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # 定期日志记录
            if self.global_step % TRAINING_CONFIG.logging_steps == 0:
                logger.info(
                    f"Step {self.global_step}: loss={loss.item():.4f}, "
                    f"avg_loss={total_loss / (batch_idx + 1):.4f}, lr={current_lr:.2e}"
                )
            
            # 定期保存检查点
            if self.global_step % TRAINING_CONFIG.save_steps == 0:
                self.save_checkpoint(optimizer, scheduler)
        
        # 计算epoch平均损失
        avg_loss = total_loss / num_batches
        
        return {
            "train_loss": avg_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            val_metrics: 验证指标
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                # 数据移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs["loss"]
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        
        return {
            "val_loss": avg_loss
        }
    
    def save_checkpoint(self, optimizer: AdamW, scheduler: object):
        """
        保存检查点
        
        Args:
            optimizer: 优化器
            scheduler: 学习率调度器
        """
        checkpoint_path = os.path.join(
            self.checkpoints_dir, 
            f"checkpoint_epoch_{self.current_epoch}_step_{self.global_step}.pt"
        )
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": self.best_loss,
            "training_history": self.training_history,
            "config": {
                "t5_config": T5_CONFIG.dict(),
                "training_config": TRAINING_CONFIG.dict()
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"检查点已保存: {checkpoint_path}")
    
    def save_model(self, save_dir: str, is_best: bool = False):
        """
        保存模型
        
        Args:
            save_dir: 保存目录
            is_best: 是否为最佳模型
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型权重
        model_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)
        
        # 保存配置
        config_path = os.path.join(save_dir, "config.json")
        config_dict = {
            "t5_config": T5_CONFIG.dict(),
            "training_config": TRAINING_CONFIG.dict(),
            "training_info": {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "best_loss": self.best_loss,
                "is_best": is_best
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        # 保存训练历史
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已保存到: {save_dir}")
    
    def train(self) -> List[Dict[str, float]]:
        """
        完整训练流程
        
        Returns:
            training_history: 训练历史
        """
        logger.info("开始T5训练...")
        
        # 设置数据加载器
        train_loader, val_loader = self.setup_data_loaders()
        
        # 设置优化器和调度器
        optimizer, scheduler = self.setup_optimizer_and_scheduler(train_loader)
        
        # 训练循环
        for epoch in range(TRAINING_CONFIG.num_epochs):
            self.current_epoch = epoch
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{TRAINING_CONFIG.num_epochs}")
            logger.info(f"{'='*50}")
            
            # 训练一个epoch
            train_metrics = self.train_epoch(train_loader, optimizer, scheduler)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 合并指标
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics["epoch"] = epoch + 1
            epoch_metrics["timestamp"] = datetime.now().isoformat()
            
            # 记录训练历史
            self.training_history.append(epoch_metrics)
            
            # 日志记录
            logger.info(f"Epoch {epoch + 1} 完成:")
            logger.info(f"  训练损失: {train_metrics['train_loss']:.4f}")
            logger.info(f"  验证损失: {val_metrics['val_loss']:.4f}")
            logger.info(f"  学习率: {train_metrics['learning_rate']:.2e}")
            
            # 保存最佳模型
            if val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self.save_model(self.best_model_dir, is_best=True)
                logger.info(f"🎉 新的最佳模型! 验证损失: {self.best_loss:.4f}")
            
            # 保存检查点
            self.save_checkpoint(optimizer, scheduler)
        
        # 保存最终模型
        self.save_model(self.final_model_dir, is_best=False)
        
        logger.info("训练完成!")
        logger.info(f"最佳验证损失: {self.best_loss:.4f}")
        
        return self.training_history


if __name__ == "__main__":
    # 测试训练器
    from config import setup_logging
    
    # 设置日志
    setup_logging()
    
    # 创建训练器
    trainer = T5Trainer()
    
    # 开始训练
    history = trainer.train()
    
    logger.info("训练器测试完成!")
