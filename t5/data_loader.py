"""
T5数据加载模块
负责数据集加载、预处理和批次生成
重点关注数据的流转和shape变化，包含详细的shape注释
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer,T5TokenizerFast
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple
import logging
from pydantic import BaseModel

from config import DATA_CONFIG, TRAINING_CONFIG, T5_CONFIG

logger = logging.getLogger("T5")


class T5DataSample(BaseModel):
    """T5数据样本的数据结构定义"""
    
    input_ids: List[int]  # 编码器输入token ids
    attention_mask: List[int]  # 编码器注意力掩码
    decoder_input_ids: List[int]  # 解码器输入token ids
    decoder_attention_mask: List[int]  # 解码器注意力掩码
    labels: List[int]  # 标签（用于计算损失）
    
    class Config:
        extra = "forbid"


class T5Dataset(Dataset):
    """
    T5数据集类
    
    数据流：
    原始文本 -> tokenizer -> T5DataSample -> tensor
    """
    
    def __init__(self, dataset_name: str = "squad", split: str = "train", max_samples: Optional[int] = None):
        """
        初始化数据集
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割（train/validation/test）
            max_samples: 最大样本数（用于快速测试）
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        
        # 初始化tokenizer
        logger.info(f"加载tokenizer: {DATA_CONFIG.tokenizer_name}")
        try:
            # 首先尝试使用本地文件
            self.tokenizer = T5Tokenizer.from_pretrained(
                DATA_CONFIG.tokenizer_name,
                cache_dir=TRAINING_CONFIG.cache_dir,
                local_files_only=True
            )
        except Exception as e:
            logger.info(f"本地文件不存在，从网络下载: {e}")
            # 如果本地文件不存在，从网络下载
            self.tokenizer = T5Tokenizer.from_pretrained(
                DATA_CONFIG.tokenizer_name,
                cache_dir=TRAINING_CONFIG.cache_dir,
                local_files_only=False
            )
        
        # 加载数据集
        logger.info(f"加载数据集: {dataset_name}, split: {split}")
        try:
            if dataset_name == "cnn_dailymail":
                self.dataset = load_dataset(
                    dataset_name,
                    TRAINING_CONFIG.dataset_config,
                    cache_dir=TRAINING_CONFIG.cache_dir
                )[split]
            else:
                self.dataset = load_dataset(
                    dataset_name,
                    cache_dir=TRAINING_CONFIG.cache_dir
                )[split]
        except Exception as e:
            logger.warning(f"加载数据集失败: {e}")
            logger.info("使用简单的测试数据集")
            # 创建一个简单的测试数据集
            self.dataset = self._create_simple_dataset()
        
        # 限制样本数量（用于快速测试）
        if max_samples is not None and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))
            logger.info(f"限制样本数量为: {max_samples}")
        
        logger.info(f"数据集加载完成，样本数量: {len(self.dataset)}")
        
        # 预处理数据
        self.processed_data = self._preprocess_data()

    def _create_simple_dataset(self):
        """创建简单的测试数据集"""
        simple_data = [
            {
                "input": "Hello world",
                "target": "Hallo Welt"
            },
            {
                "input": "How are you?",
                "target": "Wie geht es dir?"
            },
            {
                "input": "Good morning",
                "target": "Guten Morgen"
            },
            {
                "input": "Thank you",
                "target": "Danke"
            },
            {
                "input": "Good night",
                "target": "Gute Nacht"
            }
        ]

        # 重复数据以达到所需的样本数量
        if self.max_samples:
            repeat_times = max(1, self.max_samples // len(simple_data))
            simple_data = simple_data * repeat_times
            simple_data = simple_data[:self.max_samples]

        return simple_data
    
    def _preprocess_data(self) -> List[T5DataSample]:
        """
        预处理数据
        
        Returns:
            processed_data: 预处理后的数据列表
        """
        logger.info("开始预处理数据...")
        processed_data = []
        
        for i, example in enumerate(self.dataset):
            try:
                # 根据数据集类型进行不同的预处理
                if self.dataset_name == "squad" or self.dataset_name == "rajpurkar/squad":
                    sample = self._preprocess_squad_example(example)
                elif self.dataset_name == "cnn_dailymail":
                    sample = self._preprocess_cnn_dailymail_example(example)
                else:
                    # 默认处理方式（包括简单测试数据集）
                    sample = self._preprocess_default_example(example)

                processed_data.append(sample)
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"已预处理 {i + 1}/{len(self.dataset)} 个样本")
                    
            except Exception as e:
                logger.warning(f"预处理第 {i} 个样本时出错: {e}")
                continue
        
        logger.info(f"数据预处理完成，有效样本数量: {len(processed_data)}")
        return processed_data
    
    def _preprocess_squad_example(self, example: Dict) -> T5DataSample:
        """
        预处理SQuAD样本
        
        Args:
            example: 原始样本
            
        Returns:
            sample: 预处理后的样本
            
        数据流转：
        原始文本 -> 添加任务前缀 -> tokenize -> 截断/填充 -> T5DataSample
        """
        # 构建输入文本（问题 + 上下文）
        question = example["question"]
        context = example["context"]
        input_text = f"question: {question} context: {context}"
        
        # 构建目标文本（答案）
        if "answers" in example and len(example["answers"]["text"]) > 0:
            target_text = example["answers"]["text"][0]
        else:
            target_text = "无答案"
        
        # Tokenize输入
        input_encoding = self.tokenizer(
            input_text,
            max_length=DATA_CONFIG.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize目标
        target_encoding = self.tokenizer(
            target_text,
            max_length=DATA_CONFIG.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 准备解码器输入（目标序列向右移动一位）
        decoder_input_ids = target_encoding["input_ids"].clone()
        decoder_input_ids[:, 1:] = target_encoding["input_ids"][:, :-1]
        decoder_input_ids[:, 0] = T5_CONFIG.decoder_start_token_id
        
        # 准备标签（忽略padding token）
        labels = target_encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return T5DataSample(
            input_ids=input_encoding["input_ids"].squeeze().tolist(),
            attention_mask=input_encoding["attention_mask"].squeeze().tolist(),
            decoder_input_ids=decoder_input_ids.squeeze().tolist(),
            decoder_attention_mask=target_encoding["attention_mask"].squeeze().tolist(),
            labels=labels.squeeze().tolist()
        )

    def _preprocess_cnn_dailymail_example(self, example: Dict) -> T5DataSample:
        """
        预处理CNN/DailyMail样本（摘要任务）

        Args:
            example: 原始样本

        Returns:
            sample: 预处理后的样本
        """
        # 构建输入文本（文章）
        article = example["article"]
        input_text = f"summarize: {article}"

        # 构建目标文本（摘要）
        target_text = example["highlights"]

        # Tokenize输入
        input_encoding = self.tokenizer(
            input_text,
            max_length=DATA_CONFIG.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize目标
        target_encoding = self.tokenizer(
            target_text,
            max_length=DATA_CONFIG.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 准备解码器输入
        decoder_input_ids = target_encoding["input_ids"].clone()
        decoder_input_ids[:, 1:] = target_encoding["input_ids"][:, :-1]
        decoder_input_ids[:, 0] = T5_CONFIG.decoder_start_token_id

        # 准备标签
        labels = target_encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return T5DataSample(
            input_ids=input_encoding["input_ids"].squeeze().tolist(),
            attention_mask=input_encoding["attention_mask"].squeeze().tolist(),
            decoder_input_ids=decoder_input_ids.squeeze().tolist(),
            decoder_attention_mask=target_encoding["attention_mask"].squeeze().tolist(),
            labels=labels.squeeze().tolist()
        )
    
    def _preprocess_default_example(self, example: Dict) -> T5DataSample:
        """
        默认预处理方式
        
        Args:
            example: 原始样本
            
        Returns:
            sample: 预处理后的样本
        """
        # 简单的文本到文本任务
        input_text = str(example.get("input", ""))
        target_text = str(example.get("target", ""))
        
        # 添加任务前缀
        input_text = DATA_CONFIG.task_prefix + input_text
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            max_length=DATA_CONFIG.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=DATA_CONFIG.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 准备解码器输入
        decoder_input_ids = target_encoding["input_ids"].clone()
        decoder_input_ids[:, 1:] = target_encoding["input_ids"][:, :-1]
        decoder_input_ids[:, 0] = T5_CONFIG.decoder_start_token_id
        
        # 准备标签
        labels = target_encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return T5DataSample(
            input_ids=input_encoding["input_ids"].squeeze().tolist(),
            attention_mask=input_encoding["attention_mask"].squeeze().tolist(),
            decoder_input_ids=decoder_input_ids.squeeze().tolist(),
            decoder_attention_mask=target_encoding["attention_mask"].squeeze().tolist(),
            labels=labels.squeeze().tolist()
        )
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> T5DataSample:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            sample: T5数据样本
        """
        return self.processed_data[idx]


def collate_fn(batch: List[T5DataSample]) -> Dict[str, torch.Tensor]:
    """
    批次整理函数
    
    Args:
        batch: T5DataSample列表
        
    Returns:
        batch_dict: 批次数据字典
        
    数据流转：
    List[T5DataSample] -> Dict[str, torch.Tensor]
    每个字段从 List[List[int]] -> torch.Tensor (batch_size, seq_len)
    """
    # 提取各个字段
    input_ids = [torch.tensor(sample.input_ids) for sample in batch]
    attention_mask = [torch.tensor(sample.attention_mask) for sample in batch]
    decoder_input_ids = [torch.tensor(sample.decoder_input_ids) for sample in batch]
    decoder_attention_mask = [torch.tensor(sample.decoder_attention_mask) for sample in batch]
    labels = [torch.tensor(sample.labels) for sample in batch]
    
    # 堆叠成批次
    batch_dict = {
        "input_ids": torch.stack(input_ids),  # (batch_size, encoder_seq_len)
        "attention_mask": torch.stack(attention_mask),  # (batch_size, encoder_seq_len)
        "decoder_input_ids": torch.stack(decoder_input_ids),  # (batch_size, decoder_seq_len)
        "decoder_attention_mask": torch.stack(decoder_attention_mask),  # (batch_size, decoder_seq_len)
        "labels": torch.stack(labels),  # (batch_size, decoder_seq_len)
    }
    
    return batch_dict


def create_data_loader(
    dataset_name: str = "squad",
    split: str = "train",
    batch_size: Optional[int] = None,
    max_samples: Optional[int] = None,
    shuffle: bool = True
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset_name: 数据集名称
        split: 数据集分割
        batch_size: 批次大小
        max_samples: 最大样本数
        shuffle: 是否打乱数据
        
    Returns:
        data_loader: 数据加载器
    """
    if batch_size is None:
        batch_size = TRAINING_CONFIG.batch_size
    
    if max_samples is None:
        max_samples = TRAINING_CONFIG.max_samples
    
    # 创建数据集
    dataset = T5Dataset(
        dataset_name=dataset_name,
        split=split,
        max_samples=max_samples
    )
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=TRAINING_CONFIG.num_workers,
        collate_fn=collate_fn,
        pin_memory=True  # 加速GPU传输
    )
    
    logger.info(f"数据加载器创建完成，批次数量: {len(data_loader)}")
    return data_loader


if __name__ == "__main__":
    # 测试数据加载器
    logger.info("测试T5数据加载器...")
    
    # 创建数据加载器
    train_loader = create_data_loader(
        dataset_name="squad",
        split="train",
        batch_size=2,
        max_samples=10,
        shuffle=False
    )
    
    # 测试一个批次
    for batch in train_loader:
        logger.info("批次数据shape:")
        for key, value in batch.items():
            logger.info(f"  {key}: {value.shape}")
        
        logger.info("样本数据:")
        logger.info(f"  input_ids[0][:10]: {batch['input_ids'][0][:10].tolist()}")
        logger.info(f"  labels[0][:10]: {batch['labels'][0][:10].tolist()}")
        break
    
    logger.info("数据加载器测试完成！")
