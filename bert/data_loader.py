"""
数据加载器模块 - 支持预训练和微调的数据处理
重点关注mask的创建和数据流转过程，包含详细的shape注释
"""

import torch
from accelerate.commands.config.config_args import cache_dir
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import random
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm

from config import DATA_CONFIG, TRAINING_CONFIG

logger = logging.getLogger("BERT")


class BertDataCollator:
    """
    BERT数据整理器，支持MLM和NSP任务

    重点解释mask的创建逻辑：
    1. MLM掩码：随机选择15%的token进行掩码
       - 80%替换为[MASK]
       - 10%替换为随机token
       - 10%保持不变
    2. NSP标签：构造句子对，50%为连续句子，50%为随机句子
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mlm_probability = DATA_CONFIG.mlm_probability
        self.max_length = DATA_CONFIG.max_length

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        整理批次数据并应用MLM掩码和NSP标签

        Args:
            examples: 批次样本列表，每个样本包含：
                     - input_ids: (seq_len,) token ids
                     - token_type_ids: (seq_len,) 句子类型ids
                     - attention_mask: (seq_len,) 注意力掩码
                     - next_sentence_label: int NSP标签

        Returns:
            整理后的批次数据：
            - input_ids: (batch_size, max_seq_len) 掩码后的token ids
            - token_type_ids: (batch_size, max_seq_len) 句子类型ids
            - attention_mask: (batch_size, max_seq_len) 注意力掩码
            - labels: (batch_size, max_seq_len) MLM标签，-100表示不计算损失
            - next_sentence_label: (batch_size,) NSP标签

        数据流转详解：
        1. 收集批次中所有样本的各个字段
        2. 使用tokenizer的pad方法统一长度
        3. 应用MLM掩码：随机选择token进行掩码处理
        4. 创建MLM标签：只在掩码位置计算损失
        """
        batch = {}

        # 收集所有字段
        input_ids = [example["input_ids"] for example in examples]
        token_type_ids = [example["token_type_ids"] for example in examples]
        attention_mask = [example["attention_mask"] for example in examples]
        next_sentence_labels = [example["next_sentence_label"] for example in examples]

        # 填充到相同长度
        batch_encoding = self.tokenizer.pad(
            {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask},
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch["input_ids"] = batch_encoding["input_ids"]  # (batch_size, seq_len)
        batch["token_type_ids"] = batch_encoding["token_type_ids"]  # (batch_size, seq_len)
        batch["attention_mask"] = batch_encoding["attention_mask"]  # (batch_size, seq_len)
        batch["next_sentence_label"] = torch.tensor(next_sentence_labels, dtype=torch.long)  # (batch_size,)

        # 应用MLM掩码
        batch["input_ids"], batch["labels"] = self.mask_tokens(batch["input_ids"])

        return batch

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对输入应用MLM掩码 - 重点解释掩码逻辑

        Args:
            inputs: 输入token ids (batch_size, seq_len)

        Returns:
            masked_inputs: 掩码后的输入 (batch_size, seq_len)
            labels: MLM标签 (batch_size, seq_len)，-100表示不计算损失的位置
            在 data_loader.py 中，MLM标签使用 -100 表示不计算损失的位置,并非BERT论文明确规定 ，而是PyTorch框架的默认设计

        MLM掩码逻辑详解：
        1. 创建概率掩码：每个token有15%的概率被选中
        2. 排除特殊token：[CLS], [SEP], [PAD]不参与掩码
        3. 对选中的token：
           - 80%替换为[MASK] token
           - 10%替换为随机token
           - 10%保持原样（让模型学习原始分布）
        4. 创建标签：只在被选中的位置计算损失，其他位置标记为-100

        示例：
        原始: [CLS] I love cats [SEP] [PAD] [PAD]
        掩码: [CLS] I [MASK] cats [SEP] [PAD] [PAD]
        标签: [-100, -100, love, -100, -100, -100, -100]
        """
        labels = inputs.clone()  # (batch_size, seq_len)

        # 步骤1: 创建概率掩码矩阵
        probability_matrix = torch.full(labels.shape, self.mlm_probability)  # (batch_size, seq_len)

        # 步骤2: 排除特殊token，不对它们进行掩码
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)  # (batch_size, seq_len)
        for special_token_id in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
            if special_token_id is not None:
                special_tokens_mask |= labels == special_token_id

        # 将特殊token位置的概率设为0
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # 步骤3: 根据概率选择要掩码的位置
        masked_indices = torch.bernoulli(probability_matrix).bool()  # (batch_size, seq_len)

        # 步骤4: 创建标签，只在掩码位置计算损失
        labels[~masked_indices] = -100  # 非掩码位置标记为-100，CrossEntropyLoss会忽略

        # 步骤5: 应用掩码策略
        # 80%的时间用[MASK]替换
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10%的时间用随机token替换
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 剩余10%保持不变（indices_unchanged = masked_indices & ~indices_replaced & ~indices_random）

        return inputs, labels


class BertPretrainingDataset(Dataset):
    """
    BERT预训练数据集，支持MLM和NSP任务

    数据流转：
    原始文本 -> 句子分割 -> 句子对构造 -> tokenization -> 数据样本
    """

    def __init__(self, texts: List[str], tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = DATA_CONFIG.max_length
        self.nsp_probability = DATA_CONFIG.nsp_probability

        # 预处理文本，按句子分割
        self.sentences = []
        for text in texts:
            # 简单的句子分割（可以使用更复杂的方法）
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            self.sentences.extend(sentences)

        logger.info(f"创建BERT预训练数据集，文档数量: {len(texts)}, 句子数量: {len(self.sentences)}")

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本

        Returns:
            样本字典：
            - input_ids: (seq_len,) token ids
            - token_type_ids: (seq_len,) 句子类型ids，0表示句子A，1表示句子B
            - attention_mask: (seq_len,) 注意力掩码，1表示真实token，0表示padding
            - next_sentence_label: int NSP标签，1表示连续，0表示不连续

        NSP任务构造逻辑：
        1. 获取句子A（当前句子）
        2. 50%概率选择下一个句子作为句子B（正样本）
        3. 50%概率随机选择一个句子作为句子B（负样本）
        4. 使用tokenizer编码句子对，自动添加[CLS]和[SEP]
        """
        # 获取句子A
        sentence_a = self.sentences[idx]

        # 决定是否为NSP任务创建负样本
        if random.random() < self.nsp_probability:
            # 50%概率：选择下一个句子（正样本）
            if idx + 1 < len(self.sentences):
                sentence_b = self.sentences[idx + 1]
                is_next = 1
            else:
                # 如果没有下一个句子，随机选择一个
                sentence_b = random.choice(self.sentences)
                is_next = 0
        else:
            # 50%概率：随机选择一个句子（负样本）
            sentence_b = random.choice(self.sentences)
            is_next = 0

        # 编码句子对
        # tokenizer会自动添加[CLS] sentence_a [SEP] sentence_b [SEP]
        # token_type_ids会自动设置：[CLS]和sentence_a为0，sentence_b和[SEP]为1
        #  TODO 后面可以自己实现这里的功能
        encoding = self.tokenizer(
            sentence_a, sentence_b, truncation=True, max_length=self.max_length, padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # (seq_len,)
            "token_type_ids": encoding["token_type_ids"].squeeze(0),  # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (seq_len,)
            "next_sentence_label": is_next,
        }


class BertClassificationDataset(Dataset):
    """
    BERT分类数据集 - 用于微调任务

    数据流转：
    原始文本 + 标签 -> tokenization -> 数据样本
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = DATA_CONFIG.max_length

        logger.info(f"创建BERT分类数据集，样本数量: {len(texts)}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            样本字典：
            - input_ids: (seq_len,) token ids
            - token_type_ids: (seq_len,) 句子类型ids，全为0（单句）
            - attention_mask: (seq_len,) 注意力掩码
            - labels: int 分类标签

        分类任务数据处理：
        1. 对单个文本进行tokenization
        2. 自动添加[CLS]和[SEP] token
        3. token_type_ids全为0（因为是单句）
        4. 填充或截断到指定长度
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # tokenize文本
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length",
                                  return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # (seq_len,)
            "token_type_ids": encoding["token_type_ids"].squeeze(0),  # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (seq_len,)
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_wikitext_for_pretraining(max_samples: Optional[int] = None) -> List[str]:
    """
    加载WikiText数据集用于预训练

    Args:
        max_samples: 最大样本数量

    Returns:
        文本列表
    """
    logger.info(f"加载预训练数据集: {TRAINING_CONFIG.dataset_name}")

    dataset = load_dataset(
        TRAINING_CONFIG.dataset_name,
        TRAINING_CONFIG.dataset_config,
        split="train",
        cache_dir=TRAINING_CONFIG.cache_dir
    )

    texts = []
    for i, example in enumerate(tqdm(dataset, desc="加载预训练数据")):
        if max_samples and i >= max_samples:
            break

        text = example["text"].strip()
        # 过滤空文本、标题行和过短的文本
        if text and len(text) > 20 and not text.startswith("="):
            texts.append(text)

    logger.info(f"成功加载 {len(texts)} 个预训练文本样本")
    return texts


def load_imdb_dataset(max_samples: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """
    加载IMDB数据集用于分类微调

    Args:
        max_samples: 最大样本数量

    Returns:
        (文本列表, 标签列表)
    """
    logger.info("加载IMDB分类数据集")

    dataset = load_dataset("imdb", split="train", cache_dir=TRAINING_CONFIG.cache_dir)

    texts = []
    labels = []

    for i, example in enumerate(tqdm(dataset, desc="加载分类数据")):
        if max_samples and i >= max_samples:
            break

        texts.append(example["text"])
        labels.append(example["label"])

    logger.info(f"成功加载 {len(texts)} 个分类样本")
    return texts, labels


def create_pretraining_dataloader() -> Tuple[DataLoader, Any]:
    """
    创建预训练数据加载器

    Returns:
        (数据加载器, tokenizer)
    """
    # 加载tokenizer
    logger.info(f"加载tokenizer: {DATA_CONFIG.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        DATA_CONFIG.tokenizer_name,
        cache_dir=TRAINING_CONFIG.cache_dir
    )

    # 确保有必要的特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    # 加载数据
    texts = load_wikitext_for_pretraining(TRAINING_CONFIG.max_samples)

    # 创建数据集
    dataset = BertPretrainingDataset(texts, tokenizer)

    # 创建数据整理器
    data_collator = BertDataCollator(tokenizer)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG.batch_size,
        shuffle=True,
        num_workers=TRAINING_CONFIG.num_workers,
        collate_fn=data_collator,
        pin_memory=True,
    )

    logger.info(f"创建预训练数据加载器完成，批次数量: {len(dataloader)}")

    return dataloader, tokenizer


def create_classification_dataloader(texts: List[str], labels: List[int], tokenizer,
                                     shuffle: bool = True) -> DataLoader:
    """
    创建分类数据加载器

    Args:
        texts: 文本列表
        labels: 标签列表
        tokenizer: tokenizer
        shuffle: 是否打乱

    Returns:
        数据加载器
    """
    dataset = BertClassificationDataset(texts, labels, tokenizer)

    dataloader = DataLoader(dataset, batch_size=TRAINING_CONFIG.batch_size, shuffle=shuffle, pin_memory=True)

    return dataloader
