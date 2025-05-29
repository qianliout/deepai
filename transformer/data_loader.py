"""
数据加载器 - 下载和处理HuggingFace数据集
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import List, Tuple, Dict, Any
import random
from config import Config, DataConfig, TrainingConfig
from tokenizer import SimpleTokenizer
from utils import setup_logging


class TranslationDataset(Dataset):
    """
    翻译数据集类
    """

    def __init__(
        self, data: List[Tuple[str, str]], tokenizer: SimpleTokenizer, max_length: int
    ):
        """
        初始化数据集

        Args:
            data: (英语, 意大利语) 句子对列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本

        Args:
            idx: 数据索引

        Returns:
            包含编码后数据的字典
        """
        en_text, it_text = self.data[idx]

        # 编码
        en_ids = self.tokenizer.encode(en_text, "en", self.max_length)
        it_ids = self.tokenizer.encode(it_text, "it", self.max_length)

        # 是否加bos，eos，已总结成笔记
        return {
            "encoder_input": torch.tensor(en_ids, dtype=torch.long),
            "decoder_input": torch.tensor(
                it_ids[:-1], dtype=torch.long
            ),  # 去掉最后的EOS
            "decoder_target": torch.tensor(
                it_ids[1:], dtype=torch.long
            ),  # 去掉开头的BOS
        }


class DataManager:
    """
    数据管理器 - 负责数据下载、预处理和加载
    """

    def __init__(self, config: Config):
        """
        初始化数据管理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.data_config = config.data
        self.training_config = config.training
        self.model_config = config.model

        self.tokenizer = SimpleTokenizer(self.model_config)
        self.logger = setup_logging(self.training_config.log_dir)

        # 数据存储
        self.train_data: List[Tuple[str, str]] = []
        self.val_data: List[Tuple[str, str]] = []

    def download_and_process_data(
        self,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        下载并处理数据集

        Returns:
            (训练数据, 验证数据)
        """
        self.logger.info("开始下载数据集...")

        # 下载数据集
        try:
            dataset = load_dataset(
                self.data_config.dataset_name,
                self.data_config.language_pair,
                cache_dir=self.training_config.cache_dir,
            )
        except Exception as e:
            self.logger.error(f"数据集下载失败: {e}")
            raise

        self.logger.info("数据集下载完成")

        # 获取训练数据
        train_dataset = dataset["train"]
        self.logger.info(f"原始训练数据大小: {len(train_dataset)}")

        # 提取文本对
        all_pairs = []
        for item in train_dataset:
            en_text = item["translation"]["en"].strip()
            it_text = item["translation"]["it"].strip()

            # 过滤空文本和过长文本
            if (
                en_text
                and it_text
                and len(en_text.split()) <= self.model_config.max_seq_len - 2
                and len(it_text.split()) <= self.model_config.max_seq_len - 2
            ):
                all_pairs.append((en_text, it_text))

        self.logger.info(f"过滤后数据大小: {len(all_pairs)}")

        # 随机打乱
        random.shuffle(all_pairs)

        # 分割训练和验证数据
        total_needed = self.training_config.train_size + self.training_config.val_size
        if len(all_pairs) < total_needed:
            self.logger.warning(f"可用数据({len(all_pairs)})少于需求({total_needed})")
            # 使用所有可用数据
            train_size = int(len(all_pairs) * 0.8)
            val_size = len(all_pairs) - train_size
        else:
            train_size = self.training_config.train_size
            val_size = self.training_config.val_size

        train_data = all_pairs[:train_size]
        val_data = all_pairs[train_size : train_size + val_size]

        self.logger.info(f"训练数据: {len(train_data)} 条")
        self.logger.info(f"验证数据: {len(val_data)} 条")

        return train_data, val_data

    def build_vocabularies(self, train_data: List[Tuple[str, str]]):
        """
        构建词典

        Args:
            train_data: 训练数据
        """
        self.logger.info("开始构建词典...")

        # 分离英语和意大利语文本
        en_texts = [pair[0] for pair in train_data]
        it_texts = [pair[1] for pair in train_data]

        # 构建词典
        self.tokenizer.vocab_en = self.tokenizer.build_vocab(
            en_texts, "en", self.data_config.min_freq, self.data_config.max_vocab_size
        )

        self.tokenizer.vocab_it = self.tokenizer.build_vocab(
            it_texts, "it", self.data_config.min_freq, self.data_config.max_vocab_size
        )

        # 构建反向词典
        self.tokenizer.id2word_en = {v: k for k, v in self.tokenizer.vocab_en.items()}
        self.tokenizer.id2word_it = {v: k for k, v in self.tokenizer.vocab_it.items()}

        # 保存词典
        self.tokenizer.save_vocabs(self.training_config.vocab_save_dir)

        self.logger.info("词典构建完成")

    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        准备训练和验证数据

        Returns:
            (训练数据加载器, 验证数据加载器)
        """
        # 检查是否已有词典
        vocab_path = self.training_config.vocab_save_dir
        en_vocab_file = os.path.join(vocab_path, "en_vocab.json")
        it_vocab_file = os.path.join(vocab_path, "it_vocab.json")

        if os.path.exists(en_vocab_file) and os.path.exists(it_vocab_file):
            self.logger.info("发现已有词典，正在加载...")
            self.tokenizer.load_vocabs(vocab_path)

            # 如果有缓存的数据，可以在这里加载
            # 为简化，这里重新下载处理
            train_data, val_data = self.download_and_process_data()
        else:
            # 下载和处理数据
            train_data, val_data = self.download_and_process_data()

            # 构建词典
            self.build_vocabularies(train_data)

        # 更新配置中的词汇表大小
        self.config.model.vocab_size_en = len(self.tokenizer.vocab_en)
        self.config.model.vocab_size_it = len(self.tokenizer.vocab_it)

        self.logger.info(f"英语词汇表大小: {self.config.model.vocab_size_en}")
        self.logger.info(f"意大利语词汇表大小: {self.config.model.vocab_size_it}")

        # 创建数据集
        train_dataset = TranslationDataset(
            train_data, self.tokenizer, self.model_config.max_seq_len
        )
        val_dataset = TranslationDataset(
            val_data, self.tokenizer, self.model_config.max_seq_len
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=0,  # Mac M1上设为0避免问题
            pin_memory=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        self.logger.info("数据加载器创建完成")

        return train_loader, val_loader

    def get_tokenizer(self) -> SimpleTokenizer:
        """
        获取分词器

        Returns:
            分词器实例
        """
        return self.tokenizer
