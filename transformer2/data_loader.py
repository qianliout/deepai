"""
数据加载器模块 - 重构版本
处理数据集加载、预处理和批次生成
详细的数据流转注释和shape说明
"""

import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Tuple, Dict, Optional
from collections import Counter
import json
import os
from datasets import load_dataset

from config import TRANSFORMER_CONFIG, TRAINING_CONFIG, DATA_CONFIG

logger = logging.getLogger("Transformer2")


class SimpleTokenizer:
    """简单的分词器

    基于空格分词，构建词汇表，处理特殊token

    数据流转：
    原始文本 -> 分词 -> token列表 -> ID序列 -> 填充/截断 -> 最终序列
    """

    def __init__(self):
        """初始化分词器"""
        self.vocab_src = {}  # 源语言词汇表 {token: id}
        self.vocab_tgt = {}  # 目标语言词汇表 {token: id}
        self.id2token_src = {}  # 源语言ID到token映射 {id: token}
        self.id2token_tgt = {}  # 目标语言ID到token映射 {id: token}

        # 特殊token
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        logger.info("简单分词器初始化完成")

    def build_vocab(self, texts_src: List[str], texts_tgt: List[str]) -> None:
        """
        构建词汇表

        Args:
            texts_src: 源语言文本列表
            texts_tgt: 目标语言文本列表

        数据流转：
        文本列表 -> 分词 -> 统计词频 -> 过滤低频词 -> 构建词汇表
        """
        logger.info("开始构建词汇表...")

        # 构建源语言词汇表
        self._build_single_vocab(texts_src, "src")

        # 构建目标语言词汇表
        self._build_single_vocab(texts_tgt, "tgt")

        logger.info(f"词汇表构建完成:")
        logger.info(f"  - 源语言词汇表大小: {len(self.vocab_src)}")
        logger.info(f"  - 目标语言词汇表大小: {len(self.vocab_tgt)}")

    def _build_single_vocab(self, texts: List[str], lang: str) -> None:
        """
        构建单语言词汇表

        Args:
            texts: 文本列表
            lang: 语言标识 ('src' 或 'tgt')
        """
        # 统计词频
        word_counts = Counter()
        for text in texts:
            tokens = text.lower().split()
            word_counts.update(tokens)

        # 创建词汇表
        vocab = {}
        id2token = {}

        # 添加特殊token
        special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]
        for i, token in enumerate(special_tokens):
            vocab[token] = i
            id2token[i] = token

        # 添加常用词
        idx = len(special_tokens)
        for word, count in word_counts.most_common():
            if count >= DATA_CONFIG.min_freq:
                vocab[word] = idx
                id2token[idx] = word
                idx += 1

                # 限制词汇表大小
                max_vocab = TRANSFORMER_CONFIG.vocab_size_src if lang == "src" else TRANSFORMER_CONFIG.vocab_size_tgt
                if idx >= max_vocab:
                    break

        # 保存词汇表
        if lang == "src":
            self.vocab_src = vocab
            self.id2token_src = id2token
        else:
            self.vocab_tgt = vocab
            self.id2token_tgt = id2token

        logger.info(f"{lang}语言词汇表构建完成，大小: {len(vocab)}")

    def encode(self, text: str, lang: str, max_length: int, pad_to_max: bool = True) -> List[int]:
        """
        编码文本为ID序列

        Args:
            text: 输入文本
            lang: 语言标识 ('src' 或 'tgt')
            max_length: 最大长度
            pad_to_max: 是否填充到最大长度

        Returns:
            ID序列 [seq_len] (如果pad_to_max=True则填充到max_length)

        数据流转：
        文本 -> 分词 -> token列表 -> 添加特殊token -> ID序列 -> 填充/截断
        """
        vocab = self.vocab_src if lang == "src" else self.vocab_tgt
        unk_id = vocab[self.unk_token]
        pad_id = vocab[self.pad_token]
        bos_id = vocab[self.bos_token]
        eos_id = vocab[self.eos_token]

        # 分词
        tokens = text.lower().split()

        # 转换为ID
        ids = [vocab.get(token, unk_id) for token in tokens]

        # 添加BOS和EOS token
        if lang == "tgt":
            ids = [bos_id] + ids + [eos_id]
        else:
            ids = ids + [eos_id]

        # 截断
        if len(ids) > max_length:
            ids = ids[:max_length]

        # 填充（如果需要）
        if pad_to_max and len(ids) < max_length:
            ids.extend([pad_id] * (max_length - len(ids)))

        return ids

    def decode(self, ids: List[int], lang: str) -> str:
        """
        解码ID序列为文本

        Args:
            ids: ID序列
            lang: 语言标识 ('src' 或 'tgt')

        Returns:
            解码后的文本
        """
        id2token = self.id2token_src if lang == "src" else self.id2token_tgt

        tokens = []
        for id in ids:
            token = id2token.get(id, self.unk_token)
            if token in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            tokens.append(token)

        return " ".join(tokens)

    def save_vocab(self, save_dir: str) -> None:
        """保存词汇表到文件"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存源语言词汇表
        with open(os.path.join(save_dir, "vocab_src.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab_src, f, ensure_ascii=False, indent=2)

        # 保存目标语言词汇表
        with open(os.path.join(save_dir, "vocab_tgt.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab_tgt, f, ensure_ascii=False, indent=2)

        logger.info(f"词汇表已保存到: {save_dir}")

    def load_vocab(self, save_dir: str) -> None:
        """从文件加载词汇表"""
        # 加载源语言词汇表
        with open(os.path.join(save_dir, "vocab_src.json"), "r", encoding="utf-8") as f:
            self.vocab_src = json.load(f)

        # 加载目标语言词汇表
        with open(os.path.join(save_dir, "vocab_tgt.json"), "r", encoding="utf-8") as f:
            self.vocab_tgt = json.load(f)

        # 构建反向映射
        self.id2token_src = {v: k for k, v in self.vocab_src.items()}
        self.id2token_tgt = {v: k for k, v in self.vocab_tgt.items()}

        logger.info(f"词汇表已从 {save_dir} 加载")


class TranslationDataset(Dataset):
    """翻译数据集

    处理源语言-目标语言对，返回训练所需的tensor

    数据流转：
    (src_text, tgt_text) -> 分词编码 -> (src_ids, tgt_ids) ->
    创建decoder输入和目标 -> (src_tensor, decoder_input, decoder_target)
    """

    def __init__(self, data_pairs: List[Tuple[str, str]], tokenizer: SimpleTokenizer):
        """
        初始化数据集

        Args:
            data_pairs: (源文本, 目标文本) 对列表
            tokenizer: 分词器
        """
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer
        self.max_length = DATA_CONFIG.max_length

        logger.info(f"翻译数据集初始化完成，样本数量: {len(data_pairs)}")

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            包含以下键的字典:
            - src: 源序列 [max_length]
            - decoder_input: 解码器输入 [max_length] (目标序列，不包含最后一个token)
            - decoder_target: 解码器目标 [max_length] (目标序列，不包含第一个token)

        数据流转：
        (src_text, tgt_text) -> 编码 -> (src_ids, tgt_ids) ->
        创建decoder_input和decoder_target -> 转换为tensor
        """
        src_text, tgt_text = self.data_pairs[idx]

        # 编码源序列
        src_ids = self.tokenizer.encode(src_text, "src", self.max_length)

        # 编码目标序列
        tgt_ids = self.tokenizer.encode(tgt_text, "tgt", self.max_length)

        # 创建解码器输入和目标
        # decoder_input: [BOS, token1, token2, ..., tokenN-1]
        # decoder_target: [token1, token2, ..., tokenN, EOS]
        decoder_input = tgt_ids[:-1]  # 去掉最后一个token
        decoder_target = tgt_ids[1:]  # 去掉第一个token (BOS)

        # 确保长度一致，截断到max_length-1
        if len(decoder_input) > self.max_length - 1:
            decoder_input = decoder_input[: self.max_length - 1]
            decoder_target = decoder_target[: self.max_length - 1]
        elif len(decoder_input) < self.max_length - 1:
            pad_id = self.tokenizer.vocab_tgt[self.tokenizer.pad_token]
            decoder_input.extend([pad_id] * (self.max_length - 1 - len(decoder_input)))
            decoder_target.extend([pad_id] * (self.max_length - 1 - len(decoder_target)))

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),  # [max_length]
            "decoder_input": torch.tensor(decoder_input, dtype=torch.long),  # [max_length-1]
            "decoder_target": torch.tensor(decoder_target, dtype=torch.long),  # [max_length-1]
        }


class DataManager:
    """数据管理器

    负责数据集下载、预处理、分词器构建和数据加载器创建
    """

    def __init__(self):
        """初始化数据管理器"""
        self.tokenizer = SimpleTokenizer()
        self.train_data = None
        self.val_data = None

        logger.info("数据管理器初始化完成")

    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        准备训练和验证数据

        Returns:
            (train_loader, val_loader): 训练和验证数据加载器

        数据流转：
        下载数据集 -> 预处理 -> 构建词汇表 -> 创建数据集 -> 创建数据加载器
        """
        logger.info("开始准备数据...")

        # 1. 下载和预处理数据
        train_pairs, val_pairs = self._load_and_preprocess_data()

        # 2. 构建词汇表
        self._build_tokenizer(train_pairs)

        # 3. 创建数据集
        train_dataset = TranslationDataset(train_pairs, self.tokenizer)
        val_dataset = TranslationDataset(val_pairs, self.tokenizer)

        # 4. 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAINING_CONFIG.batch_size,
            shuffle=True,
            num_workers=TRAINING_CONFIG.num_workers,
            pin_memory=False,  # Mac M1建议设为False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAINING_CONFIG.batch_size,
            shuffle=False,
            num_workers=TRAINING_CONFIG.num_workers,
            pin_memory=False,
        )

        logger.info(f"数据准备完成:")
        logger.info(f"  - 训练样本: {len(train_dataset)}")
        logger.info(f"  - 验证样本: {len(val_dataset)}")
        logger.info(f"  - 批次大小: {TRAINING_CONFIG.batch_size}")

        return train_loader, val_loader

    def _load_and_preprocess_data(
        self,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        加载和预处理数据集

        Returns:
            (train_pairs, val_pairs): 训练和验证数据对
        """
        logger.info(f"正在下载数据集: {TRAINING_CONFIG.dataset_name}")

        try:
            # 下载数据集
            dataset = load_dataset(
                TRAINING_CONFIG.dataset_name,
                TRAINING_CONFIG.language_pair,
                cache_dir=TRAINING_CONFIG.cache_dir,
            )

            # 提取语言对
            src_lang, tgt_lang = TRAINING_CONFIG.language_pair.split("-")

            # 处理训练数据
            train_data = dataset["train"]
            if TRAINING_CONFIG.max_samples:
                train_data = train_data.select(range(min(TRAINING_CONFIG.max_samples, len(train_data))))

            train_pairs = []
            for item in train_data:
                src_text = item["translation"][src_lang].strip()
                tgt_text = item["translation"][tgt_lang].strip()

                # 过滤过长或过短的句子
                if (
                    DATA_CONFIG.min_length <= len(src_text.split()) <= DATA_CONFIG.max_length
                    and DATA_CONFIG.min_length <= len(tgt_text.split()) <= DATA_CONFIG.max_length
                ):
                    train_pairs.append((src_text, tgt_text))

            # 处理验证数据 (使用训练数据的一部分)
            val_size = min(len(train_pairs) // 10, 1000)  # 10%或最多1000个样本
            val_pairs = train_pairs[-val_size:]
            train_pairs = train_pairs[:-val_size]

            logger.info(f"数据加载完成:")
            logger.info(f"  - 训练对数: {len(train_pairs)}")
            logger.info(f"  - 验证对数: {len(val_pairs)}")

            return train_pairs, val_pairs

        except Exception as e:
            logger.error(f"数据集下载失败: {e}")
            raise

    def _build_tokenizer(self, train_pairs: List[Tuple[str, str]]) -> None:
        """
        构建分词器

        Args:
            train_pairs: 训练数据对
        """
        logger.info("正在构建分词器...")

        # 提取源语言和目标语言文本
        src_texts = [pair[0] for pair in train_pairs]
        tgt_texts = [pair[1] for pair in train_pairs]

        # 构建词汇表
        self.tokenizer.build_vocab(src_texts, tgt_texts)

        # 保存词汇表
        vocab_dir = TRAINING_CONFIG.vocab_save_dir
        self.tokenizer.save_vocab(vocab_dir)

        # 更新配置中的词汇表大小
        TRANSFORMER_CONFIG.vocab_size_src = len(self.tokenizer.vocab_src)
        TRANSFORMER_CONFIG.vocab_size_tgt = len(self.tokenizer.vocab_tgt)

        logger.info("分词器构建完成")

    def get_tokenizer(self) -> SimpleTokenizer:
        """获取分词器"""
        return self.tokenizer
