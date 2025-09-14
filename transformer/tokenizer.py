"""
分词器 - 自实现分词功能，构建和保存词典
"""

import re
import json
import os
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from config import ModelConfig
from utils import save_vocab, load_vocab


class SimpleTokenizer:
    """
    简单的分词器实现
    支持基本的文本清理、分词和词典构建
    """

    def __init__(self, config: ModelConfig):
        """
        初始化分词器

        Args:
            config: 模型配置
        """
        self.config = config
        self.logger = logging.getLogger('transformer.tokenizer')

        # 特殊token
        self.pad_token = config.pad_token
        self.unk_token = config.unk_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token

        # 词典
        self.vocab_en: Dict[str, int] = {}
        self.vocab_it: Dict[str, int] = {}
        self.id2word_en: Dict[int, str] = {}
        self.id2word_it: Dict[int, str] = {}

        # 特殊token的ID
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def clean_text(self, text: str) -> str:
        """
        清理文本

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        # 转换为小写
        text = text.lower()

        # 移除多余的空白字符
        text = re.sub(r"\s+", " ", text)

        # 在标点符号前后添加空格
        text = re.sub(r"([.!?,:;])", r" \1 ", text)

        # 移除特殊字符，保留字母、数字、基本标点
        text = re.sub(r"[^a-zA-Z0-9\s.!?,:;àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]", "", text)

        # 再次清理多余空格
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        分词

        Args:
            text: 输入文本

        Returns:
            token列表
        """
        # 清理文本
        cleaned_text = self.clean_text(text)

        # 简单按空格分词
        tokens = cleaned_text.split()

        return tokens

    def build_vocab(
        self,
        texts: List[str],
        lang: str,
        min_freq: int = 2,
        max_vocab_size: int = 1000000,
    ) -> Dict[str, int]:
        """
        构建词典

        Args:
            texts: 文本列表
            lang: 语言标识 ('en' 或 'it')
            min_freq: 最小词频
            max_vocab_size: 最大词典大小

        Returns:
            词典字典
        """
        self.logger.info(f"正在构建{lang}词典...")

        # 统计词频
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)

        self.logger.info(f"总共发现 {len(word_counts)} 个不同的词")

        # 过滤低频词并按频率排序
        filtered_words = [
            word for word, count in word_counts.items() if count >= min_freq
        ]

        # 按频率降序排序
        filtered_words.sort(key=lambda x: word_counts[x], reverse=True)

        # 限制词典大小（保留空间给特殊token）
        max_words = max_vocab_size - 4  # 4个特殊token
        if len(filtered_words) > max_words:
            filtered_words = filtered_words[:max_words]

        self.logger.info(f"过滤后保留 {len(filtered_words)} 个词")

        # 构建词典
        vocab = {}

        # 添加特殊token
        vocab[self.pad_token] = self.pad_id
        vocab[self.unk_token] = self.unk_id
        vocab[self.bos_token] = self.bos_id
        vocab[self.eos_token] = self.eos_id

        # 添加普通词汇
        for i, word in enumerate(filtered_words):
            vocab[word] = i + 4  # 从4开始，前面是特殊token

        self.logger.info(f"{lang}词典构建完成，大小: {len(vocab)}")

        return vocab

    def save_vocabs(self, vocab_path: str):
        """
        保存词典到文件

        Args:
            vocab_path: 词典保存路径
        """
        save_vocab(self.vocab_en, vocab_path, "en")
        save_vocab(self.vocab_it, vocab_path, "it")

        self.logger.info("词典保存完成")

    def load_vocabs(self, vocab_path: str):
        """
        从文件加载词典

        Args:
            vocab_path: 词典路径
        """
        try:
            self.vocab_en = load_vocab(vocab_path, "en")
            self.vocab_it = load_vocab(vocab_path, "it")

            # 构建反向词典
            self.id2word_en = {v: k for k, v in self.vocab_en.items()}
            self.id2word_it = {v: k for k, v in self.vocab_it.items()}

            self.logger.info("词典加载完成")
            self.logger.info(f"英语词典大小: {len(self.vocab_en)}")
            self.logger.info(f"意大利语词典大小: {len(self.vocab_it)}")

        except FileNotFoundError as e:
            self.logger.error(f"词典文件不存在: {e}")
            raise

    def encode(
        self, text: str, lang: str, max_length: Optional[int] = None
    ) -> List[int]:
        """
        将文本编码为ID序列

        Args:
            text: 输入文本
            lang: 语言 ('en' 或 'it')
            max_length: 最大长度

        Returns:
            ID序列
        """
        vocab = self.vocab_en if lang == "en" else self.vocab_it

        # 分词
        tokens = self.tokenize(text)

        # 添加BOS和EOS token
        tokens = [self.bos_token] + tokens + [self.eos_token]

        # 转换为ID
        ids = []
        for token in tokens:
            if token in vocab:
                ids.append(vocab[token])
            else:
                ids.append(self.unk_id)

        # 截断或填充
        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
                ids[-1] = self.eos_id  # 确保最后是EOS
            else:
                # 填充
                ids.extend([self.pad_id] * (max_length - len(ids)))

        return ids

    def decode(self, ids: List[int], lang: str) -> str:
        """
        将ID序列解码为文本

        Args:
            ids: ID序列
            lang: 语言 ('en' 或 'it')

        Returns:
            解码后的文本
        """
        id2word = self.id2word_en if lang == "en" else self.id2word_it

        tokens = []
        for idx in ids:
            if idx in id2word:
                token = id2word[idx]
                # 跳过特殊token（除了标点）
                if token not in [self.pad_token, self.bos_token, self.eos_token]:
                    tokens.append(token)
            else:
                tokens.append(self.unk_token)

        # 连接token
        text = " ".join(tokens)

        # 简单的后处理：移除标点前的空格
        text = re.sub(r" ([.!?,:;])", r"\1", text)

        return text

    def get_vocab_size(self, lang: str) -> int:
        """
        获取词典大小

        Args:
            lang: 语言 ('en' 或 'it')

        Returns:
            词典大小
        """
        vocab = self.vocab_en if lang == "en" else self.vocab_it
        return len(vocab)
