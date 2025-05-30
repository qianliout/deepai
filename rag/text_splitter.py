"""
文本分割器模块

该模块负责将长文档分割成适合向量化的文本块。
支持多种分割策略，确保语义完整性和检索效果。

数据流：
1. 长文档 -> 分割策略选择 -> 文本块生成 -> 重叠处理
2. 文本块大小: [chunk_size] 字符
3. 重叠大小: [chunk_overlap] 字符
"""

import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

from config import config
from logger import get_logger, log_execution_time


@dataclass
class SplitResult:
    """分割结果数据结构"""
    chunks: List[str]
    chunk_count: int
    total_chars: int
    avg_chunk_size: float
    overlap_ratio: float


class TextSplitterManager:
    """文本分割管理器

    提供多种文本分割策略，根据文档类型和内容特点选择最佳分割方法。

    支持的分割策略：
    - 递归字符分割: 按层次结构分割
    - 字符分割: 按固定字符数分割
    - Token分割: 按Token数量分割
    - 语义分割: 按语义边界分割
    """

    def __init__(self):
        """初始化文本分割管理器"""
        self.logger = get_logger("TextSplitterManager")

        # 中文分割符优先级
        self.chinese_separators = [
            "\n\n",      # 段落分隔
            "\n",        # 行分隔
            "。",        # 句号
            "！",        # 感叹号
            "？",        # 问号
            "；",        # 分号
            "，",        # 逗号
            " ",         # 空格
            ""           # 字符级分割
        ]

        # 英文分割符优先级
        self.english_separators = [
            "\n\n",      # 段落分隔
            "\n",        # 行分隔
            ". ",        # 句号+空格
            "! ",        # 感叹号+空格
            "? ",        # 问号+空格
            "; ",        # 分号+空格
            ", ",        # 逗号+空格
            " ",         # 空格
            ""           # 字符级分割
        ]

        # 代码分割符
        self.code_separators = [
            "\n\n",      # 空行
            "\nclass ",  # 类定义
            "\ndef ",    # 函数定义
            "\n\n",      # 段落
            "\n",        # 行
            " ",         # 空格
            ""           # 字符级
        ]

        self.logger.info("文本分割管理器初始化完成")

    @log_execution_time("split_documents")
    def split_documents(
        self,
        documents: List[Document],
        strategy: str = "recursive",
        **kwargs
    ) -> List[Document]:
        """分割文档列表

        Args:
            documents: 文档列表
            strategy: 分割策略 (recursive/character/token/semantic)
            **kwargs: 额外参数

        Returns:
            分割后的文档列表
        """
        if not documents:
            return []

        try:
            self.logger.info(f"开始分割 {len(documents)} 个文档，策略: {strategy}")

            # 选择分割器
            splitter = self._get_splitter(strategy, **kwargs)

            # 分割所有文档
            split_docs = []
            for doc in documents:
                chunks = self._split_single_document(doc, splitter)
                split_docs.extend(chunks)

            self.logger.info(
                f"文档分割完成 | 原始: {len(documents)} -> 分块: {len(split_docs)}"
            )

            return split_docs

        except Exception as e:
            self.logger.error(f"文档分割失败: {e}")
            raise

    def _get_splitter(self, strategy: str, **kwargs):
        """获取分割器实例

        Args:
            strategy: 分割策略
            **kwargs: 额外参数

        Returns:
            分割器实例
        """
        # 合并配置参数
        params = {
            "chunk_size": kwargs.get("chunk_size", config.text_splitter.chunk_size),
            "chunk_overlap": kwargs.get("chunk_overlap", config.text_splitter.chunk_overlap),
            "length_function": len,
            "keep_separator": config.text_splitter.keep_separator,
        }

        if strategy == "recursive":
            # 递归字符分割器（推荐）
            separators = kwargs.get("separators", self._get_separators_by_language(kwargs.get("language", "mixed")))
            return RecursiveCharacterTextSplitter(
                separators=separators,
                **params
            )

        elif strategy == "character":
            # 字符分割器
            separator = kwargs.get("separator", "\n\n")
            return CharacterTextSplitter(
                separator=separator,
                **params
            )

        elif strategy == "token":
            # Token分割器
            return TokenTextSplitter(
                **params
            )

        elif strategy == "semantic":
            # 语义分割器（自定义实现）
            return self._create_semantic_splitter(**params)

        else:
            raise ValueError(f"不支持的分割策略: {strategy}")

    def _get_separators_by_language(self, language: str) -> List[str]:
        """根据语言获取分割符

        Args:
            language: 语言类型 (chinese/english/code/mixed)

        Returns:
            分割符列表
        """
        if language == "chinese":
            return self.chinese_separators
        elif language == "english":
            return self.english_separators
        elif language == "code":
            return self.code_separators
        else:  # mixed
            # 混合语言，合并中英文分割符
            return self.chinese_separators + [sep for sep in self.english_separators if sep not in self.chinese_separators]

    def _split_single_document(self, document: Document, splitter) -> List[Document]:
        """分割单个文档

        Args:
            document: 原始文档
            splitter: 分割器实例

        Returns:
            分割后的文档列表
        """
        try:
            # 使用分割器分割文本
            chunks = splitter.split_text(document.page_content)

            # 创建新的文档对象
            split_docs = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # 跳过空块
                    # 复制原始元数据并添加分块信息
                    metadata = document.metadata.copy()
                    metadata.update({
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                        "chunk_size": len(chunk),
                        "original_doc_id": metadata.get("file_hash", "unknown")
                    })

                    split_docs.append(Document(
                        page_content=chunk,
                        metadata=metadata
                    ))

            return split_docs

        except Exception as e:
            self.logger.error(f"单文档分割失败: {e}")
            # 如果分割失败，返回原文档
            return [document]

    def _create_semantic_splitter(self, **params):
        """创建语义分割器

        基于语义边界进行分割，保持语义完整性
        """
        class SemanticSplitter:
            def __init__(self, chunk_size, chunk_overlap, **kwargs):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.logger = get_logger("SemanticSplitter")

            def split_text(self, text: str) -> List[str]:
                """语义分割实现"""
                # 首先按段落分割
                paragraphs = re.split(r'\n\s*\n', text)

                chunks = []
                current_chunk = ""

                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue

                    # 如果当前段落加入后超过chunk_size，先保存当前chunk
                    if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())

                        # 处理重叠
                        if self.chunk_overlap > 0:
                            overlap_text = current_chunk[-self.chunk_overlap:]
                            current_chunk = overlap_text + "\n" + paragraph
                        else:
                            current_chunk = paragraph
                    else:
                        # 添加到当前chunk
                        if current_chunk:
                            current_chunk += "\n\n" + paragraph
                        else:
                            current_chunk = paragraph

                # 添加最后一个chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                return chunks

        return SemanticSplitter(**params)

    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """分析文本结构

        Args:
            text: 输入文本

        Returns:
            文本结构分析结果
        """
        analysis = {
            "total_chars": len(text),
            "total_words": len(text.split()),
            "total_lines": len(text.splitlines()),
            "paragraphs": len(re.split(r'\n\s*\n', text)),
            "sentences": len(re.split(r'[。！？.!?]+', text)),
            "language": self._detect_language(text),
            "has_code": self._has_code_blocks(text),
            "structure_type": self._detect_structure_type(text)
        }

        return analysis

    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        # 简单的语言检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))

        total_chars = chinese_chars + english_chars
        if total_chars == 0:
            return "unknown"

        chinese_ratio = chinese_chars / total_chars

        if chinese_ratio > 0.7:
            return "chinese"
        elif chinese_ratio < 0.3:
            return "english"
        else:
            return "mixed"

    def _has_code_blocks(self, text: str) -> bool:
        """检测是否包含代码块"""
        code_patterns = [
            r'```[\s\S]*?```',  # Markdown代码块
            r'`[^`]+`',         # 行内代码
            r'def\s+\w+\s*\(',  # Python函数
            r'class\s+\w+\s*:', # Python类
            r'function\s+\w+\s*\(',  # JavaScript函数
            r'#include\s*<',    # C/C++头文件
        ]

        for pattern in code_patterns:
            if re.search(pattern, text):
                return True

        return False

    def _detect_structure_type(self, text: str) -> str:
        """检测文档结构类型"""
        # 检测标题结构
        if re.search(r'^#{1,6}\s+', text, re.MULTILINE):
            return "markdown"

        # 检测列表结构
        if re.search(r'^\s*[-*+]\s+', text, re.MULTILINE):
            return "list"

        # 检测表格结构
        if re.search(r'\|.*\|', text):
            return "table"

        # 检测代码结构
        if self._has_code_blocks(text):
            return "code"

        return "plain"

    def get_optimal_strategy(self, text: str) -> str:
        """获取最佳分割策略

        Args:
            text: 输入文本

        Returns:
            推荐的分割策略
        """
        analysis = self.analyze_text_structure(text)

        # 根据文本特征选择策略
        if analysis["has_code"]:
            return "recursive"  # 代码文档使用递归分割
        elif analysis["structure_type"] == "markdown":
            return "recursive"  # Markdown使用递归分割
        elif analysis["paragraphs"] > 10:
            return "semantic"   # 长文档使用语义分割
        else:
            return "recursive"  # 默认使用递归分割

    def split_with_auto_strategy(self, documents: List[Document]) -> List[Document]:
        """自动选择策略分割文档

        Args:
            documents: 文档列表

        Returns:
            分割后的文档列表
        """
        if not documents:
            return []

        # 分析第一个文档来确定策略
        sample_text = documents[0].page_content
        strategy = self.get_optimal_strategy(sample_text)

        self.logger.info(f"自动选择分割策略: {strategy}")

        return self.split_documents(documents, strategy=strategy)

    def get_split_stats(self, original_docs: List[Document], split_docs: List[Document]) -> Dict[str, Any]:
        """获取分割统计信息

        Args:
            original_docs: 原始文档列表
            split_docs: 分割后文档列表

        Returns:
            分割统计信息
        """
        if not split_docs:
            return {}

        chunk_sizes = [len(doc.page_content) for doc in split_docs]

        stats = {
            "original_count": len(original_docs),
            "split_count": len(split_docs),
            "split_ratio": len(split_docs) / len(original_docs) if original_docs else 0,
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_chars": sum(chunk_sizes),
            "chunks_per_doc": len(split_docs) / len(original_docs) if original_docs else 0
        }

        return stats
