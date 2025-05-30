"""
文档加载器模块 - 简化版

该模块负责加载和解析TXT文本文档。
专注于核心RAG功能，只支持纯文本格式。

数据流：
1. 文件路径 -> 编码检测 -> 内容提取 -> 文档对象创建
2. 支持批量加载和基本元数据提取
"""

from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass
import hashlib

from langchain_core.documents import Document

from config import config
from logger import get_logger, log_execution_time


@dataclass
class DocumentInfo:
    """文档信息数据结构"""
    filepath: str
    filename: str
    file_size: int
    file_type: str
    encoding: Optional[str] = None
    word_count: Optional[int] = None
    char_count: Optional[int] = None
    file_hash: Optional[str] = None


class DocumentLoader:
    """文档加载器 - 简化版

    只支持TXT文本文件的加载和解析，专注于核心RAG功能。

    支持的格式：
    - 文本文件 (.txt)
    """

    def __init__(self):
        """初始化文档加载器"""
        self.logger = get_logger("DocumentLoader")

        # 支持的文件类型映射
        self.supported_extensions = {
            '.txt': self._load_text,
        }

        self.logger.info(f"文档加载器初始化完成，支持 {len(self.supported_extensions)} 种文件格式")

    @log_execution_time("load_document")
    def load_document(self, filepath: Union[str, Path]) -> List[Document]:
        """加载单个文档

        Args:
            filepath: 文档文件路径

        Returns:
            文档对象列表
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        if not filepath.is_file():
            raise ValueError(f"路径不是文件: {filepath}")

        # 获取文件扩展名
        extension = filepath.suffix.lower()

        if extension not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {extension}，仅支持 .txt 文件")

        try:
            self.logger.info(f"开始加载文档: {filepath}")

            # 获取文档信息
            doc_info = self._get_document_info(filepath)

            # 调用相应的加载函数
            loader_func = self.supported_extensions[extension]
            documents = loader_func(filepath, doc_info)

            self.logger.info(f"文档加载完成: {filepath}，生成 {len(documents)} 个文档对象")

            return documents

        except Exception as e:
            self.logger.error(f"文档加载失败: {filepath}, 错误: {e}")
            raise

    def load_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        file_pattern: Optional[str] = None
    ) -> List[Document]:
        """加载目录中的所有TXT文档

        Args:
            directory: 目录路径
            recursive: 是否递归加载子目录
            file_pattern: 文件名模式过滤

        Returns:
            所有文档对象列表
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")

        if not directory.is_dir():
            raise ValueError(f"路径不是目录: {directory}")

        try:
            self.logger.info(f"开始加载目录: {directory}")

            all_documents = []
            file_count = 0

            # 获取文件列表
            if recursive:
                pattern = "**/*.txt" if not file_pattern else f"**/{file_pattern}"
                files = directory.glob(pattern)
            else:
                pattern = "*.txt" if not file_pattern else file_pattern
                files = directory.glob(pattern)

            # 过滤支持的文件类型
            supported_files = [
                f for f in files
                if f.is_file() and f.suffix.lower() in self.supported_extensions
            ]

            # 加载每个文件
            for filepath in supported_files:
                try:
                    documents = self.load_document(filepath)
                    all_documents.extend(documents)
                    file_count += 1
                except Exception as e:
                    self.logger.warning(f"跳过文件 {filepath}: {e}")
                    continue

            self.logger.info(
                f"目录加载完成: {directory}，"
                f"处理 {file_count} 个文件，生成 {len(all_documents)} 个文档对象"
            )

            return all_documents

        except Exception as e:
            self.logger.error(f"目录加载失败: {directory}, 错误: {e}")
            raise

    def _get_document_info(self, filepath: Path) -> DocumentInfo:
        """获取文档基本信息

        Args:
            filepath: 文件路径

        Returns:
            文档信息对象
        """
        stat = filepath.stat()

        # 计算文件哈希
        file_hash = self._calculate_file_hash(filepath)

        return DocumentInfo(
            filepath=str(filepath),
            filename=filepath.name,
            file_size=stat.st_size,
            file_type=filepath.suffix.lower(),
            file_hash=file_hash
        )

    def _calculate_file_hash(self, filepath: Path) -> str:
        """计算文件MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _load_text(self, filepath: Path, doc_info: DocumentInfo) -> List[Document]:
        """加载文本文档

        Args:
            filepath: 文件路径
            doc_info: 文档信息

        Returns:
            文档对象列表
        """
        # 尝试不同的编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']

        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as file:
                    content = file.read()
                    doc_info.encoding = encoding
                    break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"无法解码文件: {filepath}")

        # 更新文档信息
        doc_info.char_count = len(content)
        doc_info.word_count = len(content.split())

        # 创建元数据
        metadata = {
            **doc_info.__dict__,
            'source_type': 'text'
        }

        return [Document(page_content=content, metadata=metadata)]

    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式列表"""
        return list(self.supported_extensions.keys())

    def is_supported(self, filepath: Union[str, Path]) -> bool:
        """检查文件格式是否支持

        Args:
            filepath: 文件路径

        Returns:
            是否支持该格式
        """
        extension = Path(filepath).suffix.lower()
        return extension in self.supported_extensions