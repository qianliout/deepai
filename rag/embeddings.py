"""
文本嵌入模块

该模块负责将文本转换为向量表示，支持多种嵌入模型。
使用sentence-transformers库实现高质量的文本嵌入。

数据流：
1. 文本输入 -> 预处理 -> 模型编码 -> 向量输出
2. 支持批量处理和GPU加速
3. 向量维度: [batch_size, embedding_dim]
"""

import torch
import numpy as np
from typing import List, Union, Optional, Tuple
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from pathlib import Path
import time

from config import config
from logger import get_logger, log_execution_time, LogExecutionTime


class EmbeddingManager(Embeddings):
    """文本嵌入管理器

    基于sentence-transformers实现的文本嵌入服务，
    支持中英文文本的高质量向量化。

    Attributes:
        model: SentenceTransformer模型实例
        device: 计算设备 (cpu/mps/cuda)
        model_name: 模型名称
        embedding_dim: 嵌入向量维度
    """

    def __init__(self, model_name: Optional[str] = None):
        """初始化嵌入管理器

        Args:
            model_name: 嵌入模型名称，默认使用配置中的模型
        """
        self.logger = get_logger("EmbeddingManager")
        self.model_name = model_name or config.embedding.model_name
        self.device = config.get_device()
        self.model = None
        self.embedding_dim = None

        self._load_model()

    def _load_model(self) -> None:
        """加载嵌入模型

        数据流：模型下载/加载 -> 设备配置 -> 维度获取
        """
        with LogExecutionTime("load_embedding_model", model_name=self.model_name):
            try:
                self.logger.info(f"正在加载嵌入模型: {self.model_name}")

                # 设置模型缓存目录
                cache_dir = Path(config.embedding.cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)

                # 加载模型
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(cache_dir),
                    device=self.device
                )

                # 获取嵌入维度
                self.embedding_dim = self.model.get_sentence_embedding_dimension()

                self.logger.info(
                    f"嵌入模型加载成功 | 模型: {self.model_name} | "
                    f"设备: {self.device} | 维度: {self.embedding_dim}"
                )

            except Exception as e:
                self.logger.error(f"嵌入模型加载失败: {e}")
                raise

    @log_execution_time("embed_documents")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表

        Args:
            texts: 文本列表，shape: [num_texts]

        Returns:
            嵌入向量列表，shape: [num_texts, embedding_dim]
        """
        if not texts:
            return []

        try:
            self.logger.debug(f"开始嵌入 {len(texts)} 个文档")

            # 批量编码
            embeddings = self.model.encode(
                texts,
                batch_size=config.embedding.batch_size,
                show_progress_bar=len(texts) > 10,
                normalize_embeddings=config.embedding.normalize_embeddings,
                convert_to_numpy=True
            )

            # 转换为列表格式
            # embeddings shape: [num_texts, embedding_dim]
            result = embeddings.tolist()

            self.logger.debug(
                f"文档嵌入完成 | 数量: {len(texts)} | "
                f"向量维度: {embeddings.shape}"
            )

            return result

        except Exception as e:
            self.logger.error(f"文档嵌入失败: {e}")
            raise

    @log_execution_time("embed_query")
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本

        Args:
            text: 查询文本

        Returns:
            嵌入向量，shape: [embedding_dim]
        """
        try:
            self.logger.debug(f"开始嵌入查询: {text[:50]}...")

            # 单个文本编码
            embedding = self.model.encode(
                [text],
                batch_size=1,
                normalize_embeddings=config.embedding.normalize_embeddings,
                convert_to_numpy=True
            )

            # embedding shape: [1, embedding_dim] -> [embedding_dim]
            result = embedding[0].tolist()

            self.logger.debug(f"查询嵌入完成 | 向量维度: {len(result)}")

            return result

        except Exception as e:
            self.logger.error(f"查询嵌入失败: {e}")
            raise

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """批量嵌入文本（返回numpy数组）

        Args:
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            嵌入向量数组，shape: [num_texts, embedding_dim]
        """
        batch_size = batch_size or config.embedding.batch_size

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 10,
                normalize_embeddings=config.embedding.normalize_embeddings,
                convert_to_numpy=True
            )

            return embeddings

        except Exception as e:
            self.logger.error(f"批量嵌入失败: {e}")
            raise

    def similarity(
        self,
        text1: Union[str, List[float]],
        text2: Union[str, List[float]]
    ) -> float:
        """计算两个文本或向量的相似度

        Args:
            text1: 文本或嵌入向量
            text2: 文本或嵌入向量

        Returns:
            余弦相似度 [-1, 1]
        """
        try:
            # 获取嵌入向量
            if isinstance(text1, str):
                emb1 = np.array(self.embed_query(text1))
            else:
                emb1 = np.array(text1)

            if isinstance(text2, str):
                emb2 = np.array(self.embed_query(text2))
            else:
                emb2 = np.array(text2)

            # 计算余弦相似度
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            return float(similarity)

        except Exception as e:
            self.logger.error(f"相似度计算失败: {e}")
            raise

    def get_model_info(self) -> dict:
        """获取模型信息

        Returns:
            模型信息字典
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "max_length": config.embedding.max_length,
            "normalize_embeddings": config.embedding.normalize_embeddings
        }

    def __repr__(self) -> str:
        return (
            f"EmbeddingManager("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"dim={self.embedding_dim})"
        )
