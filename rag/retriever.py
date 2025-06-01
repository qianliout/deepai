"""
检索器模块
简化版检索器，专注于基本的向量检索功能，整合了simple_retriever.py的功能
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document

from logger import get_logger, log_execution_time
from config import VECTORSTORE_CONFIG


@dataclass
class RetrievalResult:
    """检索结果数据结构"""

    document: Document
    score: float
    retrieval_method: str
    rank: int


class RetrieverManager:
    """简化检索器管理器

    提供基本的向量检索功能，整合了simple_retriever.py的功能
    """

    def __init__(self, vector_store, embedding_manager=None):
        """初始化检索器管理器

        Args:
            vector_store: 向量存储管理器
            embedding_manager: 嵌入管理器
        """
        self.logger = get_logger("RetrieverManager")
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager or vector_store.embedding_manager

        self.logger.info("检索器管理器初始化完成")

    @log_execution_time("retrieve_documents")
    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """获取相关文档（BaseRetriever接口实现）

        Args:
            query: 查询字符串
            **kwargs: 额外参数

        Returns:
            相关文档列表
        """
        results = self.retrieve(query, **kwargs)
        return [result.document for result in results]

    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[RetrievalResult]:
        """执行检索

        Args:
            query: 查询字符串
            top_k: 返回结果数量
            **kwargs: 额外参数

        Returns:
            检索结果列表
        """
        top_k = top_k or VECTORSTORE_CONFIG.top_k

        try:
            self.logger.debug(f"开始检索: {query[:50]}... | top_k: {top_k}")

            # 使用向量存储进行相似度搜索
            results = self.vector_store.similarity_search(
                query,
                k=top_k,
                score_threshold=kwargs.get("score_threshold", VECTORSTORE_CONFIG.score_threshold)
            )

            # 转换为RetrievalResult格式
            retrieval_results = []
            for i, (doc, score) in enumerate(results):
                retrieval_results.append(
                    RetrievalResult(
                        document=doc,
                        score=score,
                        retrieval_method="vector",
                        rank=i + 1
                    )
                )

            self.logger.debug(f"检索完成，返回 {len(retrieval_results)} 个结果")
            return retrieval_results

        except Exception as e:
            self.logger.error(f"检索失败: {e}")
            return []

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        vector_stats = self.vector_store.get_stats()

        return {
            "vector_store_stats": vector_stats,
            "retrieval_method": "simple_vector",
            "top_k": VECTORSTORE_CONFIG.top_k,
            "score_threshold": VECTORSTORE_CONFIG.score_threshold,
        }


# 为了兼容性，保留原来的类名
SimpleRetrieverManager = RetrieverManager
