"""
检索器模块

该模块实现多种检索策略，包括向量检索、关键词检索和混合检索。
支持重排序和查询扩展等高级功能。

数据流：
1. 查询输入 -> 查询处理 -> 多路检索 -> 结果融合 -> 重排序 -> 最终结果
2. 检索结果格式: [(Document, score), ...]
"""

import time
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from logger import get_logger, log_execution_time, LogExecutionTime
from query_expander import QueryExpander


@dataclass
class RetrievalResult:
    """检索结果数据结构"""

    document: Document
    score: float
    retrieval_method: str
    rank: int


class RetrieverManager(BaseRetriever):
    """检索器管理器

    整合多种检索策略，提供统一的检索接口。
    支持向量检索、关键词检索、混合检索等多种方式。

    Attributes:
        vector_store: 向量存储管理器
        embedding_manager: 嵌入管理器
        retrieval_config: 检索配置
    """

    def __init__(self, vector_store, embedding_manager=None):
        """初始化检索器管理器

        Args:
            vector_store: 向量存储管理器
            embedding_manager: 嵌入管理器
        """
        super().__init__()
        self.logger = get_logger("RetrieverManager")
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager or vector_store.embedding_manager
        self.retrieval_config = config.retriever

        # 初始化查询扩展器
        self.query_expander = SimpleQueryExpander(
            enable_synonyms=config.query_expansion.enable_synonyms
        ) if config.retriever.enable_query_expansion else None

        # 初始化检索策略
        self.retrievers = {
            "vector": VectorRetriever(vector_store, embedding_manager),
            "keyword": KeywordRetriever(vector_store),
            "hybrid": HybridRetriever(vector_store, embedding_manager),
        }

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

    def retrieve(self, query: str, top_k: Optional[int] = None, strategy: Optional[str] = None, **kwargs) -> List[RetrievalResult]:
        """执行检索

        Args:
            query: 查询字符串
            top_k: 返回结果数量
            strategy: 检索策略 (vector/keyword/hybrid)
            **kwargs: 额外参数

        Returns:
            检索结果列表
        """
        top_k = top_k or config.chromadb.top_k
        strategy = strategy or "hybrid" if self.retrieval_config.enable_hybrid_search else "vector"

        try:
            self.logger.debug(f"开始检索: {query[:50]}... | 策略: {strategy} | top_k: {top_k}")

            with LogExecutionTime("retrieval_process", strategy=strategy, top_k=top_k):
                # 查询预处理
                processed_query = self._preprocess_query(query)

                # 执行检索
                if strategy in self.retrievers:
                    retriever = self.retrievers[strategy]
                    results = retriever.retrieve(processed_query, top_k, **kwargs)
                else:
                    raise ValueError(f"不支持的检索策略: {strategy}")

                # 后处理
                results = self._postprocess_results(results, query)

                # 重排序
                if self.retrieval_config.enable_reranking and len(results) > 1:
                    results = self._rerank_results(results, query)

                # 截取top_k结果
                results = results[:top_k]

                self.logger.debug(f"检索完成，返回 {len(results)} 个结果")

                return results

        except Exception as e:
            self.logger.error(f"检索失败: {e}")
            raise

    def _preprocess_query(self, query: str) -> str:
        """查询预处理

        Args:
            query: 原始查询

        Returns:
            处理后的查询
        """
        # 基础清理
        processed = query.strip()

        # 查询扩展
        if self.retrieval_config.enable_query_expansion:
            processed = self._expand_query(processed)

        return processed

    def _expand_query(self, query: str) -> str:
        """查询扩展

        Args:
            query: 原始查询

        Returns:
            扩展后的查询
        """
        if not self.query_expander:
            return query

        try:
            # 使用查询扩展器进行同义词扩展
            expansion_result = self.query_expander.expand_query(
                query,
                max_synonyms_per_word=config.query_expansion.max_synonyms_per_word,
                similarity_threshold=config.query_expansion.similarity_threshold,
                max_expansion_ratio=config.query_expansion.max_expansion_ratio
            )

            self.logger.debug(
                f"查询扩展: '{query}' -> '{expansion_result.expanded_query}' | "
                f"扩展词数: {len(expansion_result.expansion_terms)}"
            )

            return expansion_result.expanded_query

        except Exception as e:
            self.logger.warning(f"查询扩展失败，使用原始查询: {e}")
            return query

    def _postprocess_results(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """结果后处理

        Args:
            results: 原始检索结果
            query: 查询字符串

        Returns:
            处理后的结果
        """
        # 去重
        seen_content = set()
        unique_results = []

        for result in results:
            content_hash = hash(result.document.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)

        # 按分数排序
        unique_results.sort(key=lambda x: x.score, reverse=True)

        # 更新排名
        for i, result in enumerate(unique_results):
            result.rank = i + 1

        return unique_results

    def _rerank_results(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """重排序结果

        Args:
            results: 原始结果
            query: 查询字符串

        Returns:
            重排序后的结果
        """
        try:
            # 获取更多候选结果用于重排序
            candidates = results[: self.retrieval_config.rerank_top_k]

            # 计算重排序分数
            reranked_results = []
            for result in candidates:
                # 结合多个因素计算新分数
                semantic_score = result.score

                # 文档长度因子
                length_factor = min(len(result.document.page_content) / 500, 1.0)

                # 查询匹配度
                query_terms = set(query.lower().split())
                doc_terms = set(result.document.page_content.lower().split())
                term_overlap = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0

                # 综合分数
                new_score = 0.6 * semantic_score + 0.2 * length_factor + 0.2 * term_overlap

                result.score = new_score
                reranked_results.append(result)

            # 重新排序
            reranked_results.sort(key=lambda x: x.score, reverse=True)

            # 更新排名
            for i, result in enumerate(reranked_results):
                result.rank = i + 1

            # 添加未参与重排序的结果
            remaining_results = results[self.retrieval_config.rerank_top_k :]
            for i, result in enumerate(remaining_results):
                result.rank = len(reranked_results) + i + 1

            return reranked_results + remaining_results

        except Exception as e:
            self.logger.warning(f"重排序失败，使用原始结果: {e}")
            return results

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        vector_stats = self.vector_store.get_stats()

        return {
            "vector_store_stats": vector_stats,
            "retrieval_config": {
                "enable_hybrid_search": self.retrieval_config.enable_hybrid_search,
                "enable_reranking": self.retrieval_config.enable_reranking,
                "enable_query_expansion": self.retrieval_config.enable_query_expansion,
                "keyword_weight": self.retrieval_config.keyword_weight,
                "semantic_weight": self.retrieval_config.semantic_weight,
            },
        }


class BaseRetrieverStrategy(ABC):
    """检索策略基类"""

    def __init__(self, vector_store, embedding_manager=None):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def retrieve(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """执行检索"""
        pass


class VectorRetriever(BaseRetrieverStrategy):
    """向量检索器"""

    def retrieve(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """向量检索实现"""
        try:
            # 使用向量存储进行相似度搜索
            results = self.vector_store.similarity_search(
                query, k=top_k, score_threshold=kwargs.get("score_threshold", config.chromadb.score_threshold)
            )

            # 转换为RetrievalResult格式
            retrieval_results = []
            for i, (doc, score) in enumerate(results):
                retrieval_results.append(RetrievalResult(document=doc, score=score, retrieval_method="vector", rank=i + 1))

            return retrieval_results

        except Exception as e:
            self.logger.error(f"向量检索失败: {e}")
            return []


class KeywordRetriever(BaseRetrieverStrategy):
    """关键词检索器"""

    def retrieve(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """关键词检索实现（简化版本）"""
        try:
            # 获取所有文档进行关键词匹配
            # 注意：这是一个简化实现，实际应用中应该使用专门的全文搜索引擎

            # 先通过向量检索获取候选文档
            vector_results = self.vector_store.similarity_search(query, k=top_k * 3)

            if not vector_results:
                return []

            # 计算关键词匹配分数
            query_terms = set(query.lower().split())
            keyword_results = []

            for doc, _ in vector_results:
                doc_terms = set(doc.page_content.lower().split())

                # 计算TF-IDF风格的分数
                term_matches = query_terms & doc_terms
                if term_matches:
                    # 简化的TF-IDF计算
                    tf_score = len(term_matches) / len(query_terms)
                    idf_score = 1.0  # 简化，实际应该计算IDF
                    keyword_score = tf_score * idf_score

                    keyword_results.append(
                        RetrievalResult(document=doc, score=keyword_score, retrieval_method="keyword", rank=0)  # 稍后更新
                    )

            # 按分数排序
            keyword_results.sort(key=lambda x: x.score, reverse=True)

            # 更新排名并截取top_k
            for i, result in enumerate(keyword_results[:top_k]):
                result.rank = i + 1

            return keyword_results[:top_k]

        except Exception as e:
            self.logger.error(f"关键词检索失败: {e}")
            return []


class HybridRetriever(BaseRetrieverStrategy):
    """混合检索器"""

    def __init__(self, vector_store, embedding_manager):
        super().__init__(vector_store, embedding_manager)
        self.vector_retriever = VectorRetriever(vector_store, embedding_manager)
        self.keyword_retriever = KeywordRetriever(vector_store, embedding_manager)

    def retrieve(self, query: str, top_k: int, **kwargs) -> List[RetrievalResult]:
        """混合检索实现"""
        try:
            # 分别执行向量检索和关键词检索
            vector_results = self.vector_retriever.retrieve(query, top_k * 2, **kwargs)
            keyword_results = self.keyword_retriever.retrieve(query, top_k * 2, **kwargs)

            # 融合结果
            fused_results = self._fuse_results(vector_results, keyword_results)

            # 按分数排序并截取top_k
            fused_results.sort(key=lambda x: x.score, reverse=True)

            # 更新排名
            for i, result in enumerate(fused_results[:top_k]):
                result.rank = i + 1
                result.retrieval_method = "hybrid"

            return fused_results[:top_k]

        except Exception as e:
            self.logger.error(f"混合检索失败: {e}")
            return []

    def _fuse_results(self, vector_results: List[RetrievalResult], keyword_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """融合检索结果"""
        # 创建文档到结果的映射
        doc_to_vector = {hash(r.document.page_content): r for r in vector_results}
        doc_to_keyword = {hash(r.document.page_content): r for r in keyword_results}

        # 获取所有唯一文档
        all_doc_hashes = set(doc_to_vector.keys()) | set(doc_to_keyword.keys())

        fused_results = []
        for doc_hash in all_doc_hashes:
            vector_result = doc_to_vector.get(doc_hash)
            keyword_result = doc_to_keyword.get(doc_hash)

            # 计算融合分数
            vector_score = vector_result.score if vector_result else 0.0
            keyword_score = keyword_result.score if keyword_result else 0.0

            # 加权融合
            fused_score = config.retriever.semantic_weight * vector_score + config.retriever.keyword_weight * keyword_score

            # 选择文档（优先选择向量检索结果）
            document = vector_result.document if vector_result else keyword_result.document

            fused_results.append(RetrievalResult(document=document, score=fused_score, retrieval_method="hybrid", rank=0))  # 稍后更新

        return fused_results
