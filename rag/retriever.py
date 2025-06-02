"""
增强检索器模块

该模块负责从多个数据源检索相关文档。
支持ES粗排+向量精排的混合检索策略。

数据流：
1. 查询文本 -> 查询扩展 -> ES关键词检索 -> 粗排结果
2. 粗排结果 -> 向量化 -> 向量相似度计算 -> 精排结果
3. 精排结果 -> 重排序 -> 过滤 -> 返回最终结果

学习要点：
1. 混合检索策略的设计
2. ES关键词检索和向量检索的结合
3. 多阶段排序和融合算法
4. 检索性能优化策略
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document

from logger import get_logger, log_execution_time
from config import defaultConfig
from query_expander import SimpleQueryExpander

# 尝试导入ES管理器，如果失败则使用向量检索
try:
    from elasticsearch_manager import ElasticsearchManager, SearchResult
    ES_AVAILABLE = True
except ImportError:
    print("⚠️  Elasticsearch管理器不可用，使用纯向量检索")
    ES_AVAILABLE = False


@dataclass
class RetrievalResult:
    """检索结果数据结构"""

    document: Document
    score: float
    retrieval_method: str
    rank: int
    es_score: Optional[float] = None
    vector_score: Optional[float] = None
    combined_score: Optional[float] = None
    highlights: Optional[List[str]] = None


class HybridRetrieverManager:
    """混合检索器管理器

    支持ES粗排+向量精排的混合检索策略。
    提供多阶段检索和结果融合功能。

    Attributes:
        vector_store: 向量存储管理器
        embedding_manager: 嵌入管理器
        es_manager: Elasticsearch管理器
        query_expander: 查询扩展器
    """

    def __init__(self, vector_store, embedding_manager=None):
        """初始化混合检索器管理器

        Args:
            vector_store: 向量存储管理器
            embedding_manager: 嵌入管理器
        """
        self.logger = get_logger("HybridRetrieverManager")
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager or vector_store.embedding_manager

        # 初始化查询扩展器
        self.query_expander = SimpleQueryExpander(enable_expansion=True)

        # 初始化ES管理器（如果可用）
        if ES_AVAILABLE:
            try:
                self.es_manager = ElasticsearchManager()
                self.hybrid_mode = True
                self.logger.info("混合检索模式已启用 (ES + Vector)")
            except Exception as e:
                self.logger.warning(f"ES管理器初始化失败，使用纯向量检索: {e}")
                self.es_manager = None
                self.hybrid_mode = False
        else:
            self.es_manager = None
            self.hybrid_mode = False
            self.logger.info("纯向量检索模式")

        self.logger.info("混合检索器管理器初始化完成")

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

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        es_candidates: int = 50,
        use_query_expansion: bool = True,
        **kwargs
    ) -> List[RetrievalResult]:
        """执行混合检索

        Args:
            query: 查询字符串
            top_k: 最终返回结果数量
            es_candidates: ES粗排候选数量
            use_query_expansion: 是否使用查询扩展
            **kwargs: 额外参数

        Returns:
            检索结果列表
        """
        top_k = top_k or defaultConfig.vector_store.top_k

        try:
            self.logger.info(f"开始混合检索: {query[:50]}... | top_k: {top_k}")

            if self.hybrid_mode and self.es_manager:
                # 混合检索模式：ES粗排 + 向量精排
                return self._hybrid_retrieve(query, top_k, es_candidates, use_query_expansion, **kwargs)
            else:
                # 纯向量检索模式
                return self._vector_only_retrieve(query, top_k, use_query_expansion, **kwargs)

        except Exception as e:
            self.logger.error(f"检索失败: {e}")
            return []

    def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        es_candidates: int,
        use_query_expansion: bool,
        **kwargs
    ) -> List[RetrievalResult]:
        """混合检索：ES粗排 + 向量精排

        Args:
            query: 查询字符串
            top_k: 最终返回结果数量
            es_candidates: ES粗排候选数量
            use_query_expansion: 是否使用查询扩展
            **kwargs: 额外参数

        Returns:
            检索结果列表
        """
        start_time = time.time()

        # 1. 查询扩展
        expanded_query = query
        if use_query_expansion:
            expansion_result = self.query_expander.expand_query(query)
            expanded_query = expansion_result.expanded_query
            self.logger.debug(f"查询扩展: '{query}' -> '{expanded_query}'")

        # 2. ES粗排：关键词检索
        es_start = time.time()
        es_results = self.es_manager.search_documents(
            expanded_query,
            size=es_candidates
        )
        es_time = time.time() - es_start

        if not es_results:
            self.logger.warning("ES检索无结果，回退到纯向量检索")
            return self._vector_only_retrieve(query, top_k, use_query_expansion, **kwargs)

        self.logger.debug(f"ES粗排完成: {len(es_results)} 个候选 | 耗时: {es_time:.3f}s")

        # 3. 向量精排：对ES结果进行向量相似度计算
        vector_start = time.time()
        reranked_results = self._vector_rerank(query, es_results, top_k)
        vector_time = time.time() - vector_start

        total_time = time.time() - start_time

        self.logger.info(
            f"混合检索完成: ES({es_time:.3f}s) + Vector({vector_time:.3f}s) = {total_time:.3f}s | "
            f"候选: {len(es_results)} -> 精排: {len(reranked_results)}"
        )

        return reranked_results

    def _vector_only_retrieve(
        self,
        query: str,
        top_k: int,
        use_query_expansion: bool,
        **kwargs
    ) -> List[RetrievalResult]:
        """纯向量检索

        Args:
            query: 查询字符串
            top_k: 返回结果数量
            use_query_expansion: 是否使用查询扩展
            **kwargs: 额外参数

        Returns:
            检索结果列表
        """
        # 查询扩展
        search_query = query
        if use_query_expansion:
            expansion_result = self.query_expander.expand_query(query)
            search_query = expansion_result.expanded_query
            self.logger.debug(f"查询扩展: '{query}' -> '{search_query}'")

        # 向量相似度搜索
        results = self.vector_store.similarity_search(
            search_query,
            k=top_k,
            score_threshold=kwargs.get("score_threshold", defaultConfig.vector_store.score_threshold)
        )

        # 转换为RetrievalResult格式
        retrieval_results = []
        for i, (doc, score) in enumerate(results):
            retrieval_results.append(
                RetrievalResult(
                    document=doc,
                    score=score,
                    retrieval_method="vector_only",
                    rank=i + 1,
                    vector_score=score
                )
            )

        self.logger.debug(f"纯向量检索完成，返回 {len(retrieval_results)} 个结果")
        return retrieval_results

    def _vector_rerank(
        self,
        query: str,
        es_results: List[SearchResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """向量重排序ES结果

        Args:
            query: 查询字符串
            es_results: ES搜索结果
            top_k: 返回结果数量

        Returns:
            重排序后的检索结果
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_manager.embed_query(query)

            # 为每个ES结果计算向量相似度
            rerank_candidates = []
            for es_result in es_results:
                try:
                    # 生成文档向量
                    doc_embedding = self.embedding_manager.embed_documents([es_result.content])[0]

                    # 计算余弦相似度
                    vector_score = self._cosine_similarity(query_embedding, doc_embedding)

                    # 创建Document对象
                    document = Document(
                        page_content=es_result.content,
                        metadata={
                            "title": es_result.title,
                            "doc_id": es_result.doc_id,
                            **es_result.metadata
                        }
                    )

                    # 计算组合分数 (ES分数 + 向量分数的加权平均)
                    es_weight = 0.3
                    vector_weight = 0.7
                    combined_score = es_weight * es_result.score + vector_weight * vector_score

                    rerank_candidates.append(RetrievalResult(
                        document=document,
                        score=combined_score,
                        retrieval_method="hybrid",
                        rank=0,  # 将在排序后设置
                        es_score=es_result.score,
                        vector_score=vector_score,
                        combined_score=combined_score,
                        highlights=es_result.highlights
                    ))

                except Exception as e:
                    self.logger.warning(f"向量重排序单个文档失败: {e}")
                    continue

            # 按组合分数排序
            rerank_candidates.sort(key=lambda x: x.combined_score, reverse=True)

            # 设置排名并返回top_k结果
            final_results = []
            for i, result in enumerate(rerank_candidates[:top_k]):
                result.rank = i + 1
                final_results.append(result)

            return final_results

        except Exception as e:
            self.logger.error(f"向量重排序失败: {e}")
            # 回退：直接使用ES结果
            return self._fallback_es_results(es_results, top_k)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            余弦相似度值
        """
        try:
            import numpy as np

            vec1 = np.array(vec1)
            vec2 = np.array(vec2)

            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception as e:
            self.logger.error(f"余弦相似度计算失败: {e}")
            return 0.0

    def _fallback_es_results(
        self,
        es_results: List[SearchResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """回退方案：直接使用ES结果

        Args:
            es_results: ES搜索结果
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        results = []
        for i, es_result in enumerate(es_results[:top_k]):
            document = Document(
                page_content=es_result.content,
                metadata={
                    "title": es_result.title,
                    "doc_id": es_result.doc_id,
                    **es_result.metadata
                }
            )

            results.append(RetrievalResult(
                document=document,
                score=es_result.score,
                retrieval_method="es_only",
                rank=i + 1,
                es_score=es_result.score,
                highlights=es_result.highlights
            ))

        return results

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        vector_stats = self.vector_store.get_stats()

        stats = {
            "vector_store_stats": vector_stats,
            "retrieval_method": "hybrid" if self.hybrid_mode else "vector_only",
            "top_k": defaultConfig.vector_store.top_k,
            "score_threshold": defaultConfig.vector_store.score_threshold,
            "hybrid_mode": self.hybrid_mode,
            "query_expansion_enabled": True
        }

        if self.es_manager:
            es_info = self.es_manager.get_connection_info()
            stats["elasticsearch_info"] = es_info

        return stats


# 保持向后兼容性的别名
RetrieverManager = HybridRetrieverManager