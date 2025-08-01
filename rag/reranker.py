"""
重排序器模块

该模块实现了双编码器和交叉编码器的重排序功能，用于提升检索结果的相关性。
支持多种融合策略和可配置的重排序流水线。

数据流：
1. 初始检索结果 -> 双编码器重排序 -> 候选结果筛选
2. 候选结果 -> 交叉编码器精排 -> 最终排序结果
3. 分数融合 -> 结果过滤 -> 返回最终结果

学习要点：
1. 双编码器vs交叉编码器的区别和应用场景
2. 多阶段重排序策略
3. 分数融合算法
4. 性能优化技巧
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.documents import Document

from config import defaultConfig, get_device
from logger import get_logger, log_execution_time, LogExecutionTime


@dataclass
class RerankResult:
    """重排序结果数据结构"""
    
    document: Document
    original_score: float
    bi_encoder_score: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    final_score: float = 0.0
    rank: int = 0
    rerank_method: str = "none"


class BiEncoderReranker:
    """双编码器重排序器
    
    使用双编码器模型对查询和文档分别编码，
    然后计算向量相似度进行重排序。
    
    特点：
    - 查询和文档独立编码
    - 计算效率高，适合大规模候选集
    - 可以预计算文档向量
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """初始化双编码器重排序器
        
        Args:
            model_name: 模型名称，默认使用配置中的模型
        """
        self.logger = get_logger("BiEncoderReranker")
        self.model_name = model_name or defaultConfig.reranker.bi_encoder_model
        self.device = get_device() if defaultConfig.reranker.device == "auto" else defaultConfig.reranker.device
        self.model: SentenceTransformer
        
        self._load_model()
    
    def _load_model(self) -> None:
        """加载双编码器模型"""
        with LogExecutionTime("load_bi_encoder_model"):
            try:
                self.logger.info(f"正在加载双编码器模型: {self.model_name}")
                
                cache_dir = Path(defaultConfig.reranker.cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(cache_dir),
                    device=self.device,
                    local_files_only=True
                )
                
                self.logger.info(f"双编码器模型加载成功 | 模型: {self.model_name} | 设备: {self.device}")
                
            except Exception as e:
                self.logger.error(f"双编码器模型加载失败: {e}")
                raise
    
    @log_execution_time("bi_encoder_rerank")
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[RerankResult]:
        """使用双编码器重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回结果数量
            
        Returns:
            重排序后的结果列表
        """
        if not documents:
            return []
        
        top_k = top_k or defaultConfig.reranker.bi_encoder_top_k
        
        try:
            self.logger.debug(f"开始双编码器重排序: {len(documents)} 个文档")
            
            # 提取文档内容
            doc_texts = [doc.page_content for doc in documents]
            
            # 编码查询和文档
            query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
            doc_embeddings = self.model.encode(
                doc_texts,
                batch_size=defaultConfig.reranker.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(doc_texts) > 20
            )
            
            # 计算相似度分数
            similarities = np.dot(doc_embeddings, query_embedding)
            
            # 创建重排序结果
            rerank_results = []
            for i, (doc, similarity) in enumerate(zip(documents, similarities)):
                rerank_results.append(RerankResult(
                    document=doc,
                    original_score=getattr(doc, 'score', 0.0),
                    bi_encoder_score=float(similarity),
                    final_score=float(similarity),
                    rerank_method="bi_encoder"
                ))
            
            # 按分数排序
            rerank_results.sort(key=lambda x: x.bi_encoder_score, reverse=True)
            
            # 设置排名并返回top_k结果
            final_results = []
            for i, result in enumerate(rerank_results[:top_k]):
                result.rank = i + 1
                final_results.append(result)
            
            self.logger.debug(f"双编码器重排序完成: {len(documents)} -> {len(final_results)}")
            return final_results
            
        except Exception as e:
            self.logger.error(f"双编码器重排序失败: {e}")
            # 回退：返回原始结果
            return [RerankResult(
                document=doc,
                original_score=getattr(doc, 'score', 0.0),
                final_score=getattr(doc, 'score', 0.0),
                rank=i + 1,
                rerank_method="fallback"
            ) for i, doc in enumerate(documents[:top_k])]


class CrossEncoderReranker:
    """交叉编码器重排序器
    
    使用交叉编码器模型对查询-文档对进行联合编码，
    直接输出相关性分数进行重排序。
    
    特点：
    - 查询和文档联合编码
    - 精度高，适合精排阶段
    - 计算成本较高
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """初始化交叉编码器重排序器
        
        Args:
            model_name: 模型名称，默认使用配置中的模型
        """
        self.logger = get_logger("CrossEncoderReranker")
        self.model_name = model_name or defaultConfig.reranker.cross_encoder_model
        self.device = get_device() if defaultConfig.reranker.device == "auto" else defaultConfig.reranker.device
        self.model: CrossEncoder
        
        self._load_model()
    
    def _load_model(self) -> None:
        """加载交叉编码器模型"""
        with LogExecutionTime("load_cross_encoder_model"):
            try:
                self.logger.info(f"正在加载交叉编码器模型: {self.model_name}")
                
                cache_dir = Path(defaultConfig.reranker.cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                self.model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    local_files_only=True
                )
                
                self.logger.info(f"交叉编码器模型加载成功 | 模型: {self.model_name} | 设备: {self.device}")
                
            except Exception as e:
                self.logger.error(f"交叉编码器模型加载失败: {e}")
                raise
    
    @log_execution_time("cross_encoder_rerank")
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[RerankResult]:
        """使用交叉编码器重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回结果数量
            
        Returns:
            重排序后的结果列表
        """
        if not documents:
            return []
        
        top_k = top_k or defaultConfig.reranker.cross_encoder_top_k
        
        try:
            self.logger.debug(f"开始交叉编码器重排序: {len(documents)} 个文档")
            
            # 构建查询-文档对
            query_doc_pairs = [(query, doc.page_content) for doc in documents]
            
            # 计算相关性分数
            scores = self.model.predict(
                query_doc_pairs,
                batch_size=defaultConfig.reranker.batch_size,
                show_progress_bar=len(query_doc_pairs) > 10
            )
            
            # 创建重排序结果
            rerank_results = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                rerank_results.append(RerankResult(
                    document=doc,
                    original_score=getattr(doc, 'score', 0.0),
                    cross_encoder_score=float(score),
                    final_score=float(score),
                    rerank_method="cross_encoder"
                ))
            
            # 按分数排序
            rerank_results.sort(key=lambda x: x.cross_encoder_score, reverse=True)
            
            # 设置排名并返回top_k结果
            final_results = []
            for i, result in enumerate(rerank_results[:top_k]):
                result.rank = i + 1
                final_results.append(result)
            
            self.logger.debug(f"交叉编码器重排序完成: {len(documents)} -> {len(final_results)}")
            return final_results
            
        except Exception as e:
            self.logger.error(f"交叉编码器重排序失败: {e}")
            # 回退：返回原始结果
            return [RerankResult(
                document=doc,
                original_score=getattr(doc, 'score', 0.0),
                final_score=getattr(doc, 'score', 0.0),
                rank=i + 1,
                rerank_method="fallback"
            ) for i, doc in enumerate(documents[:top_k])]


class HybridReranker:
    """混合重排序器

    结合双编码器和交叉编码器的优势，实现多阶段重排序：
    1. 双编码器粗排：快速筛选候选集
    2. 交叉编码器精排：精确排序最终结果
    3. 分数融合：综合多种信号

    特点：
    - 平衡精度和效率
    - 支持多种融合策略
    - 可配置的重排序流水线
    """

    def __init__(self):
        """初始化混合重排序器"""
        self.logger = get_logger("HybridReranker")

        # 初始化子重排序器
        self.bi_encoder = None
        self.cross_encoder = None

        if defaultConfig.reranker.bi_encoder_enabled:
            try:
                self.bi_encoder = BiEncoderReranker()
                self.logger.info("双编码器重排序器初始化成功")
            except Exception as e:
                self.logger.warning(f"双编码器重排序器初始化失败: {e}")

        if defaultConfig.reranker.cross_encoder_enabled:
            try:
                self.cross_encoder = CrossEncoderReranker()
                self.logger.info("交叉编码器重排序器初始化成功")
            except Exception as e:
                self.logger.warning(f"交叉编码器重排序器初始化失败: {e}")

        self.logger.info("混合重排序器初始化完成")

    @log_execution_time("hybrid_rerank")
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[RerankResult]:
        """使用混合重排序器重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 最终返回结果数量

        Returns:
            重排序后的结果列表
        """
        if not documents:
            return []

        top_k = top_k or defaultConfig.reranker.cross_encoder_top_k

        try:
            self.logger.debug(f"开始混合重排序: {len(documents)} 个文档")

            current_docs = documents
            rerank_results = []

            # 阶段1: 双编码器粗排
            if self.bi_encoder and defaultConfig.reranker.bi_encoder_enabled:
                self.logger.debug("执行双编码器粗排")
                bi_results = self.bi_encoder.rerank(
                    query,
                    current_docs,
                    defaultConfig.reranker.bi_encoder_top_k
                )
                current_docs = [result.document for result in bi_results]
                rerank_results = bi_results
                self.logger.debug(f"双编码器粗排完成: {len(documents)} -> {len(current_docs)}")

            # 阶段2: 交叉编码器精排
            if self.cross_encoder and defaultConfig.reranker.cross_encoder_enabled:
                self.logger.debug("执行交叉编码器精排")
                cross_results = self.cross_encoder.rerank(query, current_docs, top_k)

                # 融合双编码器和交叉编码器的分数
                if rerank_results:
                    # 创建双编码器分数映射
                    bi_score_map = {
                        id(result.document): result.bi_encoder_score
                        for result in rerank_results
                    }

                    # 更新交叉编码器结果
                    for result in cross_results:
                        doc_id = id(result.document)
                        if doc_id in bi_score_map:
                            result.bi_encoder_score = bi_score_map[doc_id]

                        # 融合分数
                        result.final_score = self._fuse_scores(
                            result.original_score,
                            result.bi_encoder_score,
                            result.cross_encoder_score
                        )
                        result.rerank_method = "hybrid"

                rerank_results = cross_results
                self.logger.debug(f"交叉编码器精排完成: {len(current_docs)} -> {len(rerank_results)}")

            # 如果没有启用任何重排序器，返回原始结果
            if not rerank_results:
                rerank_results = [RerankResult(
                    document=doc,
                    original_score=getattr(doc, 'score', 0.0),
                    final_score=getattr(doc, 'score', 0.0),
                    rank=i + 1,
                    rerank_method="none"
                ) for i, doc in enumerate(documents[:top_k])]

            # 最终排序
            rerank_results.sort(key=lambda x: x.final_score, reverse=True)

            # 更新排名
            for i, result in enumerate(rerank_results):
                result.rank = i + 1

            self.logger.info(f"混合重排序完成: {len(documents)} -> {len(rerank_results)}")
            return rerank_results

        except Exception as e:
            self.logger.error(f"混合重排序失败: {e}")
            # 回退：返回原始结果
            return [RerankResult(
                document=doc,
                original_score=getattr(doc, 'score', 0.0),
                final_score=getattr(doc, 'score', 0.0),
                rank=i + 1,
                rerank_method="fallback"
            ) for i, doc in enumerate(documents[:top_k])]

    def _fuse_scores(self,
                     original_score: Optional[float],
                     bi_encoder_score: Optional[float],
                     cross_encoder_score: Optional[float]) -> float:
        """融合多种分数

        Args:
            original_score: 原始检索分数
            bi_encoder_score: 双编码器分数
            cross_encoder_score: 交叉编码器分数

        Returns:
            融合后的分数
        """
        fusion_method = defaultConfig.reranker.fusion_method

        # 处理None值
        original_score = original_score or 0.0
        bi_encoder_score = bi_encoder_score or 0.0
        cross_encoder_score = cross_encoder_score or 0.0

        if fusion_method == "weighted":
            # 加权平均
            return (
                defaultConfig.reranker.original_weight * original_score +
                defaultConfig.reranker.bi_encoder_weight * bi_encoder_score +
                defaultConfig.reranker.cross_encoder_weight * cross_encoder_score
            )

        elif fusion_method == "rrf":
            # Reciprocal Rank Fusion
            k = 60  # RRF参数
            rrf_score = 0.0

            if original_score > 0:
                rrf_score += 1.0 / (k + (1.0 / original_score))
            if bi_encoder_score > 0:
                rrf_score += 1.0 / (k + (1.0 / bi_encoder_score))
            if cross_encoder_score > 0:
                rrf_score += 1.0 / (k + (1.0 / cross_encoder_score))

            return rrf_score

        elif fusion_method == "max":
            # 取最大值
            return max(original_score, bi_encoder_score, cross_encoder_score)

        else:
            # 默认返回交叉编码器分数
            return cross_encoder_score

    def get_reranker_info(self) -> Dict[str, Any]:
        """获取重排序器信息"""
        return {
            "bi_encoder_enabled": defaultConfig.reranker.bi_encoder_enabled,
            "cross_encoder_enabled": defaultConfig.reranker.cross_encoder_enabled,
            "bi_encoder_model": defaultConfig.reranker.bi_encoder_model if self.bi_encoder else None,
            "cross_encoder_model": defaultConfig.reranker.cross_encoder_model if self.cross_encoder else None,
            "fusion_method": defaultConfig.reranker.fusion_method,
            "bi_encoder_top_k": defaultConfig.reranker.bi_encoder_top_k,
            "cross_encoder_top_k": defaultConfig.reranker.cross_encoder_top_k,
        }


# 工厂函数
def create_reranker(reranker_type: str = "hybrid") -> Union[BiEncoderReranker, CrossEncoderReranker, HybridReranker]:
    """创建重排序器实例

    Args:
        reranker_type: 重排序器类型 ("bi_encoder", "cross_encoder", "hybrid")

    Returns:
        重排序器实例
    """
    if reranker_type == "bi_encoder":
        return BiEncoderReranker()
    elif reranker_type == "cross_encoder":
        return CrossEncoderReranker()
    elif reranker_type == "hybrid":
        return HybridReranker()
    else:
        raise ValueError(f"不支持的重排序器类型: {reranker_type}")


# 便捷函数
def rerank_documents(query: str,
                    documents: List[Document],
                    reranker_type: str = "hybrid",
                    top_k: Optional[int] = None) -> List[RerankResult]:
    """重排序文档的便捷函数

    Args:
        query: 查询文本
        documents: 文档列表
        reranker_type: 重排序器类型
        top_k: 返回结果数量

    Returns:
        重排序后的结果列表
    """
    reranker = create_reranker(reranker_type)
    return reranker.rerank(query, documents, top_k)
