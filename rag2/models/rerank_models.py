"""
重排序模型管理器
统一管理BAAI/BGE重排序模型
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from FlagEmbedding import FlagReranker
import torch

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ..config.config import get_config, get_model_config
    from ..utils.logger import get_logger, log_performance, log_model_call
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from config.config import get_config, get_model_config
    from utils.logger import get_logger, log_performance, log_model_call

logger = get_logger("rerank_models")

class RerankManager:
    """重排序模型管理器"""
    
    def __init__(self):
        self.config = get_config()
        self.model_config = get_model_config()
        
        self.rerank_model = None
        self.model_name = None
        self.device = None
        self.use_fp16 = False
        
        # 初始化模型
        self._init_model()
    
    def _init_model(self):
        """初始化重排序模型"""
        rerank_config = self.model_config["reranker"]
        
        self.model_name = rerank_config["model_name"]
        self.device = rerank_config.get("device", "cpu")
        self.use_fp16 = rerank_config.get("use_fp16", False)
        
        try:
            # 检查设备可用性
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA不可用，切换到CPU")
                self.device = "cpu"
                self.use_fp16 = False  # CPU不支持fp16
            
            # 加载重排序模型
            self.rerank_model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device
            )
            
            logger.info(f"重排序模型加载成功: {self.model_name} (设备: {self.device}, FP16: {self.use_fp16})")
            
        except Exception as e:
            logger.error(f"重排序模型加载失败: {str(e)}")
            raise
    
    @log_performance()
    def rerank_documents(self, query: str, documents: List[str], 
                        top_k: int = None) -> List[Tuple[int, float]]:
        """重排序文档"""
        try:
            if not documents:
                return []
            
            start_time = time.time()
            
            # 准备查询-文档对
            query_doc_pairs = [[query, doc] for doc in documents]
            
            # 计算重排序分数
            scores = self.rerank_model.compute_score(query_doc_pairs)
            
            # 如果只有一个文档，scores可能是单个值而不是列表
            if not isinstance(scores, list):
                scores = [scores]
            
            # 创建(索引, 分数)对并排序
            indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 应用top_k限制
            if top_k is not None:
                indexed_scores = indexed_scores[:top_k]
            
            # 记录模型调用
            end_time = time.time()
            duration = end_time - start_time
            
            total_input_tokens = len(query) // 4 + sum(len(doc) // 4 for doc in documents)
            
            log_model_call(
                model_name=self.model_name,
                input_tokens=total_input_tokens,
                output_tokens=len(indexed_scores),
                duration=duration
            )
            
            logger.info(f"重排序完成: 输入{len(documents)}个文档, 输出{len(indexed_scores)}个结果")
            return indexed_scores
            
        except Exception as e:
            logger.error(f"文档重排序失败: {str(e)}")
            # 返回原始顺序
            return [(i, 0.0) for i in range(len(documents))]
    
    async def rerank_documents_async(self, query: str, documents: List[str], 
                                   top_k: int = None) -> List[Tuple[int, float]]:
        """异步重排序文档"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank_documents, query, documents, top_k)
    
    @log_performance()
    def rerank_search_results(self, query: str, 
                            search_results: List[Dict[str, Any]], 
                            content_field: str = "content",
                            top_k: int = None) -> List[Dict[str, Any]]:
        """重排序搜索结果"""
        try:
            if not search_results:
                return []
            
            # 提取文档内容
            documents = []
            for result in search_results:
                content = result.get(content_field, "")
                if isinstance(content, str):
                    documents.append(content)
                else:
                    documents.append(str(content))
            
            # 执行重排序
            rerank_results = self.rerank_documents(query, documents, top_k)
            
            # 重新组织结果
            reranked_search_results = []
            for original_index, rerank_score in rerank_results:
                result = search_results[original_index].copy()
                result["rerank_score"] = rerank_score
                result["original_index"] = original_index
                reranked_search_results.append(result)
            
            logger.info(f"搜索结果重排序完成: {len(reranked_search_results)}个结果")
            return reranked_search_results
            
        except Exception as e:
            logger.error(f"搜索结果重排序失败: {str(e)}")
            return search_results
    
    async def rerank_search_results_async(self, query: str, 
                                        search_results: List[Dict[str, Any]], 
                                        content_field: str = "content",
                                        top_k: int = None) -> List[Dict[str, Any]]:
        """异步重排序搜索结果"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.rerank_search_results, query, search_results, content_field, top_k
        )
    
    def compute_relevance_score(self, query: str, document: str) -> float:
        """计算单个文档的相关性分数"""
        try:
            score = self.rerank_model.compute_score([query, document])
            return float(score)
        except Exception as e:
            logger.error(f"相关性分数计算失败: {str(e)}")
            return 0.0
    
    async def compute_relevance_score_async(self, query: str, document: str) -> float:
        """异步计算单个文档的相关性分数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.compute_relevance_score, query, document)
    
    def batch_compute_scores(self, query_doc_pairs: List[List[str]]) -> List[float]:
        """批量计算查询-文档对的相关性分数"""
        try:
            start_time = time.time()
            
            scores = self.rerank_model.compute_score(query_doc_pairs)
            
            # 确保返回列表
            if not isinstance(scores, list):
                scores = [scores]
            
            # 记录模型调用
            end_time = time.time()
            duration = end_time - start_time
            
            total_input_tokens = sum(
                len(pair[0]) // 4 + len(pair[1]) // 4 for pair in query_doc_pairs
            )
            
            log_model_call(
                model_name=self.model_name,
                input_tokens=total_input_tokens,
                output_tokens=len(scores),
                duration=duration
            )
            
            return [float(score) for score in scores]
            
        except Exception as e:
            logger.error(f"批量相关性分数计算失败: {str(e)}")
            return [0.0] * len(query_doc_pairs)
    
    async def batch_compute_scores_async(self, query_doc_pairs: List[List[str]]) -> List[float]:
        """异步批量计算查询-文档对的相关性分数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.batch_compute_scores, query_doc_pairs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "max_length": getattr(self.rerank_model, 'max_length', 512)
        }
    
    def warm_up(self):
        """模型预热"""
        try:
            logger.info("开始重排序模型预热...")
            
            # 使用一些示例查询-文档对进行预热
            warm_up_pairs = [
                ["CVE漏洞查询", "这是一个关于CVE-2024-1234漏洞的描述"],
                ["主机安全状态", "主机192.168.1.100的安全状态良好"],
                ["镜像漏洞检查", "nginx:latest镜像存在高危漏洞"],
                ["修复建议", "建议立即更新到最新版本以修复安全漏洞"]
            ]
            
            start_time = time.time()
            self.batch_compute_scores(warm_up_pairs)
            end_time = time.time()
            
            logger.info(f"重排序模型预热完成，耗时: {end_time - start_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"重排序模型预热失败: {str(e)}")
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 测试计算一个简单的相关性分数
            test_query = "健康检查"
            test_document = "这是一个健康检查测试文档"
            
            score = self.compute_relevance_score(test_query, test_document)
            
            # 检查分数是否在合理范围内
            if not isinstance(score, (int, float)) or score < -10 or score > 10:
                logger.error(f"重排序分数异常: {score}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"重排序模型健康检查失败: {str(e)}")
            return False

class MultiStageReranker:
    """多阶段重排序器"""
    
    def __init__(self, rerank_manager: RerankManager):
        self.rerank_manager = rerank_manager
        self.config = get_config()
    
    @log_performance()
    def multi_stage_rerank(self, query: str, 
                          search_results: List[Dict[str, Any]],
                          stage1_top_k: int = 20,
                          stage2_top_k: int = 10,
                          content_field: str = "content") -> List[Dict[str, Any]]:
        """多阶段重排序"""
        try:
            if not search_results:
                return []
            
            logger.info(f"开始多阶段重排序: 输入{len(search_results)}个结果")
            
            # 第一阶段：基于原始相似度分数的粗排
            stage1_results = search_results[:stage1_top_k]
            logger.info(f"第一阶段粗排: 保留前{len(stage1_results)}个结果")
            
            # 第二阶段：使用重排序模型精排
            stage2_results = self.rerank_manager.rerank_search_results(
                query, stage1_results, content_field, stage2_top_k
            )
            logger.info(f"第二阶段精排: 输出{len(stage2_results)}个结果")
            
            return stage2_results
            
        except Exception as e:
            logger.error(f"多阶段重排序失败: {str(e)}")
            return search_results[:stage2_top_k]
    
    async def multi_stage_rerank_async(self, query: str, 
                                     search_results: List[Dict[str, Any]],
                                     stage1_top_k: int = 20,
                                     stage2_top_k: int = 10,
                                     content_field: str = "content") -> List[Dict[str, Any]]:
        """异步多阶段重排序"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.multi_stage_rerank, query, search_results, 
            stage1_top_k, stage2_top_k, content_field
        )

# 全局重排序管理器实例
_rerank_manager = None

def get_rerank_manager() -> RerankManager:
    """获取重排序管理器实例"""
    global _rerank_manager
    if _rerank_manager is None:
        _rerank_manager = RerankManager()
    return _rerank_manager
