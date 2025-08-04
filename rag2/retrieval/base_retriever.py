"""
基础检索器
定义检索器的基础接口和通用功能
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ..config.config import get_config
    from ..utils.logger import get_logger, log_performance, log_retrieval
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from config.config import get_config
    from utils.logger import get_logger, log_performance, log_retrieval

logger = get_logger("base_retriever")

class BaseRetriever(ABC):
    """检索器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.config = get_config()
        logger.info(f"检索器初始化: {self.name}")
    
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 检索结果列表
        """
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        添加文档到检索索引
        
        Args:
            documents: 文档列表
            
        Returns:
            bool: 是否成功
        """
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """
        从检索索引中删除文档
        
        Args:
            document_id: 文档ID
            
        Returns:
            bool: 是否成功
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            bool: 是否健康
        """
        pass
    
    def get_name(self) -> str:
        """获取检索器名称"""
        return self.name
    
    def format_results(self, results: List[Dict[str, Any]], 
                      query: str, method: str) -> List[Dict[str, Any]]:
        """格式化检索结果"""
        formatted_results = []
        
        for i, result in enumerate(results):
            formatted_result = {
                "rank": i + 1,
                "retriever": self.name,
                "method": method,
                "query": query,
                **result
            }
            formatted_results.append(formatted_result)
        
        return formatted_results

class VectorRetriever(BaseRetriever):
    """向量检索器基类"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.embedding_manager = None
    
    async def _get_embedding_manager(self):
        """延迟加载嵌入管理器"""
        if self.embedding_manager is None:
            from ..models.embeddings import get_embedding_manager
            self.embedding_manager = get_embedding_manager()
        return self.embedding_manager
    
    async def encode_query(self, query: str) -> List[float]:
        """编码查询文本"""
        embedding_manager = await self._get_embedding_manager()
        return await embedding_manager.encode_text_async(query)

class KeywordRetriever(BaseRetriever):
    """关键词检索器基类"""
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        import jieba
        import jieba.analyse
        
        # 使用TF-IDF提取关键词
        keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False)
        return keywords
    
    def build_query(self, keywords: List[str]) -> str:
        """构建查询字符串"""
        return " ".join(keywords)

class HybridRetriever(BaseRetriever):
    """混合检索器基类"""
    
    def __init__(self, name: str, retrievers: List[BaseRetriever], weights: List[float] = None):
        super().__init__(name)
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        
        if len(self.weights) != len(self.retrievers):
            raise ValueError("权重数量必须与检索器数量相同")
    
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """混合检索"""
        # 并发执行所有检索器
        tasks = []
        for retriever in self.retrievers:
            task = retriever.retrieve(query, top_k * 2, **kwargs)  # 获取更多结果用于融合
            tasks.append(task)
        
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果和异常
        valid_results = []
        for i, results in enumerate(results_list):
            if isinstance(results, Exception):
                logger.warning(f"检索器 {self.retrievers[i].name} 失败: {str(results)}")
            else:
                valid_results.append((results, self.weights[i]))
        
        if not valid_results:
            return []
        
        # 融合结果
        fused_results = self._fuse_results(valid_results, top_k)
        
        return self.format_results(fused_results, query, "hybrid")
    
    def _fuse_results(self, results_list: List[tuple], top_k: int) -> List[Dict[str, Any]]:
        """融合多个检索器的结果"""
        # 简单的分数加权融合
        doc_scores = {}
        
        for results, weight in results_list:
            for i, result in enumerate(results):
                doc_id = result.get("id", f"doc_{i}")
                
                # 计算归一化分数 (排名越靠前分数越高)
                normalized_score = (len(results) - i) / len(results)
                weighted_score = normalized_score * weight
                
                if doc_id in doc_scores:
                    doc_scores[doc_id]["score"] += weighted_score
                    doc_scores[doc_id]["sources"].append(result.get("retriever", "unknown"))
                else:
                    doc_scores[doc_id] = {
                        "score": weighted_score,
                        "document": result,
                        "sources": [result.get("retriever", "unknown")]
                    }
        
        # 按分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # 构建最终结果
        fused_results = []
        for doc_id, doc_info in sorted_docs[:top_k]:
            result = doc_info["document"].copy()
            result["fusion_score"] = doc_info["score"]
            result["fusion_sources"] = doc_info["sources"]
            fused_results.append(result)
        
        return fused_results
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """向所有检索器添加文档"""
        tasks = [retriever.add_documents(documents) for retriever in self.retrievers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is True)
        return success_count > 0
    
    async def delete_document(self, document_id: str) -> bool:
        """从所有检索器删除文档"""
        tasks = [retriever.delete_document(document_id) for retriever in self.retrievers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is True)
        return success_count > 0
    
    async def health_check(self) -> bool:
        """检查所有检索器健康状态"""
        tasks = [retriever.health_check() for retriever in self.retrievers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        healthy_count = sum(1 for result in results if result is True)
        return healthy_count > 0

class RetrieverManager:
    """检索器管理器"""
    
    def __init__(self):
        self.retrievers: Dict[str, BaseRetriever] = {}
        self.default_retriever: Optional[str] = None
    
    def register_retriever(self, retriever: BaseRetriever, is_default: bool = False):
        """注册检索器"""
        self.retrievers[retriever.name] = retriever
        
        if is_default or self.default_retriever is None:
            self.default_retriever = retriever.name
        
        logger.info(f"检索器已注册: {retriever.name}")
    
    def get_retriever(self, name: str = None) -> Optional[BaseRetriever]:
        """获取检索器"""
        if name is None:
            name = self.default_retriever
        
        return self.retrievers.get(name)
    
    def list_retrievers(self) -> List[str]:
        """列出所有检索器"""
        return list(self.retrievers.keys())
    
    async def retrieve(self, query: str, retriever_name: str = None, 
                      top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """使用指定检索器进行检索"""
        retriever = self.get_retriever(retriever_name)
        
        if retriever is None:
            raise ValueError(f"检索器不存在: {retriever_name}")
        
        return await retriever.retrieve(query, top_k, **kwargs)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """检查所有检索器健康状态"""
        results = {}
        
        for name, retriever in self.retrievers.items():
            try:
                results[name] = await retriever.health_check()
            except Exception as e:
                logger.error(f"检索器 {name} 健康检查失败: {str(e)}")
                results[name] = False
        
        return results

# 全局检索器管理器
_retriever_manager = None

def get_retriever_manager() -> RetrieverManager:
    """获取检索器管理器实例"""
    global _retriever_manager
    if _retriever_manager is None:
        _retriever_manager = RetrieverManager()
    return _retriever_manager
