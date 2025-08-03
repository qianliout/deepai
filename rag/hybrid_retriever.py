"""
混合检索器
融合向量检索和知识图谱检索，提供增强的RAG检索能力
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from graph_retriever import get_graph_retriever, GraphResult, QueryType
from retriever import Retriever
from config import defaultConfig

logger = logging.getLogger("RAG.HybridRetriever")


class RetrievalStrategy(Enum):
    """检索策略枚举"""
    VECTOR_ONLY = "vector_only"           # 仅向量检索
    GRAPH_ONLY = "graph_only"             # 仅图检索
    PARALLEL = "parallel"                 # 并行检索
    SEQUENTIAL = "sequential"             # 顺序检索
    ADAPTIVE = "adaptive"                 # 自适应检索


@dataclass
class HybridResult:
    """混合检索结果"""
    vector_results: List[Dict[str, Any]]
    graph_result: GraphResult
    fused_context: str
    strategy_used: RetrievalStrategy
    total_sources: int
    confidence_score: float


class ResultFuser:
    """结果融合器"""
    
    def __init__(self):
        self.max_context_length = 4000  # 最大上下文长度
    
    def fuse_results(self, vector_results: List[Dict[str, Any]], 
                    graph_result: GraphResult, 
                    query: str) -> str:
        """融合向量检索和图检索结果"""
        
        # 1. 构建图谱知识部分
        graph_context = ""
        if graph_result.structured_data:
            graph_context = f"【知识图谱信息】\n{graph_result.summary}\n"
            
            # 添加结构化事实
            if graph_result.query_type == QueryType.VULNERABILITY_IMPACT:
                graph_context += self._format_vulnerability_facts(graph_result.structured_data)
            elif graph_result.query_type == QueryType.HOST_RISK_ASSESSMENT:
                graph_context += self._format_host_risk_facts(graph_result.structured_data)
            elif graph_result.query_type == QueryType.PRIORITY_RANKING:
                graph_context += self._format_priority_facts(graph_result.structured_data)
            
            graph_context += f"（数据来源：{graph_result.source}）\n\n"
        
        # 2. 构建向量检索部分
        vector_context = ""
        if vector_results:
            vector_context = "【文档检索信息】\n"
            for i, result in enumerate(vector_results[:5], 1):  # 限制前5个结果
                content = result.get('content', result.get('page_content', ''))
                source = result.get('source', result.get('metadata', {}).get('source', '未知'))
                score = result.get('score', 0.0)
                
                vector_context += f"{i}. {content[:200]}...\n"
                vector_context += f"   （来源：{source}，相似度：{score:.3f}）\n\n"
        
        # 3. 融合上下文
        fused_context = ""
        
        # 优先级：图谱结果 > 向量结果
        if graph_context:
            fused_context += graph_context
        
        if vector_context:
            fused_context += vector_context
        
        # 4. 长度控制
        if len(fused_context) > self.max_context_length:
            # 优先保留图谱信息
            if graph_context and len(graph_context) < self.max_context_length:
                remaining_length = self.max_context_length - len(graph_context)
                truncated_vector = vector_context[:remaining_length]
                fused_context = graph_context + truncated_vector + "\n...(内容已截断)"
            else:
                fused_context = fused_context[:self.max_context_length] + "\n...(内容已截断)"
        
        return fused_context
    
    def _format_vulnerability_facts(self, data: List[Dict[str, Any]]) -> str:
        """格式化漏洞影响事实"""
        if not data:
            return ""
        
        facts = "结构化事实：\n"
        
        # 按主机分组
        hosts = {}
        for item in data:
            hostname = item['hostname']
            if hostname not in hosts:
                hosts[hostname] = []
            hosts[hostname].append(item)
        
        for hostname, items in list(hosts.items())[:3]:  # 限制显示3台主机
            facts += f"• 主机 {hostname} ({items[0]['ip_address']}):\n"
            for item in items[:3]:  # 每台主机最多3个镜像
                facts += f"  - 镜像 {item['image_name']}:{item['image_tag']} 包含漏洞 {item['cve_id']}\n"
        
        return facts
    
    def _format_host_risk_facts(self, data: List[Dict[str, Any]]) -> str:
        """格式化主机风险事实"""
        if not data:
            return ""
        
        facts = "结构化事实：\n"
        
        # 按严重程度分组
        by_severity = {}
        for item in data:
            severity = item['severity']
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(item)
        
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if severity in by_severity:
                items = by_severity[severity][:3]  # 每个级别最多3个
                facts += f"• {severity}级漏洞 ({len(by_severity[severity])}个):\n"
                for item in items:
                    facts += f"  - {item['cve_id']}: 镜像 {item['image_name']}:{item['image_tag']}\n"
        
        return facts
    
    def _format_priority_facts(self, data: List[Dict[str, Any]]) -> str:
        """格式化优先级事实"""
        if not data:
            return ""
        
        facts = "结构化事实：\n"
        for i, item in enumerate(data[:5], 1):  # 前5个优先级
            facts += f"• 优先级{i}: {item['cve_id']} ({item['severity']}) - "
            facts += f"影响{item['affected_hosts']}台主机，{item['affected_images']}个镜像\n"
        
        return facts


class StrategySelector:
    """检索策略选择器"""
    
    def __init__(self):
        self.graph_keywords = [
            'CVE-', '漏洞', '主机', '镜像', '风险', '影响', '优先级', 
            '修复', '安全', '评估', 'nginx', 'mysql', 'redis'
        ]
    
    def select_strategy(self, query: str) -> RetrievalStrategy:
        """根据查询选择检索策略"""
        query_lower = query.lower()
        
        # 检查是否包含图谱相关关键词
        graph_score = sum(1 for keyword in self.graph_keywords if keyword in query_lower)
        
        # 检查查询长度和复杂度
        query_length = len(query)
        
        if graph_score >= 2:
            return RetrievalStrategy.PARALLEL  # 高图谱相关性，并行检索
        elif graph_score >= 1:
            return RetrievalStrategy.ADAPTIVE  # 中等相关性，自适应
        elif query_length > 50:
            return RetrievalStrategy.PARALLEL  # 复杂查询，并行检索
        else:
            return RetrievalStrategy.VECTOR_ONLY  # 简单查询，仅向量检索


class HybridRetriever:
    """混合检索器主类"""
    
    def __init__(self):
        self.vector_retriever = Retriever()
        self.graph_retriever = get_graph_retriever()
        self.result_fuser = ResultFuser()
        self.strategy_selector = StrategySelector()
        
        logger.info("混合检索器初始化完成")
    
    def retrieve(self, query: str, strategy: Optional[RetrievalStrategy] = None) -> HybridResult:
        """执行混合检索"""
        try:
            # 1. 选择检索策略
            if strategy is None:
                strategy = self.strategy_selector.select_strategy(query)
            
            logger.info(f"使用检索策略: {strategy.value}")
            
            # 2. 根据策略执行检索
            if strategy == RetrievalStrategy.VECTOR_ONLY:
                return self._vector_only_retrieve(query, strategy)
            elif strategy == RetrievalStrategy.GRAPH_ONLY:
                return self._graph_only_retrieve(query, strategy)
            elif strategy == RetrievalStrategy.PARALLEL:
                return self._parallel_retrieve(query, strategy)
            elif strategy == RetrievalStrategy.SEQUENTIAL:
                return self._sequential_retrieve(query, strategy)
            else:  # ADAPTIVE
                return self._adaptive_retrieve(query, strategy)
                
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            # 降级到向量检索
            return self._vector_only_retrieve(query, RetrievalStrategy.VECTOR_ONLY)
    
    def _vector_only_retrieve(self, query: str, strategy: RetrievalStrategy) -> HybridResult:
        """仅向量检索"""
        vector_results = self.vector_retriever.retrieve(query)
        
        # 创建空的图检索结果
        from graph_retriever import GraphResult, QueryType
        empty_graph_result = GraphResult(
            query_type=QueryType.UNKNOWN,
            structured_data=[],
            summary="",
            confidence=0.0
        )
        
        fused_context = self.result_fuser.fuse_results(vector_results, empty_graph_result, query)
        
        return HybridResult(
            vector_results=vector_results,
            graph_result=empty_graph_result,
            fused_context=fused_context,
            strategy_used=strategy,
            total_sources=len(vector_results),
            confidence_score=0.7
        )
    
    def _graph_only_retrieve(self, query: str, strategy: RetrievalStrategy) -> HybridResult:
        """仅图检索"""
        graph_result = self.graph_retriever.retrieve(query)
        
        fused_context = self.result_fuser.fuse_results([], graph_result, query)
        
        return HybridResult(
            vector_results=[],
            graph_result=graph_result,
            fused_context=fused_context,
            strategy_used=strategy,
            total_sources=len(graph_result.structured_data),
            confidence_score=graph_result.confidence
        )
    
    def _parallel_retrieve(self, query: str, strategy: RetrievalStrategy) -> HybridResult:
        """并行检索"""
        # 同时执行向量检索和图检索
        vector_results = self.vector_retriever.retrieve(query)
        graph_result = self.graph_retriever.retrieve(query)
        
        # 融合结果
        fused_context = self.result_fuser.fuse_results(vector_results, graph_result, query)
        
        # 计算综合置信度
        vector_confidence = 0.7 if vector_results else 0.0
        graph_confidence = graph_result.confidence
        combined_confidence = max(vector_confidence, graph_confidence)
        
        return HybridResult(
            vector_results=vector_results,
            graph_result=graph_result,
            fused_context=fused_context,
            strategy_used=strategy,
            total_sources=len(vector_results) + len(graph_result.structured_data),
            confidence_score=combined_confidence
        )
    
    def _sequential_retrieve(self, query: str, strategy: RetrievalStrategy) -> HybridResult:
        """顺序检索 - 先图后向量"""
        # 先执行图检索
        graph_result = self.graph_retriever.retrieve(query)
        
        # 如果图检索结果置信度高，可能不需要向量检索
        if graph_result.confidence > 0.8:
            vector_results = []
        else:
            vector_results = self.vector_retriever.retrieve(query)
        
        fused_context = self.result_fuser.fuse_results(vector_results, graph_result, query)
        
        return HybridResult(
            vector_results=vector_results,
            graph_result=graph_result,
            fused_context=fused_context,
            strategy_used=strategy,
            total_sources=len(vector_results) + len(graph_result.structured_data),
            confidence_score=max(0.7 if vector_results else 0.0, graph_result.confidence)
        )
    
    def _adaptive_retrieve(self, query: str, strategy: RetrievalStrategy) -> HybridResult:
        """自适应检索"""
        # 先快速评估查询类型
        graph_result = self.graph_retriever.retrieve(query)
        
        # 根据图检索结果决定是否需要向量检索
        if graph_result.confidence > 0.6 and graph_result.structured_data:
            # 图检索效果好，轻量级向量检索
            vector_results = self.vector_retriever.retrieve(query)[:3]  # 只取前3个
        else:
            # 图检索效果一般，完整向量检索
            vector_results = self.vector_retriever.retrieve(query)
        
        fused_context = self.result_fuser.fuse_results(vector_results, graph_result, query)
        
        return HybridResult(
            vector_results=vector_results,
            graph_result=graph_result,
            fused_context=fused_context,
            strategy_used=strategy,
            total_sources=len(vector_results) + len(graph_result.structured_data),
            confidence_score=max(0.7 if vector_results else 0.0, graph_result.confidence)
        )
    
    def get_enhanced_context(self, query: str) -> str:
        """获取增强的上下文（主要接口）"""
        result = self.retrieve(query)
        return result.fused_context


# 全局实例
hybrid_retriever = None

def get_hybrid_retriever() -> HybridRetriever:
    """获取混合检索器单例"""
    global hybrid_retriever
    if hybrid_retriever is None:
        hybrid_retriever = HybridRetriever()
    return hybrid_retriever


if __name__ == "__main__":
    # 测试代码
    retriever = get_hybrid_retriever()
    
    test_queries = [
        "CVE-2023-44487影响了哪些主机和镜像？",
        "web-server-01存在哪些安全风险？",
        "当前最需要优先修复的漏洞是什么？",
        "如何配置nginx的安全设置？",
        "什么是容器安全？"
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        result = retriever.retrieve(query)
        print(f"策略: {result.strategy_used.value}")
        print(f"置信度: {result.confidence_score:.2f}")
        print(f"数据源数量: {result.total_sources}")
        print(f"上下文长度: {len(result.fused_context)}")
        print("=" * 80)
