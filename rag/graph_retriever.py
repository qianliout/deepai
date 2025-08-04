"""
知识图谱检索器
实现基于Neo4j的结构化知识检索和智能查询理解
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from neo4j_manager import get_neo4j_manager
from config import defaultConfig

logger = logging.getLogger("RAG.GraphRetriever")


class QueryType(Enum):
    """查询类型枚举"""

    VULNERABILITY_IMPACT = "vulnerability_impact"  # 漏洞影响分析
    HOST_RISK_ASSESSMENT = "host_risk_assessment"  # 主机风险评估
    PRIORITY_RANKING = "priority_ranking"  # 修复优先级
    GENERAL_SEARCH = "general_search"  # 通用搜索
    UNKNOWN = "unknown"  # 未知类型


@dataclass
class GraphResult:
    """图检索结果"""

    query_type: QueryType
    structured_data: List[Dict[str, Any]]
    summary: str
    confidence: float
    source: str = "knowledge_graph"


class QueryClassifier:
    """查询分类器 - 识别用户意图"""

    def __init__(self):
        self.patterns = {
            QueryType.VULNERABILITY_IMPACT: [
                r"CVE-\d{4}-\d+",
                r"漏洞.*影响",
                r"哪些.*主机.*受影响",
                r"哪些.*镜像.*受影响",
                r".*漏洞.*涉及.*主机",
                r".*漏洞.*涉及.*镜像",
            ],
            QueryType.HOST_RISK_ASSESSMENT: [
                r"主机.*风险",
                r"主机.*漏洞",
                r".*服务器.*安全",
                r".*主机.*安全评估",
                r".*主机.*存在.*漏洞",
            ],
            QueryType.PRIORITY_RANKING: [
                r"优先级",
                r"紧急.*修复",
                r"最严重.*漏洞",
                r"修复.*顺序",
                r"哪个.*先修复",
            ],
        }

    def classify(self, query: str) -> Tuple[QueryType, float]:
        """分类查询并返回置信度"""
        query_lower = query.lower()

        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # 简单的置信度计算
                    confidence = 0.8 if "CVE-" in query else 0.7
                    return query_type, confidence

        return QueryType.GENERAL_SEARCH, 0.5

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """从查询中提取实体"""
        entities = {
            "cve_ids": [],
            "hostnames": [],
            "ip_addresses": [],
            "image_names": [],
        }

        # 提取CVE ID
        cve_pattern = r"CVE-\d{4}-\d+"
        entities["cve_ids"] = re.findall(cve_pattern, query, re.IGNORECASE)

        # 提取IP地址
        ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        entities["ip_addresses"] = re.findall(ip_pattern, query)

        # 提取主机名（简单模式）
        hostname_patterns = [
            r"web-server-\d+",
            r"db-server-\d+",
            r"app-server-\d+",
            r"[a-zA-Z]+-server-\d+",
        ]
        for pattern in hostname_patterns:
            entities["hostnames"].extend(re.findall(pattern, query, re.IGNORECASE))

        # 提取镜像名
        image_patterns = [
            r"nginx:\d+\.\d+\.\d+",
            r"mysql:\d+\.\d+\.\d+",
            r"redis:\d+\.\d+\.\d+",
            r"[a-zA-Z]+:\d+\.\d+\.\d+",
        ]
        for pattern in image_patterns:
            entities["image_names"].extend(re.findall(pattern, query, re.IGNORECASE))

        return entities


class GraphRetriever:
    """知识图谱检索器"""

    def __init__(self):
        self.neo4j_manager = get_neo4j_manager()
        self.classifier = QueryClassifier()
        self.max_results = 20

    def retrieve(self, query: str) -> GraphResult:
        """主检索方法"""
        try:
            # 1. 分类查询
            query_type, confidence = self.classifier.classify(query)
            logger.info(f"查询分类: {query_type.value}, 置信度: {confidence}")

            # 2. 提取实体
            entities = self.classifier.extract_entities(query)
            logger.info(f"提取实体: {entities}")

            # 3. 根据类型执行相应查询
            if query_type == QueryType.VULNERABILITY_IMPACT:
                return self._handle_vulnerability_impact(entities, confidence)
            elif query_type == QueryType.HOST_RISK_ASSESSMENT:
                return self._handle_host_risk_assessment(entities, confidence)
            elif query_type == QueryType.PRIORITY_RANKING:
                return self._handle_priority_ranking(confidence)
            else:
                return self._handle_general_search(query, confidence)

        except Exception as e:
            logger.error(f"图检索失败: {e}")
            return GraphResult(
                query_type=QueryType.UNKNOWN,
                structured_data=[],
                summary=f"图检索出现错误: {str(e)}",
                confidence=0.0,
            )

    def _handle_vulnerability_impact(
        self, entities: Dict[str, List[str]], confidence: float
    ) -> GraphResult:
        """处理漏洞影响分析查询"""
        results = []

        if entities["cve_ids"]:
            for cve_id in entities["cve_ids"]:
                impact_data = self.neo4j_manager.get_vulnerability_impact(cve_id)
                results.extend(impact_data)

        if not results:
            summary = "未找到相关漏洞影响信息"
        else:
            affected_hosts = set(r["hostname"] for r in results)
            affected_images = set(
                f"{r['image_name']}:{r['image_tag']}" for r in results
            )

            summary = f"漏洞影响分析：\n"
            summary += f"• 受影响主机数量: {len(affected_hosts)}\n"
            summary += f"• 受影响镜像数量: {len(affected_images)}\n"
            summary += f"• 主机列表: {', '.join(list(affected_hosts)[:5])}"
            if len(affected_hosts) > 5:
                summary += f" 等{len(affected_hosts)}台主机"

        return GraphResult(
            query_type=QueryType.VULNERABILITY_IMPACT,
            structured_data=results,
            summary=summary,
            confidence=confidence,
        )

    def _handle_host_risk_assessment(
        self, entities: Dict[str, List[str]], confidence: float
    ) -> GraphResult:
        """处理主机风险评估查询"""
        results = []

        # 优先使用明确的主机名
        hostnames = entities["hostnames"]
        if not hostnames and entities["ip_addresses"]:
            # 如果没有主机名但有IP，尝试通过IP查找主机
            for ip in entities["ip_addresses"]:
                host_query = (
                    "MATCH (h:Host {ip_address: $ip}) RETURN h.hostname as hostname"
                )
                host_result = self.neo4j_manager.execute_query(host_query, {"ip": ip})
                if host_result:
                    hostnames.append(host_result[0]["hostname"])

        if hostnames:
            for hostname in hostnames:
                risk_data = self.neo4j_manager.get_host_risk_assessment(hostname)
                results.extend(risk_data)

        if not results:
            summary = "未找到相关主机风险信息"
        else:
            total_vulns = len(results)
            critical_vulns = len([r for r in results if r["severity"] == "CRITICAL"])
            high_vulns = len([r for r in results if r["severity"] == "HIGH"])
            max_cvss = max(r["cvss_score"] for r in results) if results else 0

            summary = f"主机风险评估：\n"
            summary += f"• 总漏洞数量: {total_vulns}\n"
            summary += f"• 严重漏洞: {critical_vulns}个\n"
            summary += f"• 高危漏洞: {high_vulns}个\n"
            summary += f"• 最高CVSS评分: {max_cvss}"

        return GraphResult(
            query_type=QueryType.HOST_RISK_ASSESSMENT,
            structured_data=results,
            summary=summary,
            confidence=confidence,
        )

    def _handle_priority_ranking(self, confidence: float) -> GraphResult:
        """处理修复优先级查询"""
        results = self.neo4j_manager.get_vulnerability_priority_ranking()

        if not results:
            summary = "未找到漏洞优先级信息"
        else:
            top_vulns = results[:5]  # 取前5个
            summary = "漏洞修复优先级排序（前5个）：\n"
            for i, vuln in enumerate(top_vulns, 1):
                summary += f"{i}. {vuln['cve_id']} ({vuln['severity']}) - "
                summary += f"影响{vuln['affected_hosts']}台主机，{vuln['affected_images']}个镜像\n"

        return GraphResult(
            query_type=QueryType.PRIORITY_RANKING,
            structured_data=results,
            summary=summary,
            confidence=confidence,
        )

    def _handle_general_search(self, query: str, confidence: float) -> GraphResult:
        """处理通用搜索查询"""
        # 提取关键词
        keywords = self._extract_keywords(query)
        results = []

        for keyword in keywords:
            search_results = self.neo4j_manager.search_by_keyword(keyword)
            results.extend(search_results)

        # 去重
        seen = set()
        unique_results = []
        for result in results:
            key = str(result["node_data"])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        if not unique_results:
            summary = f"未找到与'{query}'相关的信息"
        else:
            node_types = {}
            for result in unique_results:
                node_type = result["node_type"][0] if result["node_type"] else "Unknown"
                node_types[node_type] = node_types.get(node_type, 0) + 1

            summary = f"搜索结果：\n"
            for node_type, count in node_types.items():
                summary += f"• {node_type}: {count}个\n"

        return GraphResult(
            query_type=QueryType.GENERAL_SEARCH,
            structured_data=unique_results,
            summary=summary,
            confidence=confidence,
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词"""
        # 简单的关键词提取
        stop_words = {
            "的",
            "是",
            "在",
            "有",
            "和",
            "与",
            "或",
            "但",
            "如果",
            "那么",
            "这个",
            "那个",
        }
        words = re.findall(r"\w+", query.lower())
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]
        return keywords[:3]  # 最多取3个关键词

    def format_for_llm(self, graph_result: GraphResult) -> str:
        """格式化图检索结果供LLM使用"""
        if not graph_result.structured_data:
            return f"知识图谱检索结果：{graph_result.summary}"

        formatted = f"知识图谱检索结果（{graph_result.query_type.value}）：\n"
        formatted += f"{graph_result.summary}\n\n"

        if graph_result.query_type == QueryType.VULNERABILITY_IMPACT:
            formatted += "详细影响信息：\n"
            for item in graph_result.structured_data[:10]:  # 限制数量
                formatted += f"• 主机: {item['hostname']} ({item['ip_address']}) "
                formatted += f"镜像: {item['image_name']}:{item['image_tag']} "
                formatted += f"漏洞: {item['cve_id']} ({item['severity']})\n"

        elif graph_result.query_type == QueryType.HOST_RISK_ASSESSMENT:
            formatted += "详细风险信息：\n"
            for item in graph_result.structured_data[:10]:
                formatted += f"• {item['cve_id']} ({item['severity']}, CVSS: {item['cvss_score']}) "
                formatted += f"镜像: {item['image_name']}:{item['image_tag']}\n"

        elif graph_result.query_type == QueryType.PRIORITY_RANKING:
            formatted += "详细优先级信息：\n"
            for i, item in enumerate(graph_result.structured_data[:10], 1):
                formatted += f"{i}. {item['cve_id']} ({item['severity']}, CVSS: {item['cvss_score']}) "
                formatted += f"影响{item['affected_hosts']}台主机\n"

        formatted += f"\n数据来源：{graph_result.source}"
        return formatted


# 全局实例
graph_retriever = None


def get_graph_retriever() -> GraphRetriever:
    """获取图检索器单例"""
    global graph_retriever
    if graph_retriever is None:
        graph_retriever = GraphRetriever()
    return graph_retriever


if __name__ == "__main__":
    # 测试代码
    retriever = get_graph_retriever()

    test_queries = [
        "CVE-2023-44487影响了哪些主机？",
        "web-server-01的安全风险如何？",
        "哪些漏洞需要优先修复？",
        "nginx镜像有什么问题吗？",
    ]

    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        result = retriever.retrieve(query)
        print(f"类型: {result.query_type.value}")
        print(f"置信度: {result.confidence}")
        print(f"结果: {result.summary}")
        print("-" * 50)
