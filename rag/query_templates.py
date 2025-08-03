"""
AIOps知识图谱查询模板
提供预定义的Cypher查询模板，用于常见的AIOps场景
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass


class QueryTemplate(Enum):
    """查询模板枚举"""
    VULNERABILITY_IMPACT = "vulnerability_impact"
    HOST_RISK_ASSESSMENT = "host_risk_assessment"
    IMAGE_VULNERABILITIES = "image_vulnerabilities"
    CRITICAL_VULNERABILITIES = "critical_vulnerabilities"
    HOST_IMAGE_MAPPING = "host_image_mapping"
    VULNERABILITY_STATISTICS = "vulnerability_statistics"
    REPAIR_PRIORITY = "repair_priority"
    AFFECTED_HOSTS_BY_CVE = "affected_hosts_by_cve"
    SECURITY_DASHBOARD = "security_dashboard"
    COMPLIANCE_CHECK = "compliance_check"


@dataclass
class QueryResult:
    """查询结果包装器"""
    template: QueryTemplate
    data: List[Dict[str, Any]]
    summary: str
    metadata: Dict[str, Any]


class AIOpsQueryTemplates:
    """AIOps查询模板管理器"""
    
    def __init__(self):
        self.templates = {
            QueryTemplate.VULNERABILITY_IMPACT: {
                "query": """
                MATCH (v:Vulnerability {cve_id: $cve_id})<-[:HAS_VULNERABILITY]-(i:Image)<-[:HAS_IMAGE]-(h:Host)
                RETURN 
                    v.cve_id as cve_id,
                    v.severity as severity,
                    v.cvss_score as cvss_score,
                    v.description as description,
                    v.fix_suggestion as fix_suggestion,
                    h.hostname as hostname,
                    h.ip_address as ip_address,
                    h.location as location,
                    h.status as host_status,
                    i.image_name as image_name,
                    i.image_tag as image_tag,
                    i.registry as registry
                ORDER BY h.hostname, i.image_name
                """,
                "description": "分析特定CVE漏洞的影响范围",
                "parameters": ["cve_id"]
            },
            
            QueryTemplate.HOST_RISK_ASSESSMENT: {
                "query": """
                MATCH (h:Host {hostname: $hostname})-[:HAS_IMAGE]->(i:Image)-[:HAS_VULNERABILITY]->(v:Vulnerability)
                RETURN 
                    h.hostname as hostname,
                    h.ip_address as ip_address,
                    h.location as location,
                    h.status as host_status,
                    i.image_name as image_name,
                    i.image_tag as image_tag,
                    v.cve_id as cve_id,
                    v.severity as severity,
                    v.cvss_score as cvss_score,
                    v.description as description,
                    v.fix_suggestion as fix_suggestion
                ORDER BY v.cvss_score DESC, v.severity
                """,
                "description": "评估特定主机的安全风险",
                "parameters": ["hostname"]
            },
            
            QueryTemplate.IMAGE_VULNERABILITIES: {
                "query": """
                MATCH (i:Image)-[:HAS_VULNERABILITY]->(v:Vulnerability)
                WHERE i.image_name = $image_name AND i.image_tag = $image_tag
                RETURN 
                    i.image_name as image_name,
                    i.image_tag as image_tag,
                    i.registry as registry,
                    v.cve_id as cve_id,
                    v.severity as severity,
                    v.cvss_score as cvss_score,
                    v.description as description,
                    v.fix_suggestion as fix_suggestion
                ORDER BY v.cvss_score DESC
                """,
                "description": "查询特定镜像的所有漏洞",
                "parameters": ["image_name", "image_tag"]
            },
            
            QueryTemplate.CRITICAL_VULNERABILITIES: {
                "query": """
                MATCH (v:Vulnerability)<-[:HAS_VULNERABILITY]-(i:Image)<-[:HAS_IMAGE]-(h:Host)
                WHERE v.severity = 'CRITICAL' OR v.cvss_score >= $min_cvss_score
                WITH v, count(DISTINCT h) as affected_hosts, count(DISTINCT i) as affected_images
                RETURN 
                    v.cve_id as cve_id,
                    v.severity as severity,
                    v.cvss_score as cvss_score,
                    v.description as description,
                    v.fix_suggestion as fix_suggestion,
                    affected_hosts,
                    affected_images,
                    (v.cvss_score * affected_hosts * affected_images) as priority_score
                ORDER BY priority_score DESC, v.cvss_score DESC
                LIMIT $limit
                """,
                "description": "查询严重漏洞及其影响",
                "parameters": ["min_cvss_score", "limit"]
            },
            
            QueryTemplate.HOST_IMAGE_MAPPING: {
                "query": """
                MATCH (h:Host)-[r:HAS_IMAGE]->(i:Image)
                WHERE h.location = $location OR $location IS NULL
                RETURN 
                    h.hostname as hostname,
                    h.ip_address as ip_address,
                    h.location as location,
                    h.status as host_status,
                    i.image_name as image_name,
                    i.image_tag as image_tag,
                    i.registry as registry,
                    r.container_name as container_name,
                    r.status as container_status,
                    r.ports as ports
                ORDER BY h.hostname, i.image_name
                """,
                "description": "查询主机与镜像的映射关系",
                "parameters": ["location"]
            },
            
            QueryTemplate.VULNERABILITY_STATISTICS: {
                "query": """
                MATCH (v:Vulnerability)
                WITH v.severity as severity, count(v) as vuln_count
                RETURN severity, vuln_count
                ORDER BY 
                    CASE severity 
                        WHEN 'CRITICAL' THEN 1 
                        WHEN 'HIGH' THEN 2 
                        WHEN 'MEDIUM' THEN 3 
                        WHEN 'LOW' THEN 4 
                        ELSE 5 
                    END
                """,
                "description": "获取漏洞严重程度统计",
                "parameters": []
            },
            
            QueryTemplate.REPAIR_PRIORITY: {
                "query": """
                MATCH (v:Vulnerability)<-[:HAS_VULNERABILITY]-(i:Image)<-[:HAS_IMAGE]-(h:Host)
                WITH v, 
                     count(DISTINCT h) as affected_hosts, 
                     count(DISTINCT i) as affected_images,
                     collect(DISTINCT h.location) as locations
                RETURN 
                    v.cve_id as cve_id,
                    v.severity as severity,
                    v.cvss_score as cvss_score,
                    v.description as description,
                    v.fix_suggestion as fix_suggestion,
                    affected_hosts,
                    affected_images,
                    locations,
                    (v.cvss_score * affected_hosts * affected_images) as priority_score
                ORDER BY priority_score DESC, v.cvss_score DESC
                LIMIT $limit
                """,
                "description": "获取修复优先级排序",
                "parameters": ["limit"]
            },
            
            QueryTemplate.AFFECTED_HOSTS_BY_CVE: {
                "query": """
                MATCH (v:Vulnerability {cve_id: $cve_id})<-[:HAS_VULNERABILITY]-(i:Image)<-[:HAS_IMAGE]-(h:Host)
                RETURN DISTINCT
                    h.hostname as hostname,
                    h.ip_address as ip_address,
                    h.location as location,
                    h.status as host_status,
                    collect(DISTINCT i.image_name + ':' + i.image_tag) as affected_images
                ORDER BY h.hostname
                """,
                "description": "查询受特定CVE影响的所有主机",
                "parameters": ["cve_id"]
            },
            
            QueryTemplate.SECURITY_DASHBOARD: {
                "query": """
                MATCH (h:Host)
                OPTIONAL MATCH (h)-[:HAS_IMAGE]->(i:Image)
                OPTIONAL MATCH (i)-[:HAS_VULNERABILITY]->(v:Vulnerability)
                WITH h, 
                     count(DISTINCT i) as image_count,
                     count(DISTINCT v) as vuln_count,
                     count(DISTINCT CASE WHEN v.severity = 'CRITICAL' THEN v END) as critical_count,
                     count(DISTINCT CASE WHEN v.severity = 'HIGH' THEN v END) as high_count
                RETURN 
                    h.hostname as hostname,
                    h.ip_address as ip_address,
                    h.location as location,
                    h.status as host_status,
                    image_count,
                    vuln_count,
                    critical_count,
                    high_count,
                    CASE 
                        WHEN critical_count > 0 THEN 'CRITICAL'
                        WHEN high_count > 0 THEN 'HIGH'
                        WHEN vuln_count > 0 THEN 'MEDIUM'
                        ELSE 'LOW'
                    END as risk_level
                ORDER BY critical_count DESC, high_count DESC, vuln_count DESC
                """,
                "description": "生成安全仪表板数据",
                "parameters": []
            },
            
            QueryTemplate.COMPLIANCE_CHECK: {
                "query": """
                MATCH (h:Host)-[:HAS_IMAGE]->(i:Image)
                OPTIONAL MATCH (i)-[:HAS_VULNERABILITY]->(v:Vulnerability)
                WHERE v.severity IN ['CRITICAL', 'HIGH'] OR v.cvss_score >= $compliance_threshold
                WITH h, i, count(v) as high_risk_vulns
                RETURN 
                    h.hostname as hostname,
                    h.ip_address as ip_address,
                    h.location as location,
                    i.image_name as image_name,
                    i.image_tag as image_tag,
                    high_risk_vulns,
                    CASE 
                        WHEN high_risk_vulns = 0 THEN 'COMPLIANT'
                        WHEN high_risk_vulns <= $warning_threshold THEN 'WARNING'
                        ELSE 'NON_COMPLIANT'
                    END as compliance_status
                ORDER BY high_risk_vulns DESC, h.hostname
                """,
                "description": "检查合规性状态",
                "parameters": ["compliance_threshold", "warning_threshold"]
            }
        }
    
    def get_template(self, template: QueryTemplate) -> Dict[str, Any]:
        """获取查询模板"""
        return self.templates.get(template, {})
    
    def get_query(self, template: QueryTemplate) -> str:
        """获取查询语句"""
        template_data = self.get_template(template)
        return template_data.get("query", "").strip()
    
    def get_parameters(self, template: QueryTemplate) -> List[str]:
        """获取模板参数列表"""
        template_data = self.get_template(template)
        return template_data.get("parameters", [])
    
    def get_description(self, template: QueryTemplate) -> str:
        """获取模板描述"""
        template_data = self.get_template(template)
        return template_data.get("description", "")
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """列出所有可用模板"""
        result = []
        for template_enum in QueryTemplate:
            template_data = self.get_template(template_enum)
            result.append({
                "name": template_enum.value,
                "description": template_data.get("description", ""),
                "parameters": template_data.get("parameters", [])
            })
        return result
    
    def validate_parameters(self, template: QueryTemplate, parameters: Dict[str, Any]) -> bool:
        """验证参数是否完整"""
        required_params = self.get_parameters(template)
        return all(param in parameters for param in required_params)


# 全局实例
query_templates = AIOpsQueryTemplates()


if __name__ == "__main__":
    # 测试代码
    templates = query_templates
    
    print("📋 可用查询模板:")
    for template_info in templates.list_templates():
        print(f"• {template_info['name']}: {template_info['description']}")
        if template_info['parameters']:
            print(f"  参数: {', '.join(template_info['parameters'])}")
        print()
    
    # 测试特定模板
    print("🔍 漏洞影响分析模板:")
    query = templates.get_query(QueryTemplate.VULNERABILITY_IMPACT)
    print(query)
