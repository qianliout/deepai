"""
Neo4j图数据库管理器
提供连接管理、CRUD操作和图查询功能
"""

import logging
from typing import List, Dict, Any, Optional, Union
from neo4j import GraphDatabase, Driver, Session, Transaction
from neo4j.exceptions import ServiceUnavailable, TransientError
import time
from contextlib import contextmanager

from config import defaultConfig

logger = logging.getLogger("RAG.Neo4j")


class Neo4jManager:
    """Neo4j图数据库管理器"""
    
    def __init__(self):
        """初始化Neo4j管理器"""
        self.config = defaultConfig.neo4j
        self.driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self):
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout,
                encrypted=self.config.encrypted,
                trust=self.config.trust
            )
            
            # 验证连接
            with self.driver.session(database=self.config.database) as session:
                session.run("RETURN 1")
            
            logger.info(f"✅ Neo4j连接成功: {self.config.uri}")
            
        except Exception as e:
            logger.error(f"❌ Neo4j连接失败: {e}")
            raise
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
    
    @contextmanager
    def get_session(self):
        """获取会话上下文管理器"""
        session = self.driver.session(database=self.config.database)
        try:
            yield session
        finally:
            session.close()
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行Cypher查询"""
        if parameters is None:
            parameters = {}
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                with self.get_session() as session:
                    result = session.run(query, parameters)
                    return [record.data() for record in result]
                    
            except (ServiceUnavailable, TransientError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"查询失败，重试中... (尝试 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"查询最终失败: {e}")
                    raise
            except Exception as e:
                logger.error(f"查询执行错误: {e}")
                raise
    
    def execute_write_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行写入查询（事务）"""
        if parameters is None:
            parameters = {}
        
        def _execute_write(tx: Transaction):
            result = tx.run(query, parameters)
            return [record.data() for record in result]
        
        try:
            with self.get_session() as session:
                return session.execute_write(_execute_write)
        except Exception as e:
            logger.error(f"写入查询执行错误: {e}")
            raise
    
    def create_host(self, host_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建主机节点"""
        query = """
        CREATE (h:Host {
            id: $id,
            hostname: $hostname,
            ip_address: $ip_address,
            os_type: $os_type,
            os_version: $os_version,
            cpu_cores: $cpu_cores,
            memory_gb: $memory_gb,
            disk_gb: $disk_gb,
            status: $status,
            location: $location,
            created_at: datetime($created_at),
            updated_at: datetime($updated_at)
        })
        RETURN h
        """
        result = self.execute_write_query(query, host_data)
        return result[0]['h'] if result else None
    
    def create_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建镜像节点"""
        query = """
        CREATE (i:Image {
            id: $id,
            image_name: $image_name,
            image_tag: $image_tag,
            registry: $registry,
            size_mb: $size_mb,
            architecture: $architecture,
            os: $os,
            created_at: datetime($created_at),
            updated_at: datetime($updated_at)
        })
        RETURN i
        """
        result = self.execute_write_query(query, image_data)
        return result[0]['i'] if result else None
    
    def create_vulnerability(self, vuln_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建漏洞节点"""
        query = """
        CREATE (v:Vulnerability {
            id: $id,
            cve_id: $cve_id,
            severity: $severity,
            cvss_score: $cvss_score,
            cvss_vector: $cvss_vector,
            description: $description,
            fix_suggestion: $fix_suggestion,
            published_date: date($published_date),
            source: $source,
            created_at: datetime($created_at),
            updated_at: datetime($updated_at)
        })
        RETURN v
        """
        result = self.execute_write_query(query, vuln_data)
        return result[0]['v'] if result else None
    
    def create_host_image_relationship(self, host_id: int, image_id: int, rel_data: Dict[str, Any]):
        """创建主机-镜像关系"""
        query = """
        MATCH (h:Host {id: $host_id}), (i:Image {id: $image_id})
        CREATE (h)-[r:HAS_IMAGE {
            container_name: $container_name,
            container_id: $container_id,
            status: $status,
            ports: $ports,
            volumes: $volumes,
            created_at: datetime($created_at)
        }]->(i)
        RETURN r
        """
        parameters = {
            'host_id': host_id,
            'image_id': image_id,
            **rel_data
        }
        return self.execute_write_query(query, parameters)
    
    def create_image_vulnerability_relationship(self, image_id: int, vuln_id: int, rel_data: Dict[str, Any]):
        """创建镜像-漏洞关系"""
        query = """
        MATCH (i:Image {id: $image_id}), (v:Vulnerability {id: $vuln_id})
        CREATE (i)-[r:HAS_VULNERABILITY {
            affected_package: $affected_package,
            package_version: $package_version,
            fixed_version: $fixed_version,
            layer_hash: $layer_hash,
            detected_at: datetime($detected_at)
        }]->(v)
        RETURN r
        """
        parameters = {
            'image_id': image_id,
            'vuln_id': vuln_id,
            **rel_data
        }
        return self.execute_write_query(query, parameters)
    
    def get_vulnerability_impact(self, cve_id: str) -> List[Dict[str, Any]]:
        """查询漏洞影响分析"""
        query = """
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
            i.image_name as image_name,
            i.image_tag as image_tag,
            i.registry as registry
        ORDER BY h.hostname, i.image_name
        """
        return self.execute_query(query, {'cve_id': cve_id})
    
    def get_host_risk_assessment(self, hostname: str) -> List[Dict[str, Any]]:
        """查询主机风险评估"""
        query = """
        MATCH (h:Host {hostname: $hostname})-[:HAS_IMAGE]->(i:Image)-[:HAS_VULNERABILITY]->(v:Vulnerability)
        RETURN 
            h.hostname as hostname,
            h.ip_address as ip_address,
            i.image_name as image_name,
            i.image_tag as image_tag,
            v.cve_id as cve_id,
            v.severity as severity,
            v.cvss_score as cvss_score,
            v.description as description,
            v.fix_suggestion as fix_suggestion
        ORDER BY v.cvss_score DESC, v.severity
        """
        return self.execute_query(query, {'hostname': hostname})
    
    def get_vulnerability_priority_ranking(self) -> List[Dict[str, Any]]:
        """获取漏洞修复优先级排序"""
        query = """
        MATCH (v:Vulnerability)<-[:HAS_VULNERABILITY]-(i:Image)<-[:HAS_IMAGE]-(h:Host)
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
        """
        return self.execute_query(query)
    
    def search_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """关键词搜索"""
        query = """
        MATCH (n)
        WHERE 
            (n:Host AND (n.hostname CONTAINS $keyword OR n.ip_address CONTAINS $keyword OR n.location CONTAINS $keyword))
            OR (n:Image AND (n.image_name CONTAINS $keyword OR n.image_tag CONTAINS $keyword))
            OR (n:Vulnerability AND (n.cve_id CONTAINS $keyword OR n.description CONTAINS $keyword))
        RETURN 
            labels(n) as node_type,
            n as node_data
        LIMIT 50
        """
        return self.execute_query(query, {'keyword': keyword})
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图数据库统计信息"""
        queries = {
            'host_count': "MATCH (h:Host) RETURN count(h) as count",
            'image_count': "MATCH (i:Image) RETURN count(i) as count",
            'vulnerability_count': "MATCH (v:Vulnerability) RETURN count(v) as count",
            'has_image_count': "MATCH ()-[r:HAS_IMAGE]->() RETURN count(r) as count",
            'has_vulnerability_count': "MATCH ()-[r:HAS_VULNERABILITY]->() RETURN count(r) as count"
        }
        
        stats = {}
        for key, query in queries.items():
            result = self.execute_query(query)
            stats[key] = result[0]['count'] if result else 0
        
        return stats
    
    def clear_all_data(self):
        """清空所有数据（谨慎使用）"""
        query = "MATCH (n) DETACH DELETE n"
        self.execute_write_query(query)
        logger.warning("⚠️ 所有图数据已清空")


# 全局实例
neo4j_manager = None

def get_neo4j_manager() -> Neo4jManager:
    """获取Neo4j管理器单例"""
    global neo4j_manager
    if neo4j_manager is None:
        neo4j_manager = Neo4jManager()
    return neo4j_manager


if __name__ == "__main__":
    # 测试代码
    manager = get_neo4j_manager()
    
    try:
        # 测试连接
        stats = manager.get_statistics()
        print("📊 图数据库统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 测试查询
        vulns = manager.get_vulnerability_priority_ranking()
        print(f"\n🔍 漏洞优先级排序 (前5个):")
        for vuln in vulns[:5]:
            print(f"  {vuln['cve_id']}: {vuln['severity']} (影响{vuln['affected_hosts']}台主机)")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    finally:
        manager.close()
