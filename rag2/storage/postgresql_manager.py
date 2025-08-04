"""
PostgreSQL数据库管理器
负责PostgreSQL连接和向量数据操作
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncpg
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ..config.config import get_config
    from ..utils.logger import get_logger, log_performance, LogContext
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from config.config import get_config
    from utils.logger import get_logger, log_performance, LogContext

logger = get_logger("postgresql_manager")

class PostgreSQLManager:
    """PostgreSQL数据库管理器"""
    
    def __init__(self):
        self.config = get_config()
        self.db_url = self.config.get_postgres_url()
        self.async_db_url = self.db_url.replace("postgresql://", "postgresql+asyncpg://")
        
        # 同步引擎
        self.engine = create_engine(self.db_url, echo=self.config.debug)
        
        # 异步引擎
        self.async_engine = create_async_engine(self.async_db_url, echo=self.config.debug)
        
        # 会话工厂
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.AsyncSessionLocal = sessionmaker(
            self.async_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        self._connection_pool = None
        
    async def initialize(self):
        """初始化连接池"""
        try:
            self._connection_pool = await asyncpg.create_pool(
                self.config.get_postgres_url().replace("postgresql://", ""),
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info("PostgreSQL连接池初始化成功")
        except Exception as e:
            logger.error(f"PostgreSQL连接池初始化失败: {str(e)}")
            raise
    
    async def close(self):
        """关闭连接池"""
        if self._connection_pool:
            await self._connection_pool.close()
            logger.info("PostgreSQL连接池已关闭")
    
    @log_performance()
    async def insert_document(self, title: str, content: str, source: str = None, 
                            document_type: str = None, file_path: str = None, 
                            metadata: Dict[str, Any] = None) -> str:
        """插入文档"""
        document_id = str(uuid.uuid4())
        
        async with self._connection_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO documents (id, title, content, source, document_type, file_path, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, document_id, title, content, source, document_type, file_path, metadata or {})
        
        logger.info(f"文档插入成功: {document_id}")
        return document_id
    
    @log_performance()
    async def insert_document_chunk(self, document_id: str, chunk_index: int, 
                                  content: str, embedding: List[float], 
                                  token_count: int = None, 
                                  chunk_metadata: Dict[str, Any] = None) -> str:
        """插入文档分块"""
        chunk_id = str(uuid.uuid4())
        
        # 转换embedding为pgvector格式
        embedding_vector = f"[{','.join(map(str, embedding))}]"
        
        async with self._connection_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO document_chunks (id, document_id, chunk_index, content, 
                                           embedding, token_count, chunk_metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, chunk_id, document_id, chunk_index, content, embedding_vector, 
                token_count, chunk_metadata or {})
        
        logger.info(f"文档分块插入成功: {chunk_id}")
        return chunk_id
    
    @log_performance()
    async def similarity_search(self, query_embedding: List[float], 
                              top_k: int = 10, 
                              similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """向量相似性搜索"""
        query_vector = f"[{','.join(map(str, query_embedding))}]"
        
        async with self._connection_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    dc.id,
                    dc.content,
                    dc.chunk_metadata,
                    d.title,
                    d.source,
                    d.document_type,
                    1 - (dc.embedding <=> $1::vector) as similarity
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE 1 - (dc.embedding <=> $1::vector) > $2
                ORDER BY dc.embedding <=> $1::vector
                LIMIT $3
            """, query_vector, similarity_threshold, top_k)
        
        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "content": row["content"],
                "title": row["title"],
                "source": row["source"],
                "document_type": row["document_type"],
                "similarity": float(row["similarity"]),
                "metadata": row["chunk_metadata"]
            })
        
        logger.info(f"相似性搜索完成: 查询向量维度={len(query_embedding)}, 结果数={len(results)}")
        return results
    
    @log_performance()
    async def insert_host(self, hostname: str, ip_address: str, os: str = None,
                         cpu_cores: int = None, memory_gb: int = None,
                         environment: str = None, datacenter: str = None,
                         metadata: Dict[str, Any] = None) -> str:
        """插入主机信息"""
        host_id = str(uuid.uuid4())
        
        async with self._connection_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO hosts (id, hostname, ip_address, os, cpu_cores, 
                                 memory_gb, environment, datacenter, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, host_id, hostname, ip_address, os, cpu_cores, memory_gb, 
                environment, datacenter, metadata or {})
        
        logger.info(f"主机信息插入成功: {hostname} ({host_id})")
        return host_id
    
    @log_performance()
    async def insert_image(self, name: str, base_image: str = None, version: str = None,
                          size_mb: int = None, architecture: str = None,
                          registry: str = None, tags: List[str] = None,
                          metadata: Dict[str, Any] = None) -> str:
        """插入镜像信息"""
        image_id = str(uuid.uuid4())
        
        async with self._connection_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO images (id, name, base_image, version, size_mb, 
                                  architecture, registry, tags, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, image_id, name, base_image, version, size_mb, architecture, 
                registry, tags or [], metadata or {})
        
        logger.info(f"镜像信息插入成功: {name} ({image_id})")
        return image_id
    
    @log_performance()
    async def insert_vulnerability(self, cve_id: str, title: str = None, 
                                 description: str = None, severity: str = None,
                                 cvss_score: float = None, category: str = None,
                                 published_date: datetime = None,
                                 affected_packages: List[str] = None,
                                 fix_available: bool = False, fix_version: str = None,
                                 references: List[str] = None,
                                 metadata: Dict[str, Any] = None) -> str:
        """插入漏洞信息"""
        vuln_id = str(uuid.uuid4())
        
        async with self._connection_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO vulnerabilities (id, cve_id, title, description, severity,
                                           cvss_score, category, published_date,
                                           affected_packages, fix_available, fix_version,
                                           references, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """, vuln_id, cve_id, title, description, severity, cvss_score, 
                category, published_date, affected_packages or [], fix_available, 
                fix_version, references or [], metadata or {})
        
        logger.info(f"漏洞信息插入成功: {cve_id} ({vuln_id})")
        return vuln_id
    
    @log_performance()
    async def create_host_image_relationship(self, host_id: str, image_id: str,
                                           container_name: str = None,
                                           container_status: str = "running",
                                           properties: Dict[str, Any] = None) -> str:
        """创建主机-镜像关系"""
        rel_id = str(uuid.uuid4())
        
        async with self._connection_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO host_images (id, host_id, image_id, container_name,
                                       container_status, properties)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, rel_id, host_id, image_id, container_name, container_status, 
                properties or {})
        
        logger.info(f"主机-镜像关系创建成功: {rel_id}")
        return rel_id
    
    @log_performance()
    async def create_image_vulnerability_relationship(self, image_id: str, vulnerability_id: str,
                                                    scanner: str = None, confidence: str = "high",
                                                    properties: Dict[str, Any] = None) -> str:
        """创建镜像-漏洞关系"""
        rel_id = str(uuid.uuid4())
        
        async with self._connection_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO image_vulnerabilities (id, image_id, vulnerability_id,
                                                 scanner, confidence, properties)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, rel_id, image_id, vulnerability_id, scanner, confidence, 
                properties or {})
        
        logger.info(f"镜像-漏洞关系创建成功: {rel_id}")
        return rel_id
    
    @log_performance()
    async def get_host_vulnerabilities(self, hostname: str) -> List[Dict[str, Any]]:
        """获取主机的所有漏洞"""
        async with self._connection_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    h.hostname,
                    i.name as image_name,
                    v.cve_id,
                    v.severity,
                    v.cvss_score,
                    v.title,
                    v.description,
                    iv.scanner,
                    iv.confidence
                FROM hosts h
                JOIN host_images hi ON h.id = hi.host_id
                JOIN images i ON hi.image_id = i.id
                JOIN image_vulnerabilities iv ON i.id = iv.image_id
                JOIN vulnerabilities v ON iv.vulnerability_id = v.id
                WHERE h.hostname = $1
                ORDER BY v.cvss_score DESC
            """, hostname)
        
        results = []
        for row in rows:
            results.append({
                "hostname": row["hostname"],
                "image_name": row["image_name"],
                "cve_id": row["cve_id"],
                "severity": row["severity"],
                "cvss_score": float(row["cvss_score"]) if row["cvss_score"] else None,
                "title": row["title"],
                "description": row["description"],
                "scanner": row["scanner"],
                "confidence": row["confidence"]
            })
        
        logger.info(f"获取主机漏洞完成: {hostname}, 漏洞数={len(results)}")
        return results
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with self._connection_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL健康检查失败: {str(e)}")
            return False
