"""
PostgreSQL向量存储模块

该模块负责PostgreSQL向量数据库的管理，使用pgvector扩展。
提供文档的向量化存储、相似度检索和索引管理功能。

数据流：
1. 文档输入 -> 向量化 -> 存储到PostgreSQL
2. 查询输入 -> 向量化 -> 相似度检索 -> 返回相关文档

使用PostgreSQL + pgvector作为向量存储后端。
"""

from typing import List, Dict, Any, Optional, Tuple
import uuid
import json
import numpy as np
from datetime import datetime

# PostgreSQL导入
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from langchain_core.documents import Document

from config import defaultConfig
from logger import get_logger, log_execution_time
import embeddings


class PostgreSQLVectorStoreManager:
    """PostgreSQL向量存储管理器
    
    专门管理PostgreSQL向量数据库，提供文档存储和检索功能。
    
    Attributes:
        engine: SQLAlchemy引擎
        embedding_manager: 嵌入管理器
        table_name: 文档表名
        vector_dimension: 向量维度
    """
    
    def __init__(self, embedding_manager: embeddings.EmbeddingManager):
        """初始化PostgreSQL向量存储管理器
        
        Args:
            embedding_manager: 嵌入管理器实例
        """
        self.logger = get_logger("PostgreSQLVectorStoreManager")
        self.embedding_manager = embedding_manager
        self.table_name = defaultConfig.postgresql.table_name
        self.vector_dimension = defaultConfig.postgresql.vector_dimension
        
        self._initialize_postgresql()
    
    def _initialize_postgresql(self) -> None:
        """初始化PostgreSQL连接和表结构"""
        try:
            self.logger.info("正在初始化PostgreSQL向量存储")
            
            # 创建数据库连接字符串
            db_url = (
                f"postgresql://{defaultConfig.postgresql.username}:"
                f"{defaultConfig.postgresql.password}@"
                f"{defaultConfig.postgresql.host}:"
                f"{defaultConfig.postgresql.port}/"
                f"{defaultConfig.postgresql.database}"
            )
            
            # 创建SQLAlchemy引擎
            self.engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=defaultConfig.postgresql.pool_size,
                max_overflow=defaultConfig.postgresql.max_overflow,
                pool_timeout=defaultConfig.postgresql.pool_timeout,
                echo=False
            )
            
            # 测试连接并初始化表结构
            self._create_tables()
            
            # 获取文档数量
            doc_count = self._get_document_count()
            
            self.logger.info(
                f"PostgreSQL向量存储初始化成功 | 表: {self.table_name} | "
                f"文档数量: {doc_count} | 向量维度: {self.vector_dimension}"
            )
            
        except Exception as e:
            self.logger.error(f"PostgreSQL向量存储初始化失败: {e}")
            raise
    
    def _create_tables(self) -> None:
        """创建必要的表结构"""
        try:
            with self.engine.connect() as conn:
                # 启用pgvector扩展
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                
                # 创建文档表
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id VARCHAR(255) PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({self.vector_dimension}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
                conn.execute(text(create_table_sql))
                
                # 创建向量索引以提高检索性能
                index_sql = f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
                conn.execute(text(index_sql))
                
                # 创建元数据索引
                metadata_index_sql = f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx 
                ON {self.table_name} USING gin (metadata)
                """
                conn.execute(text(metadata_index_sql))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"创建表结构失败: {e}")
            raise
    
    def _get_document_count(self) -> int:
        """获取文档数量"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {self.table_name}"))
                return result.scalar()
        except Exception as e:
            self.logger.error(f"获取文档数量失败: {e}")
            return 0
    
    @log_execution_time("postgresql_add_documents")
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到向量存储
        
        Args:
            documents: 要添加的文档列表
            
        Returns:
            添加的文档ID列表
        """
        if not documents:
            return []
        
        try:
            self.logger.info(f"开始添加 {len(documents)} 个文档到PostgreSQL")
            
            # 批量计算嵌入向量
            texts = [doc.page_content for doc in documents]
            embeddings_list = self.embedding_manager.embed_documents(texts)
            
            # 准备插入数据
            doc_ids = []
            insert_data = []
            
            for i, (doc, embedding) in enumerate(zip(documents, embeddings_list)):
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)
                
                # 转换向量为PostgreSQL格式
                vector_str = '[' + ','.join(map(str, embedding)) + ']'
                
                insert_data.append({
                    'id': doc_id,
                    'content': doc.page_content,
                    'metadata': json.dumps(doc.metadata) if doc.metadata else '{}',
                    'embedding': vector_str
                })
            
            # 批量插入
            with self.engine.connect() as conn:
                insert_sql = f"""
                INSERT INTO {self.table_name} (id, content, metadata, embedding)
                VALUES (%(id)s, %(content)s, %(metadata)s, %(embedding)s)
                """
                
                for data in insert_data:
                    conn.execute(text(insert_sql), data)
                
                conn.commit()
            
            self.logger.info(f"成功添加 {len(documents)} 个文档到PostgreSQL")
            return doc_ids
            
        except Exception as e:
            self.logger.error(f"添加文档到PostgreSQL失败: {e}")
            raise
    
    @log_execution_time("postgresql_similarity_search")
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: float = 0.0,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """相似度搜索
        
        Args:
            query: 查询字符串
            k: 返回结果数量
            score_threshold: 相似度阈值
            **kwargs: 额外参数
            
        Returns:
            (文档, 相似度分数)元组列表
        """
        try:
            self.logger.info(f"开始PostgreSQL相似度搜索: {query[:50]}...")
            
            # 计算查询向量
            query_embedding = self.embedding_manager.embed_query(query)
            query_vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # 执行相似度搜索
            with self.engine.connect() as conn:
                search_sql = f"""
                SELECT id, content, metadata, 
                       1 - (embedding <=> %(query_vector)s::vector) as similarity
                FROM {self.table_name}
                WHERE 1 - (embedding <=> %(query_vector)s::vector) >= %(threshold)s
                ORDER BY embedding <=> %(query_vector)s::vector
                LIMIT %(limit)s
                """
                
                result = conn.execute(text(search_sql), {
                    'query_vector': query_vector_str,
                    'threshold': score_threshold,
                    'limit': k
                })
                
                documents_with_scores = []
                for row in result:
                    metadata = json.loads(row.metadata) if row.metadata else {}
                    doc = Document(page_content=row.content, metadata=metadata)
                    documents_with_scores.append((doc, float(row.similarity)))
            
            self.logger.debug(f"PostgreSQL搜索完成，返回 {len(documents_with_scores)} 个结果")
            return documents_with_scores
            
        except Exception as e:
            self.logger.error(f"PostgreSQL相似度搜索失败: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            with self.engine.connect() as conn:
                # 获取文档数量
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {self.table_name}"))
                doc_count = count_result.scalar()
                
                # 获取表大小
                size_result = conn.execute(text(f"""
                SELECT pg_size_pretty(pg_total_relation_size('{self.table_name}')) as table_size
                """))
                table_size = size_result.scalar()
                
                return {
                    "backend": "postgresql",
                    "table_name": self.table_name,
                    "document_count": doc_count,
                    "table_size": table_size,
                    "vector_dimension": self.vector_dimension,
                    "connection_info": {
                        "host": defaultConfig.postgresql.host,
                        "port": defaultConfig.postgresql.port,
                        "database": defaultConfig.postgresql.database
                    }
                }
        except Exception as e:
            self.logger.error(f"获取PostgreSQL统计信息失败: {e}")
            return {"backend": "postgresql", "error": str(e)}
    
    def clear(self) -> None:
        """清空向量存储"""
        try:
            self.logger.info("正在清空PostgreSQL向量存储")
            
            with self.engine.connect() as conn:
                conn.execute(text(f"DELETE FROM {self.table_name}"))
                conn.commit()
            
            self.logger.info("PostgreSQL向量存储已清空")
            
        except Exception as e:
            self.logger.error(f"清空PostgreSQL向量存储失败: {e}")
            raise
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """删除指定文档
        
        Args:
            doc_ids: 要删除的文档ID列表
        """
        try:
            with self.engine.connect() as conn:
                delete_sql = f"DELETE FROM {self.table_name} WHERE id = ANY(%(doc_ids)s)"
                conn.execute(text(delete_sql), {"doc_ids": doc_ids})
                conn.commit()
            
            self.logger.info(f"成功删除 {len(doc_ids)} 个文档")
        except Exception as e:
            self.logger.error(f"删除文档失败: {e}")
            raise
