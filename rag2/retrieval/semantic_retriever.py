"""
语义检索器
基于向量相似度的语义检索
"""

import time
from typing import List, Dict, Any, Optional
import asyncio

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .base_retriever import VectorRetriever
    from ..storage.postgresql_manager import PostgreSQLManager
    from ..utils.logger import get_logger, log_performance, log_retrieval
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from retrieval.base_retriever import VectorRetriever
    from storage.postgresql_manager import PostgreSQLManager
    from utils.logger import get_logger, log_performance, log_retrieval

logger = get_logger("semantic_retriever")

class SemanticRetriever(VectorRetriever):
    """语义检索器"""
    
    def __init__(self, name: str = "semantic"):
        super().__init__(name)
        self.pg_manager = None
    
    async def _get_pg_manager(self) -> PostgreSQLManager:
        """获取PostgreSQL管理器"""
        if self.pg_manager is None:
            self.pg_manager = PostgreSQLManager()
            await self.pg_manager.initialize()
        return self.pg_manager
    
    @log_performance()
    async def retrieve(self, query: str, top_k: int = 10, 
                      similarity_threshold: float = None, **kwargs) -> List[Dict[str, Any]]:
        """语义检索"""
        start_time = time.time()
        
        try:
            # 1. 编码查询
            query_embedding = await self.encode_query(query)
            
            # 2. 设置相似度阈值
            if similarity_threshold is None:
                similarity_threshold = self.config.rag.similarity_threshold
            
            # 3. 向量相似度搜索
            pg_manager = await self._get_pg_manager()
            results = await pg_manager.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            # 4. 格式化结果
            formatted_results = []
            for result in results:
                formatted_result = {
                    "id": result["id"],
                    "content": result["content"],
                    "title": result["title"],
                    "source": result["source"],
                    "document_type": result["document_type"],
                    "similarity": result["similarity"],
                    "metadata": result.get("metadata", {}),
                    "retriever": self.name,
                    "method": "vector_similarity"
                }
                formatted_results.append(formatted_result)
            
            # 5. 记录检索日志
            end_time = time.time()
            duration = end_time - start_time
            
            log_retrieval(
                query=query,
                retrieved_count=len(formatted_results),
                method="semantic",
                duration=duration
            )
            
            logger.info(f"语义检索完成: 查询='{query}', 结果数={len(formatted_results)}, 耗时={duration:.4f}秒")
            return formatted_results
            
        except Exception as e:
            logger.error(f"语义检索失败: {str(e)}")
            return []
    
    @log_performance()
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """添加文档到向量索引"""
        try:
            pg_manager = await self._get_pg_manager()
            embedding_manager = await self._get_embedding_manager()
            
            success_count = 0
            
            for doc_data in documents:
                document = doc_data.get("document", {})
                chunks = doc_data.get("chunks", [])
                
                # 1. 插入文档
                document_id = await pg_manager.insert_document(
                    title=document.get("title", ""),
                    content=document.get("content", ""),
                    source=document.get("source", ""),
                    document_type=document.get("document_type", ""),
                    file_path=document.get("metadata", {}).get("file_path"),
                    metadata=document.get("metadata", {})
                )
                
                # 2. 插入文档分块
                for chunk in chunks:
                    # 如果分块没有嵌入，生成嵌入
                    if "embedding" not in chunk:
                        embedding = await embedding_manager.encode_text_async(chunk["content"])
                        chunk["embedding"] = embedding
                    
                    await pg_manager.insert_document_chunk(
                        document_id=document_id,
                        chunk_index=chunk["chunk_index"],
                        content=chunk["content"],
                        embedding=chunk["embedding"],
                        token_count=chunk.get("token_count", 0),
                        chunk_metadata=chunk.get("chunk_metadata", {})
                    )
                
                success_count += 1
                logger.info(f"文档添加成功: {document.get('title', 'Unknown')} ({len(chunks)} 个分块)")
            
            logger.info(f"批量文档添加完成: {success_count}/{len(documents)} 成功")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"文档添加失败: {str(e)}")
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """删除文档"""
        try:
            pg_manager = await self._get_pg_manager()
            
            # PostgreSQL的外键约束会自动删除相关的分块
            async with pg_manager._connection_pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM documents WHERE id = $1
                """, document_id)
            
            logger.info(f"文档删除成功: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"文档删除失败: {document_id}, 错误: {str(e)}")
            return False
    
    async def search_by_metadata(self, metadata_filters: Dict[str, Any], 
                                top_k: int = 10) -> List[Dict[str, Any]]:
        """基于元数据搜索"""
        try:
            pg_manager = await self._get_pg_manager()
            
            # 构建查询条件
            where_conditions = []
            params = []
            param_index = 1
            
            for key, value in metadata_filters.items():
                where_conditions.append(f"d.metadata->>'{key}' = ${param_index}")
                params.append(str(value))
                param_index += 1
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "TRUE"
            
            async with pg_manager._connection_pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT 
                        dc.id,
                        dc.content,
                        dc.chunk_metadata,
                        d.title,
                        d.source,
                        d.document_type,
                        d.metadata
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE {where_clause}
                    ORDER BY d.created_at DESC
                    LIMIT ${param_index}
                """, *params, top_k)
            
            results = []
            for row in rows:
                result = {
                    "id": row["id"],
                    "content": row["content"],
                    "title": row["title"],
                    "source": row["source"],
                    "document_type": row["document_type"],
                    "metadata": row["metadata"],
                    "chunk_metadata": row["chunk_metadata"],
                    "retriever": self.name,
                    "method": "metadata_filter"
                }
                results.append(result)
            
            logger.info(f"元数据搜索完成: 过滤器={metadata_filters}, 结果数={len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"元数据搜索失败: {str(e)}")
            return []
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """获取文档统计信息"""
        try:
            pg_manager = await self._get_pg_manager()
            
            async with pg_manager._connection_pool.acquire() as conn:
                # 文档统计
                doc_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_documents,
                        COUNT(DISTINCT document_type) as document_types,
                        AVG(LENGTH(content)) as avg_content_length
                    FROM documents
                """)
                
                # 分块统计
                chunk_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        AVG(token_count) as avg_token_count,
                        AVG(LENGTH(content)) as avg_chunk_length
                    FROM document_chunks
                """)
                
                # 按类型统计
                type_stats = await conn.fetch("""
                    SELECT 
                        document_type,
                        COUNT(*) as count
                    FROM documents
                    GROUP BY document_type
                    ORDER BY count DESC
                """)
            
            stats = {
                "total_documents": doc_stats["total_documents"],
                "total_chunks": chunk_stats["total_chunks"],
                "document_types": doc_stats["document_types"],
                "avg_content_length": float(doc_stats["avg_content_length"] or 0),
                "avg_token_count": float(chunk_stats["avg_token_count"] or 0),
                "avg_chunk_length": float(chunk_stats["avg_chunk_length"] or 0),
                "type_distribution": [
                    {"type": row["document_type"], "count": row["count"]} 
                    for row in type_stats
                ]
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取文档统计失败: {str(e)}")
            return {}
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            pg_manager = await self._get_pg_manager()
            embedding_manager = await self._get_embedding_manager()
            
            # 检查数据库连接
            db_healthy = await pg_manager.health_check()
            
            # 检查嵌入模型
            model_healthy = embedding_manager.health_check()
            
            return db_healthy and model_healthy
            
        except Exception as e:
            logger.error(f"语义检索器健康检查失败: {str(e)}")
            return False

# 便捷函数
async def create_semantic_retriever() -> SemanticRetriever:
    """创建语义检索器实例"""
    retriever = SemanticRetriever()
    return retriever
