"""
MySQL数据库管理器
负责对话历史、会话管理和系统配置存储
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import aiomysql
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

logger = get_logger("mysql_manager")

class MySQLManager:
    """MySQL数据库管理器"""
    
    def __init__(self):
        self.config = get_config()
        self.db_url = self.config.get_mysql_url()
        self.async_db_url = self.db_url.replace("mysql+pymysql://", "mysql+aiomysql://")
        
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
            self._connection_pool = await aiomysql.create_pool(
                host=self.config.database.mysql_host,
                port=self.config.database.mysql_port,
                user=self.config.database.mysql_user,
                password=self.config.database.mysql_password,
                db=self.config.database.mysql_db,
                minsize=5,
                maxsize=20,
                autocommit=False
            )
            logger.info("MySQL连接池初始化成功")
        except Exception as e:
            logger.error(f"MySQL连接池初始化失败: {str(e)}")
            raise
    
    async def close(self):
        """关闭连接池"""
        if self._connection_pool:
            self._connection_pool.close()
            await self._connection_pool.wait_closed()
            logger.info("MySQL连接池已关闭")
    
    # 用户管理
    @log_performance()
    async def create_user(self, username: str, email: str = None, 
                         metadata: Dict[str, Any] = None) -> str:
        """创建用户"""
        user_id = str(uuid.uuid4())
        
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO users (id, username, email, metadata)
                    VALUES (%s, %s, %s, %s)
                """, (user_id, username, email, metadata))
                await conn.commit()
        
        logger.info(f"用户创建成功: {username} ({user_id})")
        return user_id
    
    @log_performance()
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT * FROM users WHERE id = %s
                """, (user_id,))
                result = await cursor.fetchone()
        
        return result
    
    # 会话管理
    @log_performance()
    async def create_session(self, user_id: str, session_name: str = None,
                           metadata: Dict[str, Any] = None) -> str:
        """创建会话"""
        session_id = str(uuid.uuid4())
        
        if session_name is None:
            session_name = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO sessions (id, user_id, session_name, metadata)
                    VALUES (%s, %s, %s, %s)
                """, (session_id, user_id, session_name, metadata))
                await conn.commit()
        
        logger.info(f"会话创建成功: {session_name} ({session_id})")
        return session_id
    
    @log_performance()
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT * FROM sessions WHERE id = %s
                """, (session_id,))
                result = await cursor.fetchone()
        
        return result
    
    @log_performance()
    async def update_session_activity(self, session_id: str):
        """更新会话活动时间"""
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    UPDATE sessions SET last_activity = NOW() WHERE id = %s
                """, (session_id,))
                await conn.commit()
    
    @log_performance()
    async def get_user_sessions(self, user_id: str, 
                              status: str = "active") -> List[Dict[str, Any]]:
        """获取用户的会话列表"""
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT * FROM sessions 
                    WHERE user_id = %s AND status = %s
                    ORDER BY last_activity DESC
                """, (user_id, status))
                results = await cursor.fetchall()
        
        return results or []
    
    # 对话管理
    @log_performance()
    async def add_conversation(self, session_id: str, message_type: str, 
                             content: str, role: str = None, 
                             token_count: int = 0, 
                             metadata: Dict[str, Any] = None) -> str:
        """添加对话记录"""
        conversation_id = str(uuid.uuid4())
        
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO conversations (id, session_id, message_type, content, 
                                             role, token_count, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (conversation_id, session_id, message_type, content, 
                      role, token_count, metadata))
                await conn.commit()
        
        # 更新会话活动时间
        await self.update_session_activity(session_id)
        
        logger.info(f"对话记录添加成功: {conversation_id}")
        return conversation_id
    
    @log_performance()
    async def get_conversation_history(self, session_id: str, 
                                     limit: int = 50) -> List[Dict[str, Any]]:
        """获取对话历史"""
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT * FROM conversations 
                    WHERE session_id = %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (session_id, limit))
                results = await cursor.fetchall()
        
        # 反转顺序以获得正确的时间顺序
        return list(reversed(results or []))
    
    # 查询日志
    @log_performance()
    async def log_query(self, session_id: str, user_query: str, 
                       processed_query: str = None, query_type: str = None,
                       intent: str = None, entities: Dict[str, Any] = None,
                       processing_time_ms: int = None, success: bool = True,
                       error_message: str = None, 
                       metadata: Dict[str, Any] = None) -> str:
        """记录查询日志"""
        query_log_id = str(uuid.uuid4())
        
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO query_logs (id, session_id, user_query, processed_query,
                                          query_type, intent, entities, processing_time_ms,
                                          success, error_message, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (query_log_id, session_id, user_query, processed_query,
                      query_type, intent, entities, processing_time_ms,
                      success, error_message, metadata))
                await conn.commit()
        
        logger.info(f"查询日志记录成功: {query_log_id}")
        return query_log_id
    
    # 检索日志
    @log_performance()
    async def log_retrieval(self, query_log_id: str, retrieval_method: str,
                          query_vector_id: str = None, 
                          retrieved_documents: List[Dict[str, Any]] = None,
                          similarity_scores: List[float] = None,
                          rerank_scores: List[float] = None,
                          final_context: str = None, retrieval_time_ms: int = None,
                          document_count: int = None, 
                          metadata: Dict[str, Any] = None) -> str:
        """记录检索日志"""
        retrieval_log_id = str(uuid.uuid4())
        
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO retrieval_logs (id, query_log_id, retrieval_method,
                                              query_vector_id, retrieved_documents,
                                              similarity_scores, rerank_scores,
                                              final_context, retrieval_time_ms,
                                              document_count, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (retrieval_log_id, query_log_id, retrieval_method,
                      query_vector_id, retrieved_documents, similarity_scores,
                      rerank_scores, final_context, retrieval_time_ms,
                      document_count, metadata))
                await conn.commit()
        
        logger.info(f"检索日志记录成功: {retrieval_log_id}")
        return retrieval_log_id
    
    # 反馈管理
    @log_performance()
    async def add_feedback(self, session_id: str = None, query_log_id: str = None,
                         conversation_id: str = None, feedback_type: str = "rating",
                         rating: int = None, comment: str = None,
                         metadata: Dict[str, Any] = None) -> str:
        """添加用户反馈"""
        feedback_id = str(uuid.uuid4())
        
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO feedback (id, session_id, query_log_id, conversation_id,
                                        feedback_type, rating, comment, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (feedback_id, session_id, query_log_id, conversation_id,
                      feedback_type, rating, comment, metadata))
                await conn.commit()
        
        logger.info(f"用户反馈添加成功: {feedback_id}")
        return feedback_id
    
    # 系统指标
    @log_performance()
    async def record_system_metric(self, metric_name: str, metric_value: float,
                                  metric_unit: str = None, component: str = None,
                                  metadata: Dict[str, Any] = None) -> str:
        """记录系统指标"""
        metric_id = str(uuid.uuid4())
        
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO system_metrics (id, metric_name, metric_value,
                                              metric_unit, component, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (metric_id, metric_name, metric_value, metric_unit, 
                      component, metadata))
                await conn.commit()
        
        return metric_id
    
    # 配置管理
    @log_performance()
    async def get_configuration(self, config_key: str) -> Optional[Dict[str, Any]]:
        """获取配置项"""
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT * FROM configurations WHERE config_key = %s
                """, (config_key,))
                result = await cursor.fetchone()
        
        return result
    
    @log_performance()
    async def set_configuration(self, config_key: str, config_value: str,
                              config_type: str = "string", description: str = None,
                              metadata: Dict[str, Any] = None) -> str:
        """设置配置项"""
        config_id = str(uuid.uuid4())
        
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO configurations (id, config_key, config_value,
                                              config_type, description, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    config_value = VALUES(config_value),
                    config_type = VALUES(config_type),
                    description = VALUES(description),
                    metadata = VALUES(metadata),
                    updated_at = NOW()
                """, (config_id, config_key, config_value, config_type, 
                      description, metadata))
                await conn.commit()
        
        logger.info(f"配置项设置成功: {config_key}")
        return config_id
    
    # 统计查询
    @log_performance()
    async def get_query_statistics(self, start_date: datetime = None,
                                 end_date: datetime = None) -> Dict[str, Any]:
        """获取查询统计"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()
        
        async with self._connection_pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT 
                        COUNT(*) as total_queries,
                        AVG(processing_time_ms) as avg_processing_time,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                        SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as error_count,
                        query_type,
                        DATE(timestamp) as query_date
                    FROM query_logs 
                    WHERE timestamp BETWEEN %s AND %s
                    GROUP BY query_type, DATE(timestamp)
                    ORDER BY query_date DESC
                """, (start_date, end_date))
                results = await cursor.fetchall()
        
        return {"statistics": results or []}
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with self._connection_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    await cursor.fetchone()
            return True
        except Exception as e:
            logger.error(f"MySQL健康检查失败: {str(e)}")
            return False
