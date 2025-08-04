"""
Redis缓存管理器
负责会话管理、查询缓存和实时数据存储
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import ConnectionPool

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

logger = get_logger("redis_manager")

class RedisManager:
    """Redis缓存管理器"""
    
    def __init__(self):
        self.config = get_config()
        self.redis_url = self.config.get_redis_url()
        
        # 创建连接池
        self.pool = ConnectionPool.from_url(
            self.redis_url,
            max_connections=20,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={}
        )
        
        self.redis_client = None
        
        # 缓存键前缀
        self.SESSION_PREFIX = "session:"
        self.QUERY_CACHE_PREFIX = "query_cache:"
        self.CONTEXT_PREFIX = "context:"
        self.USER_PREFIX = "user:"
        self.METRICS_PREFIX = "metrics:"
        
    async def initialize(self):
        """初始化Redis连接"""
        try:
            self.redis_client = redis.Redis(connection_pool=self.pool)
            await self.redis_client.ping()
            logger.info("Redis连接初始化成功")
        except Exception as e:
            logger.error(f"Redis连接初始化失败: {str(e)}")
            raise
    
    async def close(self):
        """关闭Redis连接"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis连接已关闭")
    
    # 会话管理
    @log_performance()
    async def create_session(self, user_id: str, session_name: str = None, 
                           metadata: Dict[str, Any] = None) -> str:
        """创建新会话"""
        import uuid
        session_id = str(uuid.uuid4())
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "session_name": session_name or f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "message_count": 0,
            "metadata": metadata or {}
        }
        
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        await self.redis_client.hset(session_key, mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
            for k, v in session_data.items()
        })
        
        # 设置会话过期时间
        expire_hours = self.config.rag.session_timeout_hours
        await self.redis_client.expire(session_key, expire_hours * 3600)
        
        # 添加到用户会话列表
        user_sessions_key = f"{self.USER_PREFIX}{user_id}:sessions"
        await self.redis_client.sadd(user_sessions_key, session_id)
        
        logger.info(f"会话创建成功: {session_id}")
        return session_id
    
    @log_performance()
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        session_data = await self.redis_client.hgetall(session_key)
        
        if not session_data:
            return None
        
        # 解析JSON字段
        result = {}
        for k, v in session_data.items():
            k = k.decode() if isinstance(k, bytes) else k
            v = v.decode() if isinstance(v, bytes) else v
            
            if k in ["metadata"]:
                try:
                    result[k] = json.loads(v)
                except:
                    result[k] = v
            else:
                result[k] = v
        
        return result
    
    @log_performance()
    async def update_session_activity(self, session_id: str):
        """更新会话活动时间"""
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        await self.redis_client.hset(session_key, "last_activity", datetime.now().isoformat())
        
        # 重新设置过期时间
        expire_hours = self.config.rag.session_timeout_hours
        await self.redis_client.expire(session_key, expire_hours * 3600)
    
    # 对话上下文管理
    @log_performance()
    async def add_conversation_message(self, session_id: str, role: str, content: str, 
                                     metadata: Dict[str, Any] = None):
        """添加对话消息"""
        context_key = f"{self.CONTEXT_PREFIX}{session_id}"
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # 添加消息到列表
        await self.redis_client.lpush(context_key, json.dumps(message))
        
        # 限制上下文长度
        max_turns = self.config.rag.max_conversation_turns
        await self.redis_client.ltrim(context_key, 0, max_turns * 2 - 1)  # 每轮包含用户和助手消息
        
        # 更新会话消息计数
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        await self.redis_client.hincrby(session_key, "message_count", 1)
        
        # 更新活动时间
        await self.update_session_activity(session_id)
    
    @log_performance()
    async def get_conversation_context(self, session_id: str, 
                                     max_messages: int = None) -> List[Dict[str, Any]]:
        """获取对话上下文"""
        context_key = f"{self.CONTEXT_PREFIX}{session_id}"
        
        if max_messages is None:
            max_messages = self.config.rag.max_conversation_turns * 2
        
        messages = await self.redis_client.lrange(context_key, 0, max_messages - 1)
        
        result = []
        for msg in reversed(messages):  # 反转以获得正确的时间顺序
            try:
                message_data = json.loads(msg.decode() if isinstance(msg, bytes) else msg)
                result.append(message_data)
            except json.JSONDecodeError:
                logger.warning(f"无法解析消息: {msg}")
        
        return result
    
    # 查询缓存
    @log_performance()
    async def cache_query_result(self, query_hash: str, result: Any, 
                               expire_seconds: int = 3600):
        """缓存查询结果"""
        cache_key = f"{self.QUERY_CACHE_PREFIX}{query_hash}"
        
        # 使用pickle序列化复杂对象
        serialized_result = pickle.dumps(result)
        await self.redis_client.setex(cache_key, expire_seconds, serialized_result)
        
        logger.info(f"查询结果已缓存: {query_hash}")
    
    @log_performance()
    async def get_cached_query_result(self, query_hash: str) -> Optional[Any]:
        """获取缓存的查询结果"""
        cache_key = f"{self.QUERY_CACHE_PREFIX}{query_hash}"
        cached_data = await self.redis_client.get(cache_key)
        
        if cached_data:
            try:
                result = pickle.loads(cached_data)
                logger.info(f"命中查询缓存: {query_hash}")
                return result
            except Exception as e:
                logger.warning(f"缓存反序列化失败: {str(e)}")
                await self.redis_client.delete(cache_key)
        
        return None
    
    # 性能指标
    @log_performance()
    async def record_metric(self, metric_name: str, value: float, 
                          tags: Dict[str, str] = None):
        """记录性能指标"""
        timestamp = datetime.now().isoformat()
        metric_key = f"{self.METRICS_PREFIX}{metric_name}:{timestamp}"
        
        metric_data = {
            "value": value,
            "timestamp": timestamp,
            "tags": tags or {}
        }
        
        await self.redis_client.setex(metric_key, 86400, json.dumps(metric_data))  # 保存24小时
        
        # 添加到时间序列
        series_key = f"{self.METRICS_PREFIX}series:{metric_name}"
        await self.redis_client.zadd(series_key, {metric_key: datetime.now().timestamp()})
        
        # 清理旧数据（保留最近1000个点）
        await self.redis_client.zremrangebyrank(series_key, 0, -1001)
    
    @log_performance()
    async def get_metrics(self, metric_name: str, 
                         start_time: datetime = None, 
                         end_time: datetime = None) -> List[Dict[str, Any]]:
        """获取性能指标"""
        series_key = f"{self.METRICS_PREFIX}series:{metric_name}"
        
        # 设置时间范围
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.now()
        
        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()
        
        # 获取时间范围内的指标键
        metric_keys = await self.redis_client.zrangebyscore(series_key, start_ts, end_ts)
        
        results = []
        for key in metric_keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            metric_data = await self.redis_client.get(key_str)
            if metric_data:
                try:
                    data = json.loads(metric_data.decode() if isinstance(metric_data, bytes) else metric_data)
                    results.append(data)
                except json.JSONDecodeError:
                    continue
        
        return results
    
    # 实用工具
    async def clear_user_sessions(self, user_id: str):
        """清理用户的所有会话"""
        user_sessions_key = f"{self.USER_PREFIX}{user_id}:sessions"
        session_ids = await self.redis_client.smembers(user_sessions_key)
        
        for session_id in session_ids:
            session_id_str = session_id.decode() if isinstance(session_id, bytes) else session_id
            await self.delete_session(session_id_str)
        
        await self.redis_client.delete(user_sessions_key)
        logger.info(f"用户会话已清理: {user_id}")
    
    async def delete_session(self, session_id: str):
        """删除会话"""
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        context_key = f"{self.CONTEXT_PREFIX}{session_id}"
        
        await self.redis_client.delete(session_key, context_key)
        logger.info(f"会话已删除: {session_id}")
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis健康检查失败: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取Redis统计信息"""
        try:
            info = await self.redis_client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error(f"获取Redis统计信息失败: {str(e)}")
            return {}
