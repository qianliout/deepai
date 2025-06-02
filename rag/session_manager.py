"""
Redis会话管理模块

该模块负责管理RAG系统的对话历史，使用Redis作为存储后端。
支持会话的创建、保存、加载、清除等操作。

数据流：
1. 会话创建 -> Redis键生成 -> 初始化会话
2. 对话保存 -> 序列化消息 -> Redis存储 -> 设置过期时间
3. 历史加载 -> Redis查询 -> 反序列化 -> 返回历史记录

学习要点：
1. Redis作为会话存储的优势
2. 会话数据的序列化和反序列化
3. 会话过期和清理机制
"""

import json
import time
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from config import defaultConfig
from logger import get_logger

# 尝试导入Redis，如果失败则使用模拟版本
import redis
from redis.exceptions import ConnectionError, TimeoutError


@dataclass
class SessionMessage:
    """会话消息数据结构"""

    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: float
    message_id: str = None

    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMessage":
        """从字典创建实例"""
        return cls(**data)


@dataclass
class SessionInfo:
    """会话信息数据结构"""

    session_id: str
    created_at: float
    last_updated: float
    message_count: int
    title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionInfo":
        """从字典创建实例"""
        return cls(**data)


class RedisSessionManager:
    """Redis会话管理器

    使用Redis存储和管理RAG系统的对话历史。
    支持多会话管理、自动过期、数据持久化等功能。

    Attributes:
        redis_client: Redis客户端
        key_prefix: 键前缀
        session_expire: 会话过期时间
    """

    def __init__(self):
        """初始化Redis会话管理器"""
        self.logger = get_logger("RedisSessionManager")
        self.config = defaultConfig.redis
        self.key_prefix = self.config.key_prefix
        self.session_expire = self.config.session_expire

        # 初始化Redis连接
        self._init_redis_connection()

        self.logger.info(f"Redis会话管理器初始化完成 | 主机: {self.config.host}:{self.config.port}")

    def _init_redis_connection(self):
        """初始化Redis连接"""
        try:
            connection_params = {
                "host": self.config.host,
                "port": self.config.port,
                "db": self.config.db,
                "decode_responses": self.config.decode_responses,
                "socket_timeout": self.config.socket_timeout,
            }

            if self.config.password:
                connection_params["password"] = self.config.password

            self.redis_client = redis.Redis(**connection_params)

            # 测试连接
            self.redis_client.ping()
            self.logger.info("Redis连接测试成功")

        except Exception as e:
            self.logger.warning(f"Redis连接失败，使用模拟Redis客户端: {e}")
            raise  

    def create_session(self, title: str = "") -> str:
        """创建新会话

        Args:
            title: 会话标题

        Returns:
            会话ID
        """
        try:
            session_id = str(uuid.uuid4())
            current_time = time.time()

            # 创建会话信息
            session_info = SessionInfo(
                session_id=session_id,
                created_at=current_time,
                last_updated=current_time,
                message_count=0,
                title=title or f"会话_{session_id[:8]}",
            )

            # 保存会话信息
            info_key = f"{self.key_prefix}info:{session_id}"
            self.redis_client.setex(info_key, self.session_expire, json.dumps(session_info.to_dict(), ensure_ascii=False))

            # 初始化消息列表
            messages_key = f"{self.key_prefix}messages:{session_id}"
            self.redis_client.setex(messages_key, self.session_expire, json.dumps([], ensure_ascii=False))

            self.logger.info(f"创建新会话: {session_id} | 标题: {session_info.title}")
            return session_id

        except Exception as e:
            self.logger.error(f"创建会话失败: {e}")
            raise

    def save_message(self, session_id: str, role: str, content: str) -> bool:
        """保存消息到会话

        Args:
            session_id: 会话ID
            role: 消息角色
            content: 消息内容

        Returns:
            是否保存成功
        """
        try:
            # 创建消息对象
            message = SessionMessage(role=role, content=content, timestamp=time.time())

            # 获取现有消息
            messages = self.get_session_messages(session_id)
            messages.append(message)

            # 保存更新后的消息列表
            messages_key = f"{self.key_prefix}messages:{session_id}"
            messages_data = [msg.to_dict() for msg in messages]
            self.redis_client.setex(messages_key, self.session_expire, json.dumps(messages_data, ensure_ascii=False))

            # 更新会话信息
            self._update_session_info(session_id, len(messages))

            self.logger.debug(f"保存消息到会话 {session_id} | 角色: {role} | 长度: {len(content)}")
            return True

        except Exception as e:
            self.logger.error(f"保存消息失败: {e}")
            return False

    def get_session_messages(self, session_id: str) -> List[SessionMessage]:
        """获取会话消息历史

        Args:
            session_id: 会话ID

        Returns:
            消息列表
        """
        try:
            messages_key = f"{self.key_prefix}messages:{session_id}"
            messages_data = self.redis_client.get(messages_key)

            if not messages_data:
                return []

            messages_list = json.loads(messages_data)
            return [SessionMessage.from_dict(msg_data) for msg_data in messages_list]

        except Exception as e:
            self.logger.error(f"获取会话消息失败: {e}")
            return []

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """获取会话信息

        Args:
            session_id: 会话ID

        Returns:
            会话信息或None
        """
        try:
            info_key = f"{self.key_prefix}info:{session_id}"
            info_data = self.redis_client.get(info_key)

            if not info_data:
                return None

            info_dict = json.loads(info_data)
            return SessionInfo.from_dict(info_dict)

        except Exception as e:
            self.logger.error(f"获取会话信息失败: {e}")
            return None

    def _update_session_info(self, session_id: str, message_count: int):
        """更新会话信息

        Args:
            session_id: 会话ID
            message_count: 消息数量
        """
        try:
            session_info = self.get_session_info(session_id)
            if session_info:
                session_info.last_updated = time.time()
                session_info.message_count = message_count

                info_key = f"{self.key_prefix}info:{session_id}"
                self.redis_client.setex(info_key, self.session_expire, json.dumps(session_info.to_dict(), ensure_ascii=False))

        except Exception as e:
            self.logger.error(f"更新会话信息失败: {e}")

    def clear_session(self, session_id: str) -> bool:
        """清空会话历史

        Args:
            session_id: 会话ID

        Returns:
            是否清空成功
        """
        try:
            messages_key = f"{self.key_prefix}messages:{session_id}"
            self.redis_client.setex(messages_key, self.session_expire, json.dumps([], ensure_ascii=False))

            # 更新会话信息
            self._update_session_info(session_id, 0)

            self.logger.info(f"清空会话历史: {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"清空会话失败: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """删除会话

        Args:
            session_id: 会话ID

        Returns:
            是否删除成功
        """
        try:
            info_key = f"{self.key_prefix}info:{session_id}"
            messages_key = f"{self.key_prefix}messages:{session_id}"

            deleted_count = self.redis_client.delete(info_key, messages_key)

            self.logger.info(f"删除会话: {session_id} | 删除键数: {deleted_count}")
            return deleted_count > 0

        except Exception as e:
            self.logger.error(f"删除会话失败: {e}")
            return False

    def list_sessions(self) -> List[SessionInfo]:
        """列出所有会话

        Returns:
            会话信息列表
        """
        try:
            pattern = f"{self.key_prefix}info:*"
            session_keys = self.redis_client.keys(pattern)

            sessions = []
            for key in session_keys:
                info_data = self.redis_client.get(key)
                if info_data:
                    try:
                        info_dict = json.loads(info_data)
                        sessions.append(SessionInfo.from_dict(info_dict))
                    except json.JSONDecodeError:
                        continue

            # 按最后更新时间排序
            sessions.sort(key=lambda x: x.last_updated, reverse=True)

            return sessions

        except Exception as e:
            self.logger.error(f"列出会话失败: {e}")
            return []

    def extend_session_expire(self, session_id: str) -> bool:
        """延长会话过期时间

        Args:
            session_id: 会话ID

        Returns:
            是否延长成功
        """
        try:
            info_key = f"{self.key_prefix}info:{session_id}"
            messages_key = f"{self.key_prefix}messages:{session_id}"

            # 延长两个键的过期时间
            self.redis_client.expire(info_key, self.session_expire)
            self.redis_client.expire(messages_key, self.session_expire)

            return True

        except Exception as e:
            self.logger.error(f"延长会话过期时间失败: {e}")
            return False

    def get_connection_info(self) -> Dict[str, Any]:
        """获取Redis连接信息

        Returns:
            连接信息字典
        """
        try:
            info = self.redis_client.info()
            return {
                "connected": True,
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
            }
        except Exception as e:
            self.logger.error(f"获取Redis连接信息失败: {e}")
            return {"connected": False, "error": str(e)}
