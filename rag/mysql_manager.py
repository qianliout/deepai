"""
MySQL对话存储管理模块

该模块负责将对话数据持久化存储到MySQL数据库中。
支持对话记录的创建、查询、更新和统计功能。

数据流：
1. 对话数据 -> 数据验证 -> MySQL存储
2. 查询请求 -> SQL查询 -> 结果返回
3. 统计分析 -> 聚合查询 -> 统计结果

学习要点：
1. MySQL数据库的连接和操作
2. SQLAlchemy ORM的使用
3. 对话数据的表结构设计
4. 数据库连接池管理
"""

import json
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from contextlib import contextmanager

from config import defaultConfig
from logger import get_logger

# 尝试导入MySQL相关库，如果失败则使用模拟版本
import pymysql
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError


# 数据库表定义
Base = declarative_base()


class ConversationRecord(Base):
    """对话记录表"""

    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    user_id = Column(String(36), nullable=True, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    meta_data = Column("metadata", JSON, nullable=True)  # 避免SQLAlchemy保留字冲突
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    token_count = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "role": self.role,
            "content": self.content,
            "metadata": self.meta_data,  # 保持外部接口不变
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "token_count": self.token_count,
            "processing_time": self.processing_time,
        }


class SessionRecord(Base):
    """会话记录表"""

    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=True, index=True)
    title = Column(String(255), nullable=True)
    status = Column(String(20), default="active", nullable=False)  # active, archived, deleted
    meta_data = Column("metadata", JSON, nullable=True)  # 避免SQLAlchemy保留字冲突
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    message_count = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "status": self.status,
            "metadata": self.meta_data,  # 保持外部接口不变
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ConversationData:
    """对话数据结构"""

    session_id: str
    role: str
    content: str
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    token_count: Optional[int] = None
    processing_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


class MockMySQLManager:
    """模拟MySQL管理器"""

    def __init__(self):
        """初始化模拟MySQL管理器"""
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.connected = True

    def save_conversation(self, conversation: ConversationData) -> bool:
        """保存对话记录"""
        conv_id = str(uuid.uuid4())
        self.conversations[conv_id] = {
            "id": conv_id,
            "session_id": conversation.session_id,
            "user_id": conversation.user_id,
            "role": conversation.role,
            "content": conversation.content,
            "metadata": conversation.metadata,  # 保持接口一致
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "token_count": conversation.token_count,
            "processing_time": conversation.processing_time,
        }
        return True

    def get_session_conversations(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取会话对话记录"""
        results = []
        for conv in self.conversations.values():
            if conv["session_id"] == session_id:
                results.append(conv)
        return sorted(results, key=lambda x: x["created_at"])[:limit]

    def create_session(self, session_id: str, user_id: Optional[str] = None, title: Optional[str] = None) -> bool:
        """创建会话记录"""
        self.sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "title": title,
            "status": "active",
            "metadata": {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "message_count": 0,
            "total_tokens": 0,
        }
        return True

    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        return {"connected": True, "type": "mock", "conversations_count": len(self.conversations), "sessions_count": len(self.sessions)}


class MySQLManager:
    """MySQL对话存储管理器

    负责对话数据的持久化存储和查询操作。
    使用SQLAlchemy ORM进行数据库操作。

    Attributes:
        engine: SQLAlchemy引擎
        SessionLocal: 会话工厂
        logger: 日志记录器
    """

    def __init__(self):
        """初始化MySQL管理器"""
        self.logger = get_logger("MySQLManager")
        self.config = defaultConfig.mysql

        # 初始化数据库连接
        self._init_database_connection()

        # 创建表结构
        self._create_tables()

        self.logger.info(f"MySQL管理器初始化完成 | 数据库: {self.config.database}")

    def _init_database_connection(self):
        """初始化数据库连接"""
        try:
            # 构建连接URL
            connection_url = (
                f"mysql+pymysql://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port}/{self.config.database}"
                f"?charset={self.config.charset}"
            )

            # 创建引擎
            self.engine = create_engine(
                connection_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_pre_ping=True,
                echo=False,
            )

            # 创建会话工厂
            self.SessionLocal = sessionmaker(bind=self.engine)

            # 测试连接
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")

            self.logger.info("MySQL连接成功")

        except Exception as e:
            self.logger.warning(f"MySQL连接失败，使用模拟MySQL客户端: {e}")
            self.mock_manager = MockMySQLManager()
            self.logger.info("已启用模拟MySQL客户端")

    def _create_tables(self):
        """创建数据库表"""
        try:
            if hasattr(self, "engine"):
                Base.metadata.create_all(bind=self.engine)
                self.logger.info("数据库表创建/检查完成")
        except Exception as e:
            self.logger.error(f"创建数据库表失败: {e}")
            raise

    @contextmanager
    def get_session(self):
        """获取数据库会话上下文管理器"""
        if hasattr(self, "SessionLocal"):
            session = self.SessionLocal()
            try:
                yield session
                session.commit()
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
        else:
            # 模拟会话
            yield None

    def save_conversation(self, conversation: ConversationData) -> bool:
        """保存对话记录

        Args:
            conversation: 对话数据

        Returns:
            是否保存成功
        """
        try:
            if hasattr(self, "mock_manager"):
                return self.mock_manager.save_conversation(conversation)

            with self.get_session() as session:
                # 创建对话记录
                conv_record = ConversationRecord(
                    id=str(uuid.uuid4()),
                    session_id=conversation.session_id,
                    user_id=conversation.user_id,
                    role=conversation.role,
                    content=conversation.content,
                    meta_data=conversation.metadata,  # 字段名变更
                    token_count=conversation.token_count,
                    processing_time=conversation.processing_time,
                )

                session.add(conv_record)

                # 更新会话统计
                self._update_session_stats(session, conversation.session_id, conversation.token_count or 0)

                self.logger.debug(f"对话记录保存成功: {conversation.session_id}")
                return True

        except Exception as e:
            self.logger.error(f"保存对话记录失败: {e}")
            return False

    def _update_session_stats(self, session, session_id: str, token_count: int):
        """更新会话统计信息"""
        try:
            session_record = session.query(SessionRecord).filter_by(id=session_id).first()
            if session_record:
                session_record.message_count += 1
                session_record.total_tokens += token_count
                session_record.updated_at = datetime.utcnow()
            else:
                # 创建新的会话记录
                session_record = SessionRecord(id=session_id, message_count=1, total_tokens=token_count)
                session.add(session_record)

        except Exception as e:
            self.logger.error(f"更新会话统计失败: {e}")

    def get_session_conversations(self, session_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """获取会话对话记录

        Args:
            session_id: 会话ID
            limit: 返回记录数限制
            offset: 偏移量

        Returns:
            对话记录列表
        """
        try:
            if hasattr(self, "mock_manager"):
                return self.mock_manager.get_session_conversations(session_id, limit)

            with self.get_session() as session:
                conversations = (
                    session.query(ConversationRecord)
                    .filter_by(session_id=session_id)
                    .order_by(ConversationRecord.created_at)
                    .offset(offset)
                    .limit(limit)
                    .all()
                )

                return [conv.to_dict() for conv in conversations]

        except Exception as e:
            self.logger.error(f"获取会话对话记录失败: {e}")
            return []

    def create_session(
        self, session_id: str, user_id: Optional[str] = None, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """创建会话记录

        Args:
            session_id: 会话ID
            user_id: 用户ID
            title: 会话标题
            metadata: 元数据

        Returns:
            是否创建成功
        """
        try:
            if hasattr(self, "mock_manager"):
                return self.mock_manager.create_session(session_id, user_id, title)

            with self.get_session() as session:
                # 检查会话是否已存在
                existing = session.query(SessionRecord).filter_by(id=session_id).first()
                if existing:
                    return True

                # 创建新会话
                session_record = SessionRecord(
                    id=session_id, user_id=user_id, title=title or f"会话_{session_id[:8]}", meta_data=metadata or {}
                )

                session.add(session_record)

                self.logger.debug(f"会话记录创建成功: {session_id}")
                return True

        except Exception as e:
            self.logger.error(f"创建会话记录失败: {e}")
            return False

    def get_user_sessions(self, user_id: str, limit: int = 50, status: str = "active") -> List[Dict[str, Any]]:
        """获取用户会话列表

        Args:
            user_id: 用户ID
            limit: 返回记录数限制
            status: 会话状态

        Returns:
            会话记录列表
        """
        try:
            if hasattr(self, "mock_manager"):
                # 模拟实现
                return [s for s in self.mock_manager.sessions.values() if s.get("user_id") == user_id][:limit]

            with self.get_session() as session:
                sessions = (
                    session.query(SessionRecord)
                    .filter_by(user_id=user_id, status=status)
                    .order_by(SessionRecord.updated_at.desc())
                    .limit(limit)
                    .all()
                )

                return [sess.to_dict() for sess in sessions]

        except Exception as e:
            self.logger.error(f"获取用户会话列表失败: {e}")
            return []

    def get_conversation_stats(self, days: int = 7) -> Dict[str, Any]:
        """获取对话统计信息

        Args:
            days: 统计天数

        Returns:
            统计信息字典
        """
        try:
            if hasattr(self, "mock_manager"):
                return {
                    "total_conversations": len(self.mock_manager.conversations),
                    "total_sessions": len(self.mock_manager.sessions),
                    "period_days": days,
                }

            with self.get_session() as session:
                # 计算时间范围
                start_date = datetime.utcnow() - timedelta(days=days)

                # 统计对话数量
                total_conversations = session.query(ConversationRecord).count()
                period_conversations = session.query(ConversationRecord).filter(ConversationRecord.created_at >= start_date).count()

                # 统计会话数量
                total_sessions = session.query(SessionRecord).count()
                period_sessions = session.query(SessionRecord).filter(SessionRecord.created_at >= start_date).count()

                return {
                    "total_conversations": total_conversations,
                    "period_conversations": period_conversations,
                    "total_sessions": total_sessions,
                    "period_sessions": period_sessions,
                    "period_days": days,
                    "start_date": start_date.isoformat(),
                }

        except Exception as e:
            self.logger.error(f"获取对话统计失败: {e}")
            return {}

    def get_connection_info(self) -> Dict[str, Any]:
        """获取MySQL连接信息

        Returns:
            连接信息字典
        """
        try:
            if hasattr(self, "mock_manager"):
                return self.mock_manager.get_connection_info()

            with self.engine.connect() as conn:
                result = conn.execute("SELECT VERSION()")
                version = result.fetchone()[0]

            return {
                "connected": True,
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "version": version,
                "pool_size": self.config.pool_size,
            }
        except Exception as e:
            self.logger.error(f"获取MySQL连接信息失败: {e}")
            return {"connected": False, "error": str(e)}
