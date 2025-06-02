"""
动态上下文管理模块

该模块负责管理对话上下文，支持动态压缩技术减少token占用。
使用Redis存储上下文窗口，通过summarization技术压缩历史对话。

数据流：
1. 对话历史 -> 上下文窗口 -> Token计算 -> 压缩判断
2. 超长上下文 -> 摘要生成 -> 压缩存储 -> Redis保存
3. 上下文检索 -> Redis获取 -> 解压缩 -> 返回上下文

学习要点：
1. 上下文窗口管理策略
2. 动态文本摘要技术
3. Token计算和优化
4. Redis数据结构设计
"""

import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from config import defaultConfig
from logger import get_logger

# 尝试导入transformers用于摘要生成
from transformers import pipeline, AutoTokenizer

# 导入Redis会话管理器
from session_manager import RedisSessionManager


@dataclass
class ContextMessage:
    """上下文消息数据结构"""

    role: str
    content: str
    timestamp: float
    token_count: int
    is_compressed: bool = False
    original_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextMessage":
        """从字典创建实例"""
        return cls(**data)


@dataclass
class ContextWindow:
    """上下文窗口数据结构"""

    session_id: str
    messages: List[ContextMessage]
    total_tokens: int
    max_tokens: int
    compression_ratio: float
    last_updated: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "compression_ratio": self.compression_ratio,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextWindow":
        """从字典创建实例"""
        messages = [ContextMessage.from_dict(msg_data) for msg_data in data["messages"]]
        return cls(
            session_id=data["session_id"],
            messages=messages,
            total_tokens=data["total_tokens"],
            max_tokens=data["max_tokens"],
            compression_ratio=data["compression_ratio"],
            last_updated=data["last_updated"],
        )


class TextSummarizer:
    """文本摘要器"""

    def __init__(self):
        """初始化摘要器"""
        self.logger = get_logger("TextSummarizer")
        self.summarizer = None
        self.tokenizer = None

        self._init_summarizer()

    def _init_summarizer(self):
        """初始化摘要模型"""
        try:
            # 使用中文摘要模型（如果可用）或英文模型
            model_names = [
                "facebook/bart-large-cnn",  # 英文摘要模型
                "google/pegasus-xsum",  # 另一个英文摘要模型
            ]

            for model_name in model_names:
                try:
                    self.summarizer = pipeline("summarization", model=model_name, device=-1)  # 使用CPU
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.logger.info(f"摘要模型加载成功: {model_name}")
                    break
                except Exception as e:
                    self.logger.warning(f"模型 {model_name} 加载失败: {e}")
                    continue

            if not self.summarizer:
                self.logger.warning("所有摘要模型加载失败，将使用简单摘要方法")

        except Exception as e:
            self.logger.error(f"初始化摘要器失败: {e}")

    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """生成文本摘要

        Args:
            text: 输入文本
            max_length: 最大摘要长度

        Returns:
            摘要文本
        """
        try:
            if self.summarizer and len(text) > 100:
                # 使用transformers模型生成摘要
                summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
                return summary[0]["summary_text"]
            else:
                # 使用简单摘要方法
                return self._simple_summarize(text, max_length)

        except Exception as e:
            self.logger.error(f"生成摘要失败: {e}")
            return self._simple_summarize(text, max_length)

    def _simple_summarize(self, text: str, max_length: int = 150) -> str:
        """简单摘要方法

        提取文本的关键句子作为摘要
        """
        try:
            # 按句子分割
            sentences = re.split(r"[。！？.!?]", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                return text[:max_length]

            # 如果只有一句话，直接截断
            if len(sentences) == 1:
                return sentences[0][:max_length]

            # 选择前几句作为摘要
            summary_sentences = []
            current_length = 0

            for sentence in sentences:
                if current_length + len(sentence) <= max_length:
                    summary_sentences.append(sentence)
                    current_length += len(sentence)
                else:
                    break

            if not summary_sentences:
                return sentences[0][:max_length]

            return "。".join(summary_sentences) + "。"

        except Exception as e:
            self.logger.error(f"简单摘要失败: {e}")
            return text[:max_length]


class ContextManager:
    """动态上下文管理器

    负责管理对话上下文窗口，支持动态压缩和Token优化。
    使用Redis存储上下文数据，通过摘要技术减少Token占用。

    Attributes:
        redis_manager: Redis会话管理器
        summarizer: 文本摘要器
        max_context_length: 最大上下文长度
        compression_threshold: 压缩阈值
    """

    def __init__(self):
        """初始化上下文管理器"""
        self.logger = get_logger("ContextManager")
        self.config = defaultConfig.redis
        self.max_context_length = self.config.max_context_length
        self.compression_threshold = int(self.max_context_length * 0.8)  # 80%时开始压缩

        # 初始化组件
        self.redis_manager = RedisSessionManager()
        self.summarizer = TextSummarizer()

        self.logger.info(f"上下文管理器初始化完成 | 最大长度: {self.max_context_length}")

    def add_message(self, session_id: str, role: str, content: str, auto_compress: bool = True) -> bool:
        """添加消息到上下文窗口

        Args:
            session_id: 会话ID
            role: 消息角色
            content: 消息内容
            auto_compress: 是否自动压缩

        Returns:
            是否添加成功
        """
        try:
            # 计算Token数量
            token_count = self._estimate_tokens(content)

            # 创建消息对象
            message = ContextMessage(role=role, content=content, timestamp=time.time(), token_count=token_count)

            # 获取当前上下文窗口
            context_window = self.get_context_window(session_id)
            if not context_window:
                context_window = ContextWindow(
                    session_id=session_id,
                    messages=[],
                    total_tokens=0,
                    max_tokens=self.max_context_length,
                    compression_ratio=0.0,
                    last_updated=time.time(),
                )

            # 添加新消息
            context_window.messages.append(message)
            context_window.total_tokens += token_count
            context_window.last_updated = time.time()

            # 检查是否需要压缩
            if auto_compress and context_window.total_tokens > self.compression_threshold:
                context_window = self._compress_context(context_window)

            # 保存到Redis
            self._save_context_window(context_window)

            self.logger.debug(f"消息添加成功: {session_id}, tokens: {token_count}")
            return True

        except Exception as e:
            self.logger.error(f"添加消息失败: {e}")
            return False

    def get_context_window(self, session_id: str) -> Optional[ContextWindow]:
        """获取上下文窗口

        Args:
            session_id: 会话ID

        Returns:
            上下文窗口或None
        """
        try:
            context_key = f"{self.config.context_window_key}{session_id}"
            context_data = self.redis_manager.redis_client.get(context_key)

            if context_data:
                context_dict = json.loads(context_data)
                return ContextWindow.from_dict(context_dict)
            else:
                return None

        except Exception as e:
            self.logger.error(f"获取上下文窗口失败: {e}")
            return None

    def get_context_messages(self, session_id: str, max_tokens: Optional[int] = None) -> List[ContextMessage]:
        """获取上下文消息列表

        Args:
            session_id: 会话ID
            max_tokens: 最大Token数量限制

        Returns:
            上下文消息列表
        """
        try:
            context_window = self.get_context_window(session_id)
            if not context_window:
                return []

            messages = context_window.messages

            # 如果指定了Token限制，从后往前选择消息
            if max_tokens:
                selected_messages = []
                current_tokens = 0

                for message in reversed(messages):
                    if current_tokens + message.token_count <= max_tokens:
                        selected_messages.insert(0, message)
                        current_tokens += message.token_count
                    else:
                        break

                return selected_messages

            return messages

        except Exception as e:
            self.logger.error(f"获取上下文消息失败: {e}")
            return []

    def _compress_context(self, context_window: ContextWindow) -> ContextWindow:
        """压缩上下文窗口

        Args:
            context_window: 原始上下文窗口

        Returns:
            压缩后的上下文窗口
        """
        try:
            messages = context_window.messages
            if len(messages) <= 2:  # 保留最少2条消息
                return context_window

            # 保留最新的几条消息，压缩较早的消息
            keep_recent = 3  # 保留最新3条消息
            recent_messages = messages[-keep_recent:]
            old_messages = messages[:-keep_recent]

            if not old_messages:
                return context_window

            # 将旧消息按对话轮次分组
            conversation_pairs = self._group_conversation_pairs(old_messages)

            # 压缩每个对话轮次
            compressed_messages = []
            total_compressed_tokens = 0

            for pair in conversation_pairs:
                compressed_content = self._compress_conversation_pair(pair)
                if compressed_content:
                    compressed_tokens = self._estimate_tokens(compressed_content)
                    compressed_message = ContextMessage(
                        role="system",
                        content=f"[压缩摘要] {compressed_content}",
                        timestamp=pair[0].timestamp,
                        token_count=compressed_tokens,
                        is_compressed=True,
                        original_length=sum(msg.token_count for msg in pair),
                    )
                    compressed_messages.append(compressed_message)
                    total_compressed_tokens += compressed_tokens

            # 计算压缩比例
            original_tokens = sum(msg.token_count for msg in old_messages)
            compression_ratio = 1.0 - (total_compressed_tokens / original_tokens) if original_tokens > 0 else 0.0

            # 构建新的上下文窗口
            new_messages = compressed_messages + recent_messages
            new_total_tokens = sum(msg.token_count for msg in new_messages)

            compressed_window = ContextWindow(
                session_id=context_window.session_id,
                messages=new_messages,
                total_tokens=new_total_tokens,
                max_tokens=context_window.max_tokens,
                compression_ratio=compression_ratio,
                last_updated=time.time(),
            )

            self.logger.info(
                f"上下文压缩完成: {context_window.session_id} | "
                f"原始tokens: {context_window.total_tokens} -> {new_total_tokens} | "
                f"压缩比: {compression_ratio:.2%}"
            )

            return compressed_window

        except Exception as e:
            self.logger.error(f"上下文压缩失败: {e}")
            return context_window

    def _group_conversation_pairs(self, messages: List[ContextMessage]) -> List[List[ContextMessage]]:
        """将消息按对话轮次分组

        Args:
            messages: 消息列表

        Returns:
            分组后的对话轮次列表
        """
        pairs = []
        current_pair = []

        for message in messages:
            current_pair.append(message)

            # 如果是assistant消息，结束当前轮次
            if message.role == "assistant":
                pairs.append(current_pair)
                current_pair = []

        # 处理剩余的消息
        if current_pair:
            pairs.append(current_pair)

        return pairs

    def _compress_conversation_pair(self, messages: List[ContextMessage]) -> str:
        """压缩对话轮次

        Args:
            messages: 对话轮次中的消息列表

        Returns:
            压缩后的文本
        """
        try:
            # 合并对话内容
            conversation_text = ""
            for message in messages:
                role_prefix = {"user": "用户", "assistant": "助手", "system": "系统"}.get(message.role, message.role)
                conversation_text += f"{role_prefix}: {message.content}\n"

            # 生成摘要
            max_summary_length = min(100, len(conversation_text) // 3)
            summary = self.summarizer.summarize_text(conversation_text, max_summary_length)

            return summary

        except Exception as e:
            self.logger.error(f"压缩对话轮次失败: {e}")
            return ""

    def _save_context_window(self, context_window: ContextWindow):
        """保存上下文窗口到Redis

        Args:
            context_window: 上下文窗口
        """
        try:
            context_key = f"{self.config.context_window_key}{context_window.session_id}"
            context_data = json.dumps(context_window.to_dict(), ensure_ascii=False)

            # 设置过期时间
            self.redis_manager.redis_client.setex(context_key, self.config.session_expire, context_data)

        except Exception as e:
            self.logger.error(f"保存上下文窗口失败: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """估算文本的Token数量

        Args:
            text: 输入文本

        Returns:
            估算的Token数量
        """
        try:
            # 简单的Token估算：中文字符按1个token计算，英文单词按平均4个字符1个token计算
            chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
            english_chars = len(re.findall(r"[a-zA-Z]", text))
            other_chars = len(text) - chinese_chars - english_chars

            # 估算公式
            estimated_tokens = chinese_chars + (english_chars // 4) + (other_chars // 2)

            return max(1, estimated_tokens)

        except Exception as e:
            self.logger.error(f"Token估算失败: {e}")
            return len(text) // 4  # 简单回退方案

    def clear_context(self, session_id: str) -> bool:
        """清空会话上下文

        Args:
            session_id: 会话ID

        Returns:
            是否清空成功
        """
        try:
            context_key = f"{self.config.context_window_key}{session_id}"
            self.redis_manager.redis_client.delete(context_key)

            self.logger.info(f"上下文已清空: {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"清空上下文失败: {e}")
            return False

    def get_context_stats(self, session_id: str) -> Dict[str, Any]:
        """获取上下文统计信息

        Args:
            session_id: 会话ID

        Returns:
            统计信息字典
        """
        try:
            context_window = self.get_context_window(session_id)
            if not context_window:
                return {"exists": False}

            compressed_count = sum(1 for msg in context_window.messages if msg.is_compressed)

            return {
                "exists": True,
                "total_messages": len(context_window.messages),
                "total_tokens": context_window.total_tokens,
                "max_tokens": context_window.max_tokens,
                "compression_ratio": context_window.compression_ratio,
                "compressed_messages": compressed_count,
                "last_updated": context_window.last_updated,
                "utilization": context_window.total_tokens / context_window.max_tokens,
            }

        except Exception as e:
            self.logger.error(f"获取上下文统计失败: {e}")
            return {"exists": False, "error": str(e)}
