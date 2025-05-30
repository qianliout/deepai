"""
大语言模型接口模块

该模块负责与通义百炼大模型API的交互，提供文本生成和对话功能。
支持流式输出、重试机制和错误处理。

数据流：
1. 用户输入 -> 提示词构建 -> API调用 -> 响应解析 -> 结果返回
2. 支持上下文管理和对话历史
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Generator
from dataclasses import dataclass
import json

try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk

from config import config
from logger import get_logger, log_execution_time, LogExecutionTime


@dataclass
class ChatMessage:
    """聊天消息数据结构"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式"""
        return {"role": self.role, "content": self.content}


class LLMManager(LLM):
    """大语言模型管理器

    基于通义百炼API实现的大语言模型接口，
    支持文本生成、对话和流式输出。

    Attributes:
        api_key: API密钥
        model_name: 模型名称
        generation_config: 生成配置参数
        chat_history: 对话历史
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """初始化LLM管理器

        Args:
            api_key: API密钥，默认使用配置中的密钥
            model_name: 模型名称，默认使用配置中的模型
        """
        super().__init__()

        if not DASHSCOPE_AVAILABLE:
            raise ImportError("dashscope未安装，请运行: pip install dashscope")

        self.logger = get_logger("LLMManager")
        self.api_key = api_key or config.llm.api_key
        self.model_name = model_name or config.llm.model_name
        self.chat_history: List[ChatMessage] = []

        # 设置API密钥
        if not self.api_key:
            raise ValueError("API密钥未设置，请在配置文件中设置或通过环境变量DASHSCOPE_API_KEY设置")

        dashscope.api_key = self.api_key

        # 生成配置
        self.generation_config = {
            "model": self.model_name,
            "temperature": config.llm.temperature,
            "top_p": config.llm.top_p,
            "max_tokens": config.llm.max_tokens,
        }

        self.logger.info(f"LLM管理器初始化成功 | 模型: {self.model_name}")

    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "dashscope_qwen"

    @log_execution_time("llm_generate")
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """调用LLM生成文本

        Args:
            prompt: 输入提示词
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 额外参数

        Returns:
            生成的文本
        """
        try:
            self.logger.debug(f"开始生成文本，提示词长度: {len(prompt)}")

            # 准备消息
            messages = [{"role": "user", "content": prompt}]

            # 调用API
            response = self._call_api(messages, **kwargs)

            # 提取生成的文本
            if response and response.output and response.output.choices:
                generated_text = response.output.choices[0].message.content
                self.logger.debug(f"文本生成成功，长度: {len(generated_text)}")
                return generated_text
            else:
                raise ValueError("API响应格式错误")

        except Exception as e:
            self.logger.error(f"文本生成失败: {e}")
            raise

    def _call_api(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """调用通义千问API

        Args:
            messages: 消息列表
            **kwargs: 额外参数

        Returns:
            API响应
        """
        # 合并配置参数
        params = {**self.generation_config, **kwargs}
        params["messages"] = messages

        # 重试机制
        max_retries = config.llm.max_retries
        for attempt in range(max_retries + 1):
            try:
                response = Generation.call(**params)

                if response.status_code == 200:
                    return response
                else:
                    error_msg = f"API调用失败，状态码: {response.status_code}, 错误: {response.message}"
                    if attempt == max_retries:
                        raise Exception(error_msg)
                    else:
                        self.logger.warning(f"{error_msg}，正在重试 ({attempt + 1}/{max_retries})")
                        time.sleep(2 ** attempt)  # 指数退避

            except Exception as e:
                if attempt == max_retries:
                    raise
                else:
                    self.logger.warning(f"API调用异常: {e}，正在重试 ({attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)

    @log_execution_time("llm_chat")
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        clear_history: bool = False
    ) -> str:
        """对话接口

        Args:
            message: 用户消息
            system_prompt: 系统提示词
            clear_history: 是否清空历史记录

        Returns:
            助手回复
        """
        try:
            if clear_history:
                self.clear_history()

            # 构建消息列表
            messages = []

            # 添加系统提示词
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # 添加历史对话
            for msg in self.chat_history:
                messages.append(msg.to_dict())

            # 添加当前用户消息
            messages.append({"role": "user", "content": message})

            # 调用API
            response = self._call_api(messages)

            # 提取回复
            if response and response.output and response.output.choices:
                assistant_reply = response.output.choices[0].message.content

                # 更新对话历史
                self.chat_history.append(ChatMessage("user", message, time.time()))
                self.chat_history.append(ChatMessage("assistant", assistant_reply, time.time()))

                return assistant_reply
            else:
                raise ValueError("API响应格式错误")

        except Exception as e:
            self.logger.error(f"对话失败: {e}")
            raise

    def stream_chat(
        self,
        message: str,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """流式对话接口

        Args:
            message: 用户消息
            system_prompt: 系统提示词

        Yields:
            流式生成的文本片段
        """
        try:
            # 构建消息列表
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for msg in self.chat_history:
                messages.append(msg.to_dict())

            messages.append({"role": "user", "content": message})

            # 流式调用API
            params = {**self.generation_config}
            params["messages"] = messages
            params["stream"] = True

            full_response = ""

            for response in Generation.call(**params):
                if response.status_code == 200:
                    if response.output and response.output.choices:
                        chunk = response.output.choices[0].message.content
                        if chunk:
                            full_response += chunk
                            yield chunk
                else:
                    raise Exception(f"流式API调用失败: {response.message}")

            # 更新对话历史
            self.chat_history.append(ChatMessage("user", message, time.time()))
            self.chat_history.append(ChatMessage("assistant", full_response, time.time()))

        except Exception as e:
            self.logger.error(f"流式对话失败: {e}")
            raise

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """基于上下文生成回答（RAG核心功能）

        Args:
            query: 用户查询
            context: 检索到的上下文
            system_prompt: 系统提示词

        Returns:
            基于上下文的回答
        """
        # 构建RAG提示词
        if not system_prompt:
            system_prompt = """你是一个智能助手，请基于提供的上下文信息回答用户的问题。
要求：
1. 回答必须基于上下文信息，不要编造不存在的信息
2. 如果上下文中没有相关信息，请明确说明
3. 回答要准确、简洁、有条理
4. 可以适当引用上下文中的具体内容"""

        prompt = f"""上下文信息：
{context}

用户问题：{query}

请基于上述上下文信息回答用户问题："""

        return self.chat(prompt, system_prompt, clear_history=True)

    def clear_history(self) -> None:
        """清空对话历史"""
        self.chat_history.clear()
        self.logger.debug("对话历史已清空")

    def get_history(self) -> List[Dict[str, Any]]:
        """获取对话历史

        Returns:
            对话历史列表
        """
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in self.chat_history
        ]

    def save_history(self, filepath: str) -> None:
        """保存对话历史到文件

        Args:
            filepath: 保存路径
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.get_history(), f, ensure_ascii=False, indent=2)
            self.logger.info(f"对话历史已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"保存对话历史失败: {e}")
            raise

    def load_history(self, filepath: str) -> None:
        """从文件加载对话历史

        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                history_data = json.load(f)

            self.chat_history = [
                ChatMessage(
                    role=item["role"],
                    content=item["content"],
                    timestamp=item.get("timestamp")
                )
                for item in history_data
            ]

            self.logger.info(f"对话历史已从 {filepath} 加载，共 {len(self.chat_history)} 条记录")

        except Exception as e:
            self.logger.error(f"加载对话历史失败: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息

        Returns:
            模型信息字典
        """
        return {
            "model_name": self.model_name,
            "temperature": config.llm.temperature,
            "top_p": config.llm.top_p,
            "max_tokens": config.llm.max_tokens,
            "chat_history_length": len(self.chat_history)
        }
