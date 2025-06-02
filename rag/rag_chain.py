"""
RAG链模块
简化版RAG处理链，专注于基本的RAG功能，合并了simple_rag_chain.py的功能
"""

import time
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass

from logger import get_logger, log_execution_time
from config import defaultConfig
from embeddings import EmbeddingManager
from vector_store import VectorStoreManager
from llm import LLMManager
from retriever import RetrieverManager
from session_manager import RedisSessionManager

@dataclass
class RAGResponse:
    """RAG响应数据结构"""
    
    answer: str
    sources: List[str]
    retrieval_time: float
    generation_time: float


class RAGChain:
    """简化RAG链

    提供基本的RAG问答功能，整合了检索和生成，支持Redis会话管理
    """

    def __init__(self, session_id: Optional[str] = None):
        """初始化RAG链

        Args:
            session_id: 会话ID，如果为None则创建新会话
        """
        self.logger = get_logger("RAGChain")

        try:
            # 初始化组件
            self.logger.info("正在初始化RAG组件...")

            # 嵌入管理器
            self.embedding_manager = EmbeddingManager()

            # 向量存储
            self.vector_store = VectorStoreManager(self.embedding_manager)

            # 检索器 - 延迟导入避免循环依赖
            # from retriever import RetrieverManager
            self.retriever = RetrieverManager(self.vector_store, self.embedding_manager)

            # LLM
            self.llm = LLMManager()

            # Redis会话管理器
            self.session_manager = RedisSessionManager()

            # 会话管理
            if session_id:
                self.session_id = session_id
                self.logger.info(f"使用现有会话: {session_id}")
            else:
                self.session_id = self.session_manager.create_session()
                self.logger.info(f"创建新会话: {self.session_id}")

            # 加载会话历史
            self._load_session_history()

            self.logger.info("RAG链初始化完成")

        except Exception as e:
            self.logger.error(f"RAG链初始化失败: {e}")
            raise

    def _load_session_history(self):
        """从Redis加载会话历史"""
        try:
            messages = self.session_manager.get_session_messages(self.session_id)

            # 将Redis消息转换为LLM历史格式
            self.llm.clear_history()
            for message in messages:
                if message.role in ['user', 'assistant']:
                    self.llm.chat_history.append(
                        type('ChatMessage', (), {
                            'role': message.role,
                            'content': message.content,
                            'timestamp': message.timestamp,
                            'to_dict': lambda: {'role': message.role, 'content': message.content}
                        })()
                    )

            self.logger.info(f"加载会话历史: {len(messages)} 条消息")

        except Exception as e:
            self.logger.error(f"加载会话历史失败: {e}")

    def _save_to_session(self, role: str, content: str):
        """保存消息到Redis会话

        Args:
            role: 消息角色
            content: 消息内容
        """
        try:
            self.session_manager.save_message(self.session_id, role, content)
            # 延长会话过期时间
            self.session_manager.extend_session_expire(self.session_id)
        except Exception as e:
            self.logger.error(f"保存消息到会话失败: {e}")
    
    @log_execution_time("rag_query")
    def query(self, question: str, top_k: int = 5, save_to_session: bool = True) -> str:
        """单次查询

        Args:
            question: 用户问题
            top_k: 检索文档数量
            save_to_session: 是否保存到会话历史

        Returns:
            回答文本
        """
        try:
            self.logger.info(f"处理查询: {question[:50]}...")

            # 保存用户问题到会话
            if save_to_session:
                self._save_to_session("user", question)

            # 1. 检索相关文档
            start_time = time.time()

            retrieval_results = self.retriever.retrieve(question, top_k=top_k)
            retrieval_time = time.time() - start_time

            if not retrieval_results:
                answer = "抱歉，我没有找到相关的信息来回答您的问题。"
                if save_to_session:
                    self._save_to_session("assistant", answer)
                return answer

            # 2. 构建上下文
            context_docs = [result.document.page_content for result in retrieval_results]
            context = "\n\n".join(context_docs)

            # 3. 构建提示词
            prompt = self._build_prompt(question, context)
            self.logger.info(f"提示词构建完成: {prompt[:100]}...")

            # 4. 生成回答
            start_time = time.time()
            answer = self.llm.generate(prompt)
            generation_time = time.time() - start_time

            # 保存助手回答到会话
            if save_to_session:
                self._save_to_session("assistant", answer)

            self.logger.info(
                f"查询完成 | 检索时间: {retrieval_time:.3f}s | "
                f"生成时间: {generation_time:.3f}s | "
                f"检索文档: {len(retrieval_results)}"
            )

            return answer

        except Exception as e:
            self.logger.error(f"查询失败: {e}")
            error_msg = f"抱歉，处理您的问题时出现了错误: {e}"
            if save_to_session:
                self._save_to_session("assistant", error_msg)
            return error_msg
    
    def stream_chat(self, question: str, top_k: int = 5, save_to_session: bool = True) -> Generator[str, None, None]:
        """流式对话

        Args:
            question: 用户问题
            top_k: 检索文档数量
            save_to_session: 是否保存到会话历史

        Yields:
            回答文本片段
        """
        try:
            # 保存用户问题到会话
            if save_to_session:
                self._save_to_session("user", question)

            # 检索相关文档
            retrieval_results = self.retriever.retrieve(question, top_k=top_k)

            if not retrieval_results:
                error_msg = "抱歉，我没有找到相关的信息来回答您的问题。"
                if save_to_session:
                    self._save_to_session("assistant", error_msg)
                yield error_msg
                return

            # 构建上下文
            context_docs = [result.document.page_content for result in retrieval_results]
            context = "\n\n".join(context_docs)

            self.logger.info(f"上下文构建完成: context: {context[:100]}...")
            # 构建提示词
            prompt = self._build_prompt(question, context)

            # 流式生成回答
            full_response = ""
            for chunk in self.llm.stream_chat(prompt):
                full_response += chunk
                yield chunk

            # 保存完整回答到会话
            if save_to_session and full_response:
                self._save_to_session("assistant", full_response)

        except Exception as e:
            self.logger.error(f"流式对话失败: {e}")
            error_msg = f"抱歉，处理您的问题时出现了错误: {e}"
            if save_to_session:
                self._save_to_session("assistant", error_msg)
            yield error_msg
    
    def _build_prompt(self, question: str, context: str) -> str:
        """构建提示词
        
        Args:
            question: 用户问题
            context: 检索到的上下文
            
        Returns:
            构建好的提示词
        """
        prompt = f"""基于以下上下文信息，回答用户的问题。如果上下文中没有相关信息，请诚实地说不知道。

上下文信息：
{context}

用户问题：{question}

请基于上下文信息给出准确、有用的回答："""
        
        return prompt
    
    def clear_history(self):
        """清空对话历史"""
        try:
            # 清空Redis会话历史
            self.session_manager.clear_session(self.session_id)
            # 清空LLM历史
            self.llm.clear_history()
            self.logger.info(f"对话历史已清空: {self.session_id}")
        except Exception as e:
            self.logger.error(f"清空对话历史失败: {e}")

    def get_session_id(self) -> str:
        """获取当前会话ID"""
        return self.session_id

    def switch_session(self, session_id: str):
        """切换到指定会话

        Args:
            session_id: 目标会话ID
        """
        try:
            self.session_id = session_id
            self._load_session_history()
            self.logger.info(f"切换到会话: {session_id}")
        except Exception as e:
            self.logger.error(f"切换会话失败: {e}")
            raise

    def create_new_session(self, title: str = "") -> str:
        """创建新会话

        Args:
            title: 会话标题

        Returns:
            新会话ID
        """
        try:
            new_session_id = self.session_manager.create_session(title)
            self.session_id = new_session_id
            self.llm.clear_history()
            self.logger.info(f"创建新会话: {new_session_id}")
            return new_session_id
        except Exception as e:
            self.logger.error(f"创建新会话失败: {e}")
            raise

    def get_session_history(self) -> List[Dict[str, Any]]:
        """获取当前会话历史

        Returns:
            会话历史列表
        """
        try:
            messages = self.session_manager.get_session_messages(self.session_id)
            return [msg.to_dict() for msg in messages]
        except Exception as e:
            self.logger.error(f"获取会话历史失败: {e}")
            return []

    def list_all_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话

        Returns:
            会话信息列表
        """
        try:
            sessions = self.session_manager.list_sessions()
            return [session.to_dict() for session in sessions]
        except Exception as e:
            self.logger.error(f"列出会话失败: {e}")
            return []
    
    def add_documents(self, documents):
        """添加文档到知识库
        
        Args:
            documents: 文档列表
        """
        try:
            self.vector_store.add_documents(documents)
            self.logger.info(f"添加了 {len(documents)} 个文档到知识库")
        except Exception as e:
            self.logger.error(f"添加文档失败: {e}")
            raise
    
    def get_stats(self) -> dict:
        """获取RAG系统统计信息"""
        try:
            vector_stats = self.vector_store.get_stats()
            retrieval_stats = self.retriever.get_retrieval_stats()
            redis_info = self.session_manager.get_connection_info()
            session_info = self.session_manager.get_session_info(self.session_id)

            stats = {
                "vector_store": vector_stats,
                "retrieval": retrieval_stats,
                "llm_model": defaultConfig.llm.model_name,
                "current_session_id": self.session_id,
                "redis_connection": redis_info,
            }

            if session_info:
                stats["session_info"] = session_info.to_dict()

            return stats

        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}
