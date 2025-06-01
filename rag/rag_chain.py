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


@dataclass
class RAGResponse:
    """RAG响应数据结构"""
    
    answer: str
    sources: List[str]
    retrieval_time: float
    generation_time: float


class RAGChain:
    """简化RAG链
    
    提供基本的RAG问答功能，整合了检索和生成
    """
    
    def __init__(self):
        """初始化RAG链"""
        self.logger = get_logger("RAGChain")
        
        try:
            # 初始化组件
            self.logger.info("正在初始化RAG组件...")
            
            # 嵌入管理器
            self.embedding_manager = EmbeddingManager()
            
            # 向量存储
            self.vector_store = VectorStoreManager(self.embedding_manager)
            
            # 检索器 - 延迟导入避免循环依赖
            from retriever import RetrieverManager
            self.retriever = RetrieverManager(self.vector_store, self.embedding_manager)
            
            # LLM
            self.llm = LLMManager()
            
            # 对话历史
            self.chat_history = []
            
            self.logger.info("RAG链初始化完成")
            
        except Exception as e:
            self.logger.error(f"RAG链初始化失败: {e}")
            raise
    
    @log_execution_time("rag_query")
    def query(self, question: str, top_k: int = 5) -> str:
        """单次查询
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            
        Returns:
            回答文本
        """
        try:
            self.logger.info(f"处理查询: {question[:50]}...")
            
            # 1. 检索相关文档
            start_time = time.time()
            
            retrieval_results = self.retriever.retrieve(question, top_k=top_k)
            retrieval_time = time.time() - start_time
            
            if not retrieval_results:
                return "抱歉，我没有找到相关的信息来回答您的问题。"
            
            # 2. 构建上下文
            context_docs = [result.document.page_content for result in retrieval_results]
            context = "\n\n".join(context_docs)
            
            # 3. 构建提示词
            prompt = self._build_prompt(question, context)
            
            # 4. 生成回答
            start_time = time.time()
            answer = self.llm.generate(prompt)
            generation_time = time.time() - start_time
            
            self.logger.info(
                f"查询完成 | 检索时间: {retrieval_time:.3f}s | "
                f"生成时间: {generation_time:.3f}s | "
                f"检索文档: {len(retrieval_results)}"
            )
            
            return answer
            
        except Exception as e:
            self.logger.error(f"查询失败: {e}")
            return f"抱歉，处理您的问题时出现了错误: {e}"
    
    def stream_chat(self, question: str, top_k: int = 5) -> Generator[str, None, None]:
        """流式对话
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            
        Yields:
            回答文本片段
        """
        try:
            # 检索相关文档
            retrieval_results = self.retriever.retrieve(question, top_k=top_k)
            
            if not retrieval_results:
                yield "抱歉，我没有找到相关的信息来回答您的问题。"
                return
            
            # 构建上下文
            context_docs = [result.document.page_content for result in retrieval_results]
            context = "\n\n".join(context_docs)
            
            # 构建提示词
            prompt = self._build_prompt(question, context)
            
            # 流式生成回答
            for chunk in self.llm.stream_chat(prompt):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"流式对话失败: {e}")
            yield f"抱歉，处理您的问题时出现了错误: {e}"
    
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
        self.chat_history = []
        self.llm.clear_history()
        self.logger.info("对话历史已清空")
    
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
            
            return {
                "vector_store": vector_stats,
                "retrieval": retrieval_stats,
                "llm_model": defaultConfig.llm.model_name,
                "chat_history_length": len(self.chat_history),
            }
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}
