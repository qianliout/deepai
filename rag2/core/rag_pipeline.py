"""
RAG管道
整合检索、重排序和生成的完整RAG流程
"""

import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ..config.config import get_config
    from ..utils.logger import get_logger, log_performance, LogContext
    from ..models.llm_client import get_llm_client
    from ..models.rerank_models import get_rerank_manager
    from ..retrieval.base_retriever import get_retriever_manager
    from ..storage.redis_manager import RedisManager
    from ..storage.mysql_manager import MySQLManager
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from config.config import get_config
    from utils.logger import get_logger, log_performance, LogContext
    from models.llm_client import get_llm_client
    from models.rerank_models import get_rerank_manager
    from retrieval.base_retriever import get_retriever_manager
    from storage.redis_manager import RedisManager
    from storage.mysql_manager import MySQLManager

logger = get_logger("rag_pipeline")

class RAGPipeline:
    """RAG管道"""
    
    def __init__(self):
        self.config = get_config()
        self.llm_client = None
        self.rerank_manager = None
        self.retriever_manager = None
        self.redis_manager = None
        self.mysql_manager = None
    
    async def _initialize_components(self):
        """初始化组件"""
        if self.llm_client is None:
            self.llm_client = await get_llm_client()
        
        if self.rerank_manager is None:
            self.rerank_manager = get_rerank_manager()
        
        if self.retriever_manager is None:
            self.retriever_manager = get_retriever_manager()
        
        if self.redis_manager is None:
            self.redis_manager = RedisManager()
            await self.redis_manager.initialize()
        
        if self.mysql_manager is None:
            self.mysql_manager = MySQLManager()
            await self.mysql_manager.initialize()
    
    @log_performance()
    async def query(self, query: str, session_id: str = None, 
                   user_id: str = None, **kwargs) -> Dict[str, Any]:
        """执行RAG查询"""
        start_time = time.time()
        
        with LogContext("RAG查询", query=query, session_id=session_id):
            await self._initialize_components()
            
            try:
                # 1. 查询预处理
                processed_query = await self._preprocess_query(query, session_id)
                
                # 2. 判断是否需要检索
                need_retrieval = await self._should_retrieve(processed_query, session_id)
                
                retrieved_docs = []
                context = ""
                
                if need_retrieval:
                    # 3. 检索相关文档
                    retrieved_docs = await self._retrieve_documents(processed_query, **kwargs)
                    
                    # 4. 重排序
                    if retrieved_docs:
                        retrieved_docs = await self._rerank_documents(processed_query, retrieved_docs)
                    
                    # 5. 构建上下文
                    context = self._build_context(retrieved_docs)
                
                # 6. 获取对话历史
                conversation_history = await self._get_conversation_history(session_id)
                
                # 7. 生成回答
                response = await self._generate_response(
                    query=processed_query,
                    context=context,
                    conversation_history=conversation_history
                )
                
                # 8. 记录对话
                if session_id:
                    await self._save_conversation(session_id, query, response, retrieved_docs)
                
                # 9. 记录查询日志
                end_time = time.time()
                duration = (end_time - start_time) * 1000  # 转换为毫秒
                
                await self._log_query(
                    session_id=session_id,
                    user_query=query,
                    processed_query=processed_query,
                    retrieved_docs=retrieved_docs,
                    response=response,
                    duration=duration
                )
                
                return {
                    "query": query,
                    "response": response,
                    "retrieved_documents": retrieved_docs,
                    "context_used": bool(context),
                    "processing_time_ms": duration,
                    "session_id": session_id
                }
                
            except Exception as e:
                logger.error(f"RAG查询失败: {str(e)}")
                
                # 记录错误
                if session_id:
                    await self._log_query(
                        session_id=session_id,
                        user_query=query,
                        error_message=str(e),
                        success=False
                    )
                
                return {
                    "query": query,
                    "response": "抱歉，处理您的查询时遇到了问题，请稍后重试。",
                    "error": str(e),
                    "retrieved_documents": [],
                    "context_used": False,
                    "session_id": session_id
                }
    
    async def _preprocess_query(self, query: str, session_id: str = None) -> str:
        """查询预处理"""
        # 简单的查询清理
        processed_query = query.strip()
        
        # TODO: 添加更多预处理逻辑
        # - 查询扩展
        # - 同义词替换
        # - 拼写纠错
        
        return processed_query
    
    async def _should_retrieve(self, query: str, session_id: str = None) -> bool:
        """判断是否需要检索"""
        try:
            # 获取对话上下文
            context = []
            if session_id:
                context = await self.redis_manager.get_conversation_context(session_id, max_messages=6)
            
            # 使用LLM判断
            should_retrieve = await self.llm_client.should_retrieve(query, context)
            
            logger.info(f"检索决策: 查询='{query}', 需要检索={should_retrieve}")
            return should_retrieve
            
        except Exception as e:
            logger.warning(f"检索决策失败，默认进行检索: {str(e)}")
            return True
    
    async def _retrieve_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """检索相关文档"""
        try:
            top_k = kwargs.get("top_k", self.config.rag.retrieval_top_k)
            
            # 使用默认检索器
            results = await self.retriever_manager.retrieve(
                query=query,
                top_k=top_k,
                **kwargs
            )
            
            logger.info(f"文档检索完成: 查询='{query}', 结果数={len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
            return []
    
    async def _rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重排序文档"""
        try:
            if not documents:
                return documents
            
            top_k = self.config.rag.rerank_top_k
            
            reranked_docs = await self.rerank_manager.rerank_search_results_async(
                query=query,
                search_results=documents,
                content_field="content",
                top_k=top_k
            )
            
            logger.info(f"文档重排序完成: 输入{len(documents)}个, 输出{len(reranked_docs)}个")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"文档重排序失败: {str(e)}")
            return documents[:self.config.rag.rerank_top_k]
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """构建上下文"""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            title = doc.get("title", "")
            source = doc.get("source", "")
            
            context_part = f"文档{i}:"
            if title:
                context_part += f" 标题: {title}"
            if source:
                context_part += f" 来源: {source}"
            context_part += f"\n内容: {content}\n"
            
            context_parts.append(context_part)
        
        context = "\n".join(context_parts)
        
        # 检查上下文长度
        if len(context) > self.config.rag.max_context_length:
            # 简单截断
            context = context[:self.config.rag.max_context_length] + "..."
            logger.warning(f"上下文过长，已截断到{self.config.rag.max_context_length}字符")
        
        return context
    
    async def _get_conversation_history(self, session_id: str = None) -> List[Dict[str, str]]:
        """获取对话历史"""
        if not session_id:
            return []
        
        try:
            # 从Redis获取最近的对话
            messages = await self.redis_manager.get_conversation_context(
                session_id, 
                max_messages=self.config.rag.max_conversation_turns * 2
            )
            
            # 转换格式
            conversation = []
            for msg in messages:
                conversation.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            return conversation
            
        except Exception as e:
            logger.warning(f"获取对话历史失败: {str(e)}")
            return []
    
    async def _generate_response(self, query: str, context: str, 
                               conversation_history: List[Dict[str, str]]) -> str:
        """生成回答"""
        try:
            # 构建提示词
            system_prompt = """你是一个专业的AIOps助手，专门帮助用户解决运维和安全相关的问题。

请根据提供的上下文信息回答用户的问题。如果上下文中没有相关信息，请基于你的知识给出有用的建议。

回答要求：
1. 准确、专业、有用
2. 如果涉及安全问题，要特别谨慎和详细
3. 提供具体的操作建议
4. 保持友好和专业的语调"""

            messages = [{"role": "system", "content": system_prompt}]
            
            # 添加对话历史
            messages.extend(conversation_history[-6:])  # 最近3轮对话
            
            # 构建用户消息
            user_message = f"用户问题: {query}"
            if context:
                user_message += f"\n\n相关上下文:\n{context}"
            
            messages.append({"role": "user", "content": user_message})
            
            # 生成回答
            response = await self.llm_client.generate_response(
                messages=messages,
                temperature=0.1,
                max_tokens=2048
            )
            
            return response
            
        except Exception as e:
            logger.error(f"回答生成失败: {str(e)}")
            return "抱歉，我无法生成回答，请稍后重试。"
    
    async def _save_conversation(self, session_id: str, query: str, 
                               response: str, retrieved_docs: List[Dict[str, Any]]):
        """保存对话记录"""
        try:
            # 保存到Redis（实时访问）
            await self.redis_manager.add_conversation_message(
                session_id=session_id,
                role="user",
                content=query
            )
            
            await self.redis_manager.add_conversation_message(
                session_id=session_id,
                role="assistant",
                content=response,
                metadata={"retrieved_docs_count": len(retrieved_docs)}
            )
            
            # 保存到MySQL（持久化）
            await self.mysql_manager.add_conversation(
                session_id=session_id,
                message_type="user",
                content=query
            )
            
            await self.mysql_manager.add_conversation(
                session_id=session_id,
                message_type="assistant",
                content=response,
                metadata={"retrieved_docs_count": len(retrieved_docs)}
            )
            
        except Exception as e:
            logger.error(f"保存对话记录失败: {str(e)}")
    
    async def _log_query(self, session_id: str = None, user_query: str = "", 
                        processed_query: str = "", retrieved_docs: List = None,
                        response: str = "", duration: float = 0,
                        error_message: str = None, success: bool = True):
        """记录查询日志"""
        try:
            # 分类查询
            query_classification = await self.llm_client.classify_query(user_query)
            
            # 记录到MySQL
            query_log_id = await self.mysql_manager.log_query(
                session_id=session_id,
                user_query=user_query,
                processed_query=processed_query,
                query_type=query_classification.get("type", "general"),
                intent=query_classification.get("intent", ""),
                entities=query_classification.get("entities", []),
                processing_time_ms=int(duration),
                success=success,
                error_message=error_message
            )
            
            # 记录检索日志
            if retrieved_docs and success:
                await self.mysql_manager.log_retrieval(
                    query_log_id=query_log_id,
                    retrieval_method="semantic",
                    retrieved_documents=retrieved_docs,
                    document_count=len(retrieved_docs),
                    retrieval_time_ms=int(duration * 0.3)  # 估算检索时间
                )
            
        except Exception as e:
            logger.error(f"记录查询日志失败: {str(e)}")
    
    async def stream_query(self, query: str, session_id: str = None, 
                          user_id: str = None, **kwargs) -> AsyncGenerator[str, None]:
        """流式RAG查询"""
        await self._initialize_components()
        
        try:
            # 执行检索部分
            processed_query = await self._preprocess_query(query, session_id)
            need_retrieval = await self._should_retrieve(processed_query, session_id)
            
            retrieved_docs = []
            context = ""
            
            if need_retrieval:
                retrieved_docs = await self._retrieve_documents(processed_query, **kwargs)
                if retrieved_docs:
                    retrieved_docs = await self._rerank_documents(processed_query, retrieved_docs)
                context = self._build_context(retrieved_docs)
            
            conversation_history = await self._get_conversation_history(session_id)
            
            # 构建消息
            system_prompt = "你是一个专业的AIOps助手..."  # 同上
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history[-6:])
            
            user_message = f"用户问题: {query}"
            if context:
                user_message += f"\n\n相关上下文:\n{context}"
            messages.append({"role": "user", "content": user_message})
            
            # 流式生成
            full_response = ""
            async for chunk in self.llm_client.generate_stream(messages):
                full_response += chunk
                yield chunk
            
            # 保存完整对话
            if session_id:
                await self._save_conversation(session_id, query, full_response, retrieved_docs)
            
        except Exception as e:
            logger.error(f"流式RAG查询失败: {str(e)}")
            yield f"抱歉，处理您的查询时遇到了问题: {str(e)}"
    
    async def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        await self._initialize_components()
        
        results = {}
        
        # 检查各组件
        try:
            results["llm"] = await self.llm_client.health_check()
        except:
            results["llm"] = False
        
        try:
            results["reranker"] = self.rerank_manager.health_check()
        except:
            results["reranker"] = False
        
        try:
            results["retrievers"] = await self.retriever_manager.health_check_all()
        except:
            results["retrievers"] = {}
        
        try:
            results["redis"] = await self.redis_manager.health_check()
        except:
            results["redis"] = False
        
        try:
            results["mysql"] = await self.mysql_manager.health_check()
        except:
            results["mysql"] = False
        
        return results

# 全局RAG管道实例
_rag_pipeline = None

async def get_rag_pipeline() -> RAGPipeline:
    """获取RAG管道实例"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
