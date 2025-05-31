"""
RAG链模块

该模块实现完整的RAG处理链，整合检索和生成功能。
提供端到端的问答服务，支持多种RAG策略和优化技术。

数据流：
1. 用户查询 -> 检索相关文档 -> 构建上下文 -> 生成回答 -> 后处理 -> 返回结果
2. 支持流式输出和批量处理
"""

import time
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, asdict
import json

from langchain_core.documents import Document
from langchain_core.runnables import Runnable


from logger import get_logger, log_execution_time, LogExecutionTime
from retriever import RetrieverManager, RetrievalResult
from llm import LLMManager
from embeddings import EmbeddingManager
from vector_store import VectorStoreManager


@dataclass
class RAGResponse:
    """RAG响应数据结构"""

    query: str
    answer: str
    retrieved_documents: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float
    total_time: float
    confidence_score: float
    sources: List[str]
    metadata: Dict[str, Any]


class RAGChain(Runnable):
    """RAG处理链

    整合检索器和生成器，提供完整的RAG问答服务。
    支持多种RAG策略、上下文优化和结果后处理。

    Attributes:
        retriever: 检索器管理器
        llm: 大语言模型管理器
        embedding_manager: 嵌入管理器
        vector_store: 向量存储管理器
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        vector_store: Optional[VectorStoreManager] = None,
        llm: Optional[LLMManager] = None,
        retriever: Optional[RetrieverManager] = None,
    ):
        """初始化RAG链

        Args:
            embedding_manager: 嵌入管理器
            vector_store: 向量存储管理器
            llm: 大语言模型管理器
            retriever: 检索器管理器
        """
        self.logger = get_logger("RAGChain")

        # 初始化组件
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.vector_store = vector_store or VectorStoreManager(self.embedding_manager)
        self.llm = llm or LLMManager()
        self.retriever = retriever or RetrieverManager(self.vector_store, self.embedding_manager)

        # RAG配置
        self.max_context_length = config.llm.max_tokens // 2  # 为生成预留空间
        self.min_confidence_threshold = 0.5

        # 提示词模板
        self.system_prompt = self._get_system_prompt()
        self.context_template = self._get_context_template()

        self.logger.info("RAG链初始化完成")

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个智能助手，专门基于提供的上下文信息回答用户问题。

请遵循以下原则：
1. 仅基于提供的上下文信息回答问题，不要编造不存在的信息
2. 如果上下文中没有足够信息回答问题，请明确说明
3. 回答要准确、简洁、有条理
4. 可以适当引用上下文中的具体内容
5. 如果问题涉及多个方面，请分点回答
6. 保持客观中立的语调"""

    def _get_context_template(self) -> str:
        """获取上下文模板"""
        return """基于以下上下文信息回答用户问题：

上下文信息：
{context}

用户问题：{question}

请基于上述上下文信息提供准确的回答："""

    @log_execution_time("rag_invoke")
    def invoke(self, input_data: Dict[str, Any], config_data: Optional[Dict] = None) -> RAGResponse:
        """执行RAG处理（Runnable接口实现）

        Args:
            input_data: 输入数据，包含query字段
            config_data: 配置数据

        Returns:
            RAG响应对象
        """
        query = input_data.get("query", "")
        if not query:
            raise ValueError("查询不能为空")

        return self.process_query(query, **(config_data or {}))

    @log_execution_time("rag_process_query")
    def process_query(
        self, query: str, top_k: Optional[int] = None, retrieval_strategy: Optional[str] = None, include_sources: bool = True, **kwargs
    ) -> RAGResponse:
        """处理查询请求

        Args:
            query: 用户查询
            top_k: 检索文档数量
            retrieval_strategy: 检索策略
            include_sources: 是否包含来源信息
            **kwargs: 额外参数

        Returns:
            RAG响应对象
        """
        start_time = time.time()

        try:
            self.logger.info(f"开始处理查询: {query[:100]}...")

            # 1. 检索相关文档
            retrieval_start = time.time()
            retrieved_results = self.retriever.retrieve(
                query=query, top_k=top_k or config.chromadb.top_k, strategy=retrieval_strategy, **kwargs
            )
            retrieval_time = time.time() - retrieval_start

            if not retrieved_results:
                return self._create_no_results_response(query, retrieval_time)

            # 2. 构建上下文
            context = self._build_context(retrieved_results)

            # 3. 生成回答
            generation_start = time.time()
            answer = self._generate_answer(query, context)
            generation_time = time.time() - generation_start

            # 4. 计算置信度
            confidence_score = self._calculate_confidence(retrieved_results, answer)

            # 5. 提取来源信息
            sources = self._extract_sources(retrieved_results) if include_sources else []

            # 6. 构建响应
            total_time = time.time() - start_time

            response = RAGResponse(
                query=query,
                answer=answer,
                retrieved_documents=[
                    {
                        "content": result.document.page_content,
                        "metadata": result.document.metadata,
                        "score": result.score,
                        "rank": result.rank,
                        "method": result.retrieval_method,
                    }
                    for result in retrieved_results
                ],
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                confidence_score=confidence_score,
                sources=sources,
                metadata={
                    "retrieval_strategy": retrieval_strategy or "default",
                    "context_length": len(context),
                    "num_retrieved_docs": len(retrieved_results),
                },
            )

            self.logger.info(
                f"查询处理完成 | 耗时: {total_time:.3f}s | "
                f"检索: {retrieval_time:.3f}s | 生成: {generation_time:.3f}s | "
                f"置信度: {confidence_score:.3f}"
            )

            return response

        except Exception as e:
            self.logger.error(f"查询处理失败: {e}")
            raise

    def _build_context(self, retrieved_results: List[RetrievalResult]) -> str:
        """构建上下文

        Args:
            retrieved_results: 检索结果列表

        Returns:
            构建的上下文字符串
        """
        context_parts = []
        current_length = 0

        for i, result in enumerate(retrieved_results):
            # 格式化文档内容
            doc_content = result.document.page_content.strip()

            # 添加文档标识
            doc_header = f"文档 {i+1} (相似度: {result.score:.3f}):"
            doc_text = f"{doc_header}\n{doc_content}\n"

            # 检查长度限制
            if current_length + len(doc_text) > self.max_context_length:
                # 如果添加当前文档会超出限制，尝试截断
                remaining_length = self.max_context_length - current_length - len(doc_header) - 10
                if remaining_length > 100:  # 至少保留100字符
                    truncated_content = doc_content[:remaining_length] + "..."
                    doc_text = f"{doc_header}\n{truncated_content}\n"
                    context_parts.append(doc_text)
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """生成回答

        Args:
            query: 用户查询
            context: 上下文信息

        Returns:
            生成的回答
        """
        try:
            # 构建完整提示词
            prompt = self.context_template.format(context=context, question=query)

            # 使用LLM生成回答
            answer = self.llm.generate_with_context(query=query, context=context, system_prompt=self.system_prompt)

            return answer.strip()

        except Exception as e:
            self.logger.error(f"回答生成失败: {e}")
            return "抱歉，生成回答时出现错误。"

    def _calculate_confidence(self, retrieved_results: List[RetrievalResult], answer: str) -> float:
        """计算置信度分数

        Args:
            retrieved_results: 检索结果
            answer: 生成的回答

        Returns:
            置信度分数 [0, 1]
        """
        if not retrieved_results:
            return 0.0

        try:
            # 基于检索分数计算基础置信度
            avg_retrieval_score = sum(r.score for r in retrieved_results) / len(retrieved_results)

            # 基于回答长度的调整因子
            answer_length_factor = min(len(answer) / 100, 1.0)  # 100字符为基准

            # 基于检索结果数量的调整因子
            num_docs_factor = min(len(retrieved_results) / 3, 1.0)  # 3个文档为基准

            # 综合置信度
            confidence = 0.6 * avg_retrieval_score + 0.2 * answer_length_factor + 0.2 * num_docs_factor

            return min(max(confidence, 0.0), 1.0)

        except Exception as e:
            self.logger.warning(f"置信度计算失败: {e}")
            return 0.5  # 默认中等置信度

    def _extract_sources(self, retrieved_results: List[RetrievalResult]) -> List[str]:
        """提取来源信息

        Args:
            retrieved_results: 检索结果

        Returns:
            来源列表
        """
        sources = []
        seen_sources = set()

        for result in retrieved_results:
            metadata = result.document.metadata

            # 提取文件来源
            source = metadata.get("source", metadata.get("filename", "未知来源"))

            # 添加页码信息（如果有）
            if "page_number" in metadata:
                source += f" (第{metadata['page_number']}页)"
            elif "chunk_index" in metadata:
                source += f" (片段{metadata['chunk_index']})"

            # 去重
            if source not in seen_sources:
                sources.append(source)
                seen_sources.add(source)

        return sources

    def _create_no_results_response(self, query: str, retrieval_time: float) -> RAGResponse:
        """创建无结果响应"""
        return RAGResponse(
            query=query,
            answer="抱歉，我没有找到与您的问题相关的信息。请尝试重新表述您的问题或提供更多详细信息。",
            retrieved_documents=[],
            retrieval_time=retrieval_time,
            generation_time=0.0,
            total_time=retrieval_time,
            confidence_score=0.0,
            sources=[],
            metadata={"no_results": True},
        )

    def stream_process_query(self, query: str, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """流式处理查询

        Args:
            query: 用户查询
            **kwargs: 额外参数

        Yields:
            流式响应数据
        """
        try:
            self.logger.info(f"开始流式处理查询: {query[:100]}...")

            # 发送开始信号
            yield {"type": "start", "query": query}

            # 1. 检索阶段
            yield {"type": "retrieval_start"}

            retrieval_start = time.time()
            retrieved_results = self.retriever.retrieve(query, **kwargs)
            retrieval_time = time.time() - retrieval_start

            yield {"type": "retrieval_complete", "num_documents": len(retrieved_results), "retrieval_time": retrieval_time}

            if not retrieved_results:
                yield {"type": "complete", "answer": "抱歉，没有找到相关信息。", "confidence_score": 0.0}
                return

            # 2. 生成阶段
            yield {"type": "generation_start"}

            context = self._build_context(retrieved_results)

            # 流式生成回答
            full_answer = ""
            for chunk in self.llm.stream_chat(message=query, system_prompt=self.system_prompt):
                full_answer += chunk
                yield {"type": "generation_chunk", "chunk": chunk, "partial_answer": full_answer}

            # 3. 完成阶段
            confidence_score = self._calculate_confidence(retrieved_results, full_answer)
            sources = self._extract_sources(retrieved_results)

            yield {
                "type": "complete",
                "answer": full_answer,
                "confidence_score": confidence_score,
                "sources": sources,
                "retrieved_documents": [
                    {
                        "content": result.document.page_content[:200] + "...",
                        "score": result.score,
                        "source": result.document.metadata.get("source", "未知"),
                    }
                    for result in retrieved_results[:3]  # 只返回前3个
                ],
            }

        except Exception as e:
            self.logger.error(f"流式处理失败: {e}")
            yield {"type": "error", "error": str(e)}

    def batch_process_queries(self, queries: List[str], **kwargs) -> List[RAGResponse]:
        """批量处理查询

        Args:
            queries: 查询列表
            **kwargs: 额外参数

        Returns:
            响应列表
        """
        try:
            self.logger.info(f"开始批量处理 {len(queries)} 个查询")

            responses = []
            for i, query in enumerate(queries):
                try:
                    response = self.process_query(query, **kwargs)
                    responses.append(response)

                    if (i + 1) % 10 == 0:
                        self.logger.info(f"已处理 {i + 1}/{len(queries)} 个查询")

                except Exception as e:
                    self.logger.error(f"查询 {i+1} 处理失败: {e}")
                    # 创建错误响应
                    error_response = RAGResponse(
                        query=query,
                        answer=f"处理查询时出现错误: {str(e)}",
                        retrieved_documents=[],
                        retrieval_time=0.0,
                        generation_time=0.0,
                        total_time=0.0,
                        confidence_score=0.0,
                        sources=[],
                        metadata={"error": True},
                    )
                    responses.append(error_response)

            self.logger.info(f"批量处理完成，成功处理 {len(responses)} 个查询")
            return responses

        except Exception as e:
            self.logger.error(f"批量处理失败: {e}")
            raise

    def get_chain_stats(self) -> Dict[str, Any]:
        """获取链统计信息"""
        return {
            "retriever_stats": self.retriever.get_retrieval_stats(),
            "llm_info": self.llm.get_model_info(),
            "embedding_info": self.embedding_manager.get_model_info(),
            "vector_store_stats": self.vector_store.get_stats(),
        }

    def save_response(self, response: RAGResponse, filepath: str) -> None:
        """保存响应到文件

        Args:
            response: RAG响应对象
            filepath: 保存路径
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(asdict(response), f, ensure_ascii=False, indent=2)
            self.logger.info(f"响应已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"保存响应失败: {e}")
            raise
