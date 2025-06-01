"""
向量存储模块 - 简化版

该模块负责ChromaDB向量数据库的管理。
提供文档的向量化存储、相似度检索和索引管理功能。

数据流：
1. 文档输入 -> 向量化 -> 存储到ChromaDB
2. 查询输入 -> 向量化 -> 相似度检索 -> 返回相关文档

使用ChromaDB作为唯一的向量存储后端，底层使用SQLite持久化。
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid

# ChromaDB导入
import chromadb
from chromadb.config import Settings

from langchain_core.documents import Document

from config import defaultConfig
from logger import get_logger, log_execution_time
import embeddings


class VectorStoreManager:
    """ChromaDB向量存储管理器

    专门管理ChromaDB向量数据库，提供文档存储和检索功能。

    Attributes:
        client: ChromaDB客户端
        collection: ChromaDB集合
        embedding_manager: 嵌入管理器
    """

    def __init__(self, embedding_manager: embeddings.EmbeddingManager):
        """初始化向量存储管理器

        Args:
            embedding_manager: 嵌入管理器实例
        """
        self.logger = get_logger("VectorStoreManager")
        self.embedding_manager = embedding_manager
        self.client: chromadb.Client
        self.collection: chromadb.Collection

        self._initialize_chromadb()

    def _initialize_chromadb(self) -> None:
        """初始化ChromaDB存储"""
        try:
            self.logger.info("正在初始化ChromaDB存储")

            # 创建持久化目录
            persist_dir = Path(defaultConfig.vector_store.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)

            # 初始化ChromaDB客户端
            client = chromadb.PersistentClient(path=str(persist_dir),
                                               settings=Settings(anonymized_telemetry=False, allow_reset=True))

            # 获取或创建集合
            collection = client.get_or_create_collection(
                name=defaultConfig.vector_store.collection_name, metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )

            self.logger.info(
                f"ChromaDB初始化成功 | 集合: {defaultConfig.vector_store.collection_name} | " f"文档数量: {collection.count()}")

            self.client = client
            self.collection = collection


        except Exception as e:
            self.logger.error(f"ChromaDB初始化失败: {e}")
            raise

    @log_execution_time("add_documents")
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """添加文档到向量存储

        Args:
            documents: 文档列表
            embeddings: 预计算的嵌入向量，如果为None则自动计算

        Returns:
            文档ID列表
        """
        if not documents:
            return []

        try:
            self.logger.info(f"开始添加 {len(documents)} 个文档到ChromaDB")

            # 计算嵌入向量
            if embeddings is None:
                texts = [doc.page_content for doc in documents]
                embeddings = self.embedding_manager.embed_documents(texts)

            # 生成文档ID
            doc_ids = [str(uuid.uuid4()) for _ in documents]

            # 准备数据
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # 添加到ChromaDB
            self.collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=doc_ids)

            self.logger.info(f"成功添加 {len(documents)} 个文档到ChromaDB")
            return doc_ids

        except Exception as e:
            self.logger.error(f"添加文档失败: {e}")
            raise

    @log_execution_time("similarity_search")
    def similarity_search(self, query: str, k: int = None, score_threshold: float = None) -> List[
        Tuple[Document, float]]:
        """相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值

        Returns:
            (文档, 相似度分数) 元组列表，按相似度降序排列
        """
        k = k or defaultConfig.vector_store.top_k
        score_threshold = score_threshold or defaultConfig.vector_store.score_threshold

        try:
            self.logger.info(f"开始相似度搜索: {query[:50]}...")

            # 计算查询向量
            query_embedding = self.embedding_manager.embed_query(query)

            # 在ChromaDB中搜索
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=k, include=["documents", "metadatas", "distances"]
            )

            documents_with_scores = []
            for i in range(len(results["documents"][0])):
                # ChromaDB返回的是距离，需要转换为相似度
                distance = results["distances"][0][i]
                similarity = 1 - distance  # 余弦距离转相似度

                if similarity >= score_threshold:
                    doc = Document(page_content=results["documents"][0][i], metadata=results["metadatas"][0][i] or {})
                    documents_with_scores.append((doc, similarity))

            self.logger.debug(f"搜索完成，返回 {len(documents_with_scores)} 个结果")
            return documents_with_scores

        except Exception as e:
            self.logger.error(f"相似度搜索失败: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """获取向量存储统计信息"""
        try:
            count = self.collection.count()

            return {
                "store_type": "chromadb",
                "document_count": count,
                "embedding_dim": self.embedding_manager.embedding_dim,
                "collection_name": defaultConfig.vector_store.collection_name,
                "persist_directory": defaultConfig.vector_store.persist_directory,
            }
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {
                "store_type": "chromadb",
                "document_count": 0,
                "embedding_dim": 0,
                "collection_name": defaultConfig.vector_store.collection_name,
                "error": str(e),
            }

    def clear(self) -> None:
        """清空向量存储"""
        try:
            self.logger.info("正在清空ChromaDB向量存储")

            # 删除现有集合
            self.client.delete_collection(defaultConfig.vector_store.collection_name)

            # 重新创建集合
            self.collection = self.client.get_or_create_collection(name=defaultConfig.vector_store.collection_name,
                                                                   metadata={"hnsw:space": "cosine"})

            self.logger.info("ChromaDB向量存储已清空")

        except Exception as e:
            self.logger.error(f"清空向量存储失败: {e}")
            raise

    def delete_documents(self, doc_ids: List[str]) -> None:
        """删除指定文档

        Args:
            doc_ids: 要删除的文档ID列表
        """
        try:
            self.collection.delete(ids=doc_ids)
            self.logger.info(f"成功删除 {len(doc_ids)} 个文档")
        except Exception as e:
            self.logger.error(f"删除文档失败: {e}")
            raise

    def update_documents(self, doc_ids: List[str], documents: List[Document],
                         embeddings: Optional[List[List[float]]] = None) -> None:
        """更新指定文档

        Args:
            doc_ids: 要更新的文档ID列表
            documents: 新的文档内容
            embeddings: 新的嵌入向量
        """
        try:
            # 计算嵌入向量
            if embeddings is None:
                texts = [doc.page_content for doc in documents]
                embeddings = self.embedding_manager.embed_documents(texts)

            # 准备数据
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # 更新ChromaDB中的文档
            self.collection.update(ids=doc_ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

            self.logger.info(f"成功更新 {len(doc_ids)} 个文档")

        except Exception as e:
            self.logger.error(f"更新文档失败: {e}")
            raise
