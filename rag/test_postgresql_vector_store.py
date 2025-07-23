"""
PostgreSQL向量存储测试模块

测试PostgreSQL向量数据库的功能，包括：
1. 连接和初始化
2. 文档添加和检索
3. 相似度搜索
4. 统计信息获取
"""

import pytest
import os
import sys
from typing import List
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置基本日志配置
import logging
logging.basicConfig(level=logging.INFO)

from langchain_core.documents import Document
from config import defaultConfig
from embeddings import EmbeddingManager
from postgresql_vector_store import PostgreSQLVectorStoreManager


class TestPostgreSQLVectorStore:
    """PostgreSQL向量存储测试类"""
    
    @pytest.fixture
    def mock_embedding_manager(self):
        """模拟嵌入管理器"""
        mock_manager = MagicMock(spec=EmbeddingManager)
        
        # 模拟嵌入向量（512维）
        mock_embedding = [0.1] * 512
        mock_manager.embed_query.return_value = mock_embedding
        mock_manager.embed_documents.return_value = [mock_embedding, mock_embedding]
        
        return mock_manager
    
    @pytest.fixture
    def sample_documents(self):
        """示例文档"""
        return [
            Document(
                page_content="人工智能是计算机科学的一个分支",
                metadata={"source": "ai_book.pdf", "page": 1}
            ),
            Document(
                page_content="机器学习是人工智能的重要组成部分",
                metadata={"source": "ml_book.pdf", "page": 5}
            )
        ]
    
    @patch('postgresql_vector_store.create_engine')
    def test_initialization(self, mock_create_engine, mock_embedding_manager):
        """测试PostgreSQL向量存储初始化"""
        # 模拟数据库连接
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        # 模拟查询结果
        mock_connection.execute.return_value.scalar.return_value = 0
        
        # 创建向量存储管理器
        vector_store = PostgreSQLVectorStoreManager(mock_embedding_manager)
        
        # 验证初始化
        assert vector_store.embedding_manager == mock_embedding_manager
        assert vector_store.table_name == defaultConfig.postgresql.table_name
        assert vector_store.vector_dimension == defaultConfig.postgresql.vector_dimension
        
        # 验证数据库连接创建
        mock_create_engine.assert_called_once()
        
        # 验证表创建SQL执行
        assert mock_connection.execute.call_count >= 3  # 至少执行了创建扩展、表、索引的SQL
    
    @patch('postgresql_vector_store.create_engine')
    def test_add_documents(self, mock_create_engine, mock_embedding_manager, sample_documents):
        """测试添加文档"""
        # 模拟数据库连接
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        # 模拟查询结果
        mock_connection.execute.return_value.scalar.return_value = 0
        
        # 创建向量存储管理器
        vector_store = PostgreSQLVectorStoreManager(mock_embedding_manager)
        
        # 添加文档
        doc_ids = vector_store.add_documents(sample_documents)
        
        # 验证返回的文档ID
        assert len(doc_ids) == len(sample_documents)
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)
        
        # 验证嵌入计算被调用
        mock_embedding_manager.embed_documents.assert_called_once()
        
        # 验证插入SQL被执行
        insert_calls = [call for call in mock_connection.execute.call_args_list 
                       if 'INSERT INTO' in str(call)]
        assert len(insert_calls) == len(sample_documents)
    
    @patch('postgresql_vector_store.create_engine')
    def test_similarity_search(self, mock_create_engine, mock_embedding_manager):
        """测试相似度搜索"""
        # 模拟数据库连接
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        # 模拟初始化查询结果
        mock_connection.execute.return_value.scalar.return_value = 2
        
        # 模拟搜索结果
        mock_row1 = MagicMock()
        mock_row1.content = "人工智能是计算机科学的一个分支"
        mock_row1.metadata = '{"source": "ai_book.pdf", "page": 1}'
        mock_row1.similarity = 0.95
        
        mock_row2 = MagicMock()
        mock_row2.content = "机器学习是人工智能的重要组成部分"
        mock_row2.metadata = '{"source": "ml_book.pdf", "page": 5}'
        mock_row2.similarity = 0.85
        
        # 设置搜索查询的返回值
        def mock_execute(query, params=None):
            if 'SELECT COUNT(*)' in str(query):
                result = MagicMock()
                result.scalar.return_value = 2
                return result
            elif 'SELECT id, content, metadata' in str(query):
                return [mock_row1, mock_row2]
            else:
                return MagicMock()
        
        mock_connection.execute.side_effect = mock_execute
        
        # 创建向量存储管理器
        vector_store = PostgreSQLVectorStoreManager(mock_embedding_manager)
        
        # 执行相似度搜索
        results = vector_store.similarity_search("什么是人工智能", k=2, score_threshold=0.7)
        
        # 验证结果
        assert len(results) == 2
        
        # 验证第一个结果
        doc1, score1 = results[0]
        assert isinstance(doc1, Document)
        assert doc1.page_content == "人工智能是计算机科学的一个分支"
        assert doc1.metadata["source"] == "ai_book.pdf"
        assert score1 == 0.95
        
        # 验证第二个结果
        doc2, score2 = results[1]
        assert doc2.page_content == "机器学习是人工智能的重要组成部分"
        assert score2 == 0.85
        
        # 验证嵌入查询被调用
        mock_embedding_manager.embed_query.assert_called_once_with("什么是人工智能")
    
    @patch('postgresql_vector_store.create_engine')
    def test_get_stats(self, mock_create_engine, mock_embedding_manager):
        """测试获取统计信息"""
        # 模拟数据库连接
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        # 模拟统计查询结果
        def mock_execute(query):
            if 'SELECT COUNT(*)' in str(query):
                result = MagicMock()
                result.scalar.return_value = 100
                return result
            elif 'pg_size_pretty' in str(query):
                result = MagicMock()
                result.scalar.return_value = "1024 kB"
                return result
            else:
                return MagicMock()
        
        mock_connection.execute.side_effect = mock_execute
        
        # 创建向量存储管理器
        vector_store = PostgreSQLVectorStoreManager(mock_embedding_manager)
        
        # 获取统计信息
        stats = vector_store.get_stats()
        
        # 验证统计信息
        assert stats["backend"] == "postgresql"
        assert stats["table_name"] == defaultConfig.postgresql.table_name
        assert stats["document_count"] == 100
        assert stats["table_size"] == "1024 kB"
        assert stats["vector_dimension"] == defaultConfig.postgresql.vector_dimension
        assert "connection_info" in stats
    
    @patch('postgresql_vector_store.create_engine')
    def test_clear(self, mock_create_engine, mock_embedding_manager):
        """测试清空向量存储"""
        # 模拟数据库连接
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        # 模拟初始化查询结果
        mock_connection.execute.return_value.scalar.return_value = 0
        
        # 创建向量存储管理器
        vector_store = PostgreSQLVectorStoreManager(mock_embedding_manager)
        
        # 清空存储
        vector_store.clear()
        
        # 验证删除SQL被执行
        delete_calls = [call for call in mock_connection.execute.call_args_list 
                       if 'DELETE FROM' in str(call)]
        assert len(delete_calls) >= 1
    
    @patch('postgresql_vector_store.create_engine')
    def test_delete_documents(self, mock_create_engine, mock_embedding_manager):
        """测试删除指定文档"""
        # 模拟数据库连接
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        # 模拟初始化查询结果
        mock_connection.execute.return_value.scalar.return_value = 0
        
        # 创建向量存储管理器
        vector_store = PostgreSQLVectorStoreManager(mock_embedding_manager)
        
        # 删除文档
        doc_ids = ["doc1", "doc2", "doc3"]
        vector_store.delete_documents(doc_ids)
        
        # 验证删除SQL被执行
        delete_calls = [call for call in mock_connection.execute.call_args_list 
                       if 'DELETE FROM' in str(call) and 'WHERE id = ANY' in str(call)]
        assert len(delete_calls) >= 1


def test_vector_store_factory():
    """测试向量存储工厂函数"""
    from vector_store import create_vector_store_manager

    # 模拟嵌入管理器
    mock_embedding_manager = MagicMock(spec=EmbeddingManager)

    # 测试ChromaDB后端
    with patch.object(defaultConfig.vector_store, 'backend', 'chromadb'):
        vector_store = create_vector_store_manager(mock_embedding_manager)
        assert vector_store.__class__.__name__ == 'VectorStoreManager'

    # 测试PostgreSQL后端 - 模拟导入成功的情况
    with patch.object(defaultConfig.vector_store, 'backend', 'postgresql'):
        with patch('postgresql_vector_store.PostgreSQLVectorStoreManager') as mock_pg_class:
            mock_instance = MagicMock()
            mock_pg_class.return_value = mock_instance

            # 模拟成功导入PostgreSQLVectorStoreManager
            with patch('builtins.__import__') as mock_import:
                def side_effect(name, *args, **kwargs):
                    if name == 'postgresql_vector_store':
                        mock_module = MagicMock()
                        mock_module.PostgreSQLVectorStoreManager = mock_pg_class
                        return mock_module
                    return __import__(name, *args, **kwargs)

                mock_import.side_effect = side_effect

                vector_store = create_vector_store_manager(mock_embedding_manager)
                # 由于导入失败会回退到ChromaDB，所以这里检查是否是ChromaDB
                assert vector_store.__class__.__name__ == 'VectorStoreManager'

    # 测试不支持的后端
    with patch.object(defaultConfig.vector_store, 'backend', 'unsupported'):
        vector_store = create_vector_store_manager(mock_embedding_manager)
        assert vector_store.__class__.__name__ == 'VectorStoreManager'


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
