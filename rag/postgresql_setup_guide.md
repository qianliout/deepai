# PostgreSQL向量数据库设置指南

本指南将帮助您在RAG项目中配置和使用PostgreSQL作为向量数据库。

## 前置要求

### 1. 安装PostgreSQL
```bash
# macOS (使用Homebrew)
brew install postgresql@15
brew services start postgresql@15

# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-15 postgresql-contrib-15

# CentOS/RHEL
sudo yum install postgresql15-server postgresql15-contrib
sudo postgresql-15-setup initdb
sudo systemctl start postgresql-15
```

### 2. 安装pgvector扩展
```bash
# macOS
brew install pgvector

# Ubuntu/Debian
sudo apt install postgresql-15-pgvector

# 从源码编译 (如果包管理器没有)
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 3. 安装Python依赖
```bash
# 激活您的conda环境
conda activate aideep2

# 安装PostgreSQL相关依赖
pip install psycopg2-binary pgvector
```

## 数据库配置

### 1. 创建数据库和用户
```sql
-- 以postgres用户身份连接
sudo -u postgres psql

-- 创建数据库
CREATE DATABASE rag_vectordb;

-- 创建用户 (可选)
CREATE USER rag_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE rag_vectordb TO rag_user;

-- 退出
\q
```

### 2. 初始化数据库结构
```bash
# 执行初始化脚本
psql -U postgres -d rag_vectordb -f init_postgresql.sql
```

## 项目配置

### 1. 更新config.py配置
在您的配置文件中设置PostgreSQL参数：

```python
# 在config.py中或环境变量中设置
vector_store:
  backend: "postgresql"  # 改为postgresql
  collection_name: "knowledge_base"
  top_k: 5
  score_threshold: 0.3

postgresql:
  host: "localhost"
  port: 5432
  username: "postgres"  # 或您创建的用户
  password: "your_password"
  database: "rag_vectordb"
  table_name: "documents"
  vector_dimension: 512  # 根据您的嵌入模型调整
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
```

### 2. 环境变量配置 (推荐)
创建`.env`文件：
```bash
# PostgreSQL配置
POSTGRESQL_HOST=localhost
POSTGRESQL_PORT=5432
POSTGRESQL_USERNAME=postgres
POSTGRESQL_PASSWORD=your_password
POSTGRESQL_DATABASE=rag_vectordb
VECTOR_STORE_BACKEND=postgresql
```

## 使用示例

### 1. 基本使用
```python
from rag_chain import RAGChain
from config import defaultConfig

# 确保配置中backend设置为postgresql
defaultConfig.vector_store.backend = "postgresql"

# 初始化RAG系统
rag = RAGChain()

# 添加文档
documents = [
    Document(page_content="人工智能是计算机科学的一个分支", 
             metadata={"source": "ai_book.pdf"}),
    Document(page_content="机器学习是人工智能的重要组成部分", 
             metadata={"source": "ml_book.pdf"})
]

rag.add_documents(documents)

# 查询
answer = rag.query("什么是人工智能？")
print(answer)
```

### 2. 直接使用PostgreSQL向量存储
```python
from embeddings import EmbeddingManager
from postgresql_vector_store import PostgreSQLVectorStoreManager

# 初始化组件
embedding_manager = EmbeddingManager()
vector_store = PostgreSQLVectorStoreManager(embedding_manager)

# 添加文档
doc_ids = vector_store.add_documents(documents)

# 相似度搜索
results = vector_store.similarity_search(
    "人工智能应用", 
    k=5, 
    score_threshold=0.7
)

for doc, score in results:
    print(f"相似度: {score:.3f}")
    print(f"内容: {doc.page_content[:100]}...")
    print(f"元数据: {doc.metadata}")
    print("-" * 50)
```

### 3. 获取统计信息
```python
# 获取向量存储统计
stats = vector_store.get_stats()
print(f"文档数量: {stats['document_count']}")
print(f"表大小: {stats['table_size']}")
print(f"向量维度: {stats['vector_dimension']}")
```

## 性能优化

### 1. 索引优化
```sql
-- 调整IVFFlat索引的lists参数
-- lists = sqrt(行数) 通常是一个好的起点
DROP INDEX IF EXISTS documents_embedding_idx;
CREATE INDEX documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1000);  -- 根据数据量调整

-- 对于大数据集，考虑使用HNSW索引 (需要pgvector 0.5.0+)
CREATE INDEX documents_embedding_hnsw_idx 
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### 2. 连接池配置
```python
# 在config.py中调整连接池参数
postgresql:
  pool_size: 20        # 增加连接池大小
  max_overflow: 40     # 增加最大溢出连接
  pool_timeout: 60     # 增加超时时间
```

### 3. 批量操作
```python
# 批量添加文档以提高性能
batch_size = 100
for i in range(0, len(large_document_list), batch_size):
    batch = large_document_list[i:i + batch_size]
    vector_store.add_documents(batch)
```

## 监控和维护

### 1. 监控查询
```sql
-- 查看慢查询
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
WHERE query LIKE '%documents%' 
ORDER BY mean_exec_time DESC;

-- 查看索引使用情况
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes 
WHERE tablename = 'documents';
```

### 2. 定期维护
```sql
-- 更新表统计信息
ANALYZE documents;

-- 重建索引 (如果需要)
REINDEX INDEX documents_embedding_idx;

-- 清理旧数据 (使用我们创建的函数)
SELECT cleanup_old_documents(365);  -- 删除365天前的文档
```

## 故障排除

### 1. 常见错误

**错误**: `extension "vector" does not exist`
**解决**: 确保已安装pgvector扩展并在数据库中启用

**错误**: `connection refused`
**解决**: 检查PostgreSQL服务是否运行，端口是否正确

**错误**: `dimension mismatch`
**解决**: 确保配置中的vector_dimension与嵌入模型的维度一致

### 2. 性能问题

**问题**: 搜索速度慢
**解决**: 
- 检查索引是否存在
- 调整索引参数
- 考虑增加内存配置

**问题**: 插入速度慢
**解决**:
- 使用批量插入
- 临时禁用索引，插入完成后重建
- 调整PostgreSQL配置参数

## 从ChromaDB迁移

如果您之前使用ChromaDB，可以使用以下脚本迁移数据：

```python
def migrate_from_chromadb_to_postgresql():
    """从ChromaDB迁移到PostgreSQL"""
    from vector_store import VectorStoreManager
    from postgresql_vector_store import PostgreSQLVectorStoreManager
    from embeddings import EmbeddingManager
    
    embedding_manager = EmbeddingManager()
    
    # 初始化两个存储
    chromadb_store = VectorStoreManager(embedding_manager)
    postgresql_store = PostgreSQLVectorStoreManager(embedding_manager)
    
    # 获取ChromaDB中的所有文档
    # 注意：这需要您实现获取所有文档的方法
    # documents = chromadb_store.get_all_documents()
    
    # 迁移到PostgreSQL
    # postgresql_store.add_documents(documents)
    
    print("迁移完成！")
```

## 总结

PostgreSQL向量数据库为RAG系统提供了企业级的可靠性和性能。通过合理的配置和优化，可以处理大规模的向量数据并提供快速的相似度搜索功能。
