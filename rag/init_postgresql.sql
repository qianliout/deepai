-- PostgreSQL向量数据库初始化脚本
-- 用于创建RAG系统所需的数据库、扩展和表结构

-- 创建数据库 (需要以超级用户身份执行)
-- CREATE DATABASE rag_vectordb OWNER postgres;

-- 连接到目标数据库后执行以下命令
-- \c rag_vectordb;

-- 启用pgvector扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建文档表
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(255) PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(512),  -- 向量维度，根据嵌入模型调整
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建向量索引以提高检索性能
-- 使用IVFFlat索引，适合大规模数据
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 创建元数据索引，支持基于元数据的过滤
CREATE INDEX IF NOT EXISTS documents_metadata_idx 
ON documents USING gin (metadata);

-- 创建内容全文搜索索引
CREATE INDEX IF NOT EXISTS documents_content_idx 
ON documents USING gin (to_tsvector('english', content));

-- 创建时间索引
CREATE INDEX IF NOT EXISTS documents_created_at_idx 
ON documents (created_at);

-- 创建更新时间触发器
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- 创建用于统计的视图
CREATE OR REPLACE VIEW documents_stats AS
SELECT 
    COUNT(*) as total_documents,
    AVG(length(content)) as avg_content_length,
    MIN(created_at) as first_document_date,
    MAX(created_at) as last_document_date,
    pg_size_pretty(pg_total_relation_size('documents')) as table_size
FROM documents;

-- 创建相似度搜索函数
CREATE OR REPLACE FUNCTION similarity_search(
    query_embedding vector(512),
    similarity_threshold float DEFAULT 0.0,
    max_results int DEFAULT 10
)
RETURNS TABLE(
    id VARCHAR(255),
    content TEXT,
    metadata JSONB,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) as similarity
    FROM documents d
    WHERE 1 - (d.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- 创建混合搜索函数（结合全文搜索和向量搜索）
CREATE OR REPLACE FUNCTION hybrid_search(
    search_text TEXT,
    query_embedding vector(512),
    similarity_threshold float DEFAULT 0.0,
    max_results int DEFAULT 10
)
RETURNS TABLE(
    id VARCHAR(255),
    content TEXT,
    metadata JSONB,
    similarity float,
    text_rank float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) as similarity,
        ts_rank(to_tsvector('english', d.content), plainto_tsquery('english', search_text)) as text_rank
    FROM documents d
    WHERE 
        (1 - (d.embedding <=> query_embedding) >= similarity_threshold)
        OR (to_tsvector('english', d.content) @@ plainto_tsquery('english', search_text))
    ORDER BY 
        (1 - (d.embedding <=> query_embedding)) * 0.7 + 
        ts_rank(to_tsvector('english', d.content), plainto_tsquery('english', search_text)) * 0.3 DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- 创建清理旧文档的函数
CREATE OR REPLACE FUNCTION cleanup_old_documents(days_old int DEFAULT 365)
RETURNS int AS $$
DECLARE
    deleted_count int;
BEGIN
    DELETE FROM documents 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '%s days' % days_old;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- 授权给应用用户（如果需要）
-- GRANT ALL PRIVILEGES ON documents TO your_app_user;
-- GRANT USAGE ON SCHEMA public TO your_app_user;

-- 显示初始化完成信息
DO $$
BEGIN
    RAISE NOTICE 'PostgreSQL向量数据库初始化完成！';
    RAISE NOTICE '表名: documents';
    RAISE NOTICE '向量维度: 512';
    RAISE NOTICE '索引: embedding (IVFFlat), metadata (GIN), content (GIN)';
END $$;
