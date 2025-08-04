-- RAG2 MySQL initialization script
-- Create tables for dialogue history and session management

USE rag2_dialogue;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    metadata JSON,
    INDEX idx_username (username),
    INDEX idx_email (email)
);

-- Create sessions table for conversation sessions
CREATE TABLE IF NOT EXISTS sessions (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    user_id VARCHAR(36),
    session_name VARCHAR(255),
    status ENUM('active', 'inactive', 'archived') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_last_activity (last_activity)
);

-- Create conversations table for storing dialogue history
CREATE TABLE IF NOT EXISTS conversations (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    session_id VARCHAR(36) NOT NULL,
    message_type ENUM('user', 'assistant', 'system') NOT NULL,
    content TEXT NOT NULL,
    role VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    token_count INT DEFAULT 0,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    INDEX idx_session_id (session_id),
    INDEX idx_message_type (message_type),
    INDEX idx_timestamp (timestamp)
);

-- Create query_logs table for tracking user queries
CREATE TABLE IF NOT EXISTS query_logs (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    session_id VARCHAR(36),
    user_query TEXT NOT NULL,
    processed_query TEXT,
    query_type VARCHAR(50),
    intent VARCHAR(100),
    entities JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL,
    INDEX idx_session_id (session_id),
    INDEX idx_query_type (query_type),
    INDEX idx_timestamp (timestamp),
    INDEX idx_success (success)
);

-- Create retrieval_logs table for tracking retrieval results
CREATE TABLE IF NOT EXISTS retrieval_logs (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    query_log_id VARCHAR(36),
    retrieval_method VARCHAR(50),
    query_vector_id VARCHAR(36),
    retrieved_documents JSON,
    similarity_scores JSON,
    rerank_scores JSON,
    final_context TEXT,
    retrieval_time_ms INT,
    document_count INT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (query_log_id) REFERENCES query_logs(id) ON DELETE CASCADE,
    INDEX idx_query_log_id (query_log_id),
    INDEX idx_retrieval_method (retrieval_method),
    INDEX idx_timestamp (timestamp)
);

-- Create feedback table for user feedback
CREATE TABLE IF NOT EXISTS feedback (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    session_id VARCHAR(36),
    query_log_id VARCHAR(36),
    conversation_id VARCHAR(36),
    feedback_type ENUM('thumbs_up', 'thumbs_down', 'rating', 'comment') NOT NULL,
    rating INT CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL,
    FOREIGN KEY (query_log_id) REFERENCES query_logs(id) ON DELETE SET NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE SET NULL,
    INDEX idx_session_id (session_id),
    INDEX idx_query_log_id (query_log_id),
    INDEX idx_feedback_type (feedback_type),
    INDEX idx_timestamp (timestamp)
);

-- Create system_metrics table for performance monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    metric_unit VARCHAR(20),
    component VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    INDEX idx_metric_name (metric_name),
    INDEX idx_component (component),
    INDEX idx_timestamp (timestamp)
);

-- Create configuration table for system settings
CREATE TABLE IF NOT EXISTS configurations (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value TEXT,
    config_type ENUM('string', 'integer', 'float', 'boolean', 'json') DEFAULT 'string',
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    metadata JSON,
    INDEX idx_config_key (config_key)
);

-- Insert default configurations
INSERT IGNORE INTO configurations (config_key, config_value, config_type, description) VALUES
('max_conversation_history', '20', 'integer', 'Maximum number of conversation turns to keep in context'),
('context_window_size', '4000', 'integer', 'Maximum context window size in tokens'),
('retrieval_top_k', '10', 'integer', 'Number of documents to retrieve initially'),
('rerank_top_k', '5', 'integer', 'Number of documents after reranking'),
('similarity_threshold', '0.7', 'float', 'Minimum similarity threshold for document retrieval'),
('session_timeout_hours', '24', 'integer', 'Session timeout in hours'),
('enable_feedback', 'true', 'boolean', 'Enable user feedback collection'),
('log_retention_days', '30', 'integer', 'Number of days to retain logs');

-- Create views for common queries
CREATE VIEW conversation_summary AS
SELECT 
    s.id as session_id,
    s.session_name,
    u.username,
    COUNT(c.id) as message_count,
    MAX(c.timestamp) as last_message,
    s.status,
    s.created_at as session_created
FROM sessions s
LEFT JOIN users u ON s.user_id = u.id
LEFT JOIN conversations c ON s.id = c.session_id
GROUP BY s.id, s.session_name, u.username, s.status, s.created_at;

CREATE VIEW query_performance AS
SELECT 
    DATE(ql.timestamp) as query_date,
    ql.query_type,
    COUNT(*) as query_count,
    AVG(ql.processing_time_ms) as avg_processing_time,
    AVG(rl.retrieval_time_ms) as avg_retrieval_time,
    SUM(CASE WHEN ql.success = TRUE THEN 1 ELSE 0 END) as success_count,
    SUM(CASE WHEN ql.success = FALSE THEN 1 ELSE 0 END) as error_count
FROM query_logs ql
LEFT JOIN retrieval_logs rl ON ql.id = rl.query_log_id
GROUP BY DATE(ql.timestamp), ql.query_type;

-- Grant permissions to rag2_user
GRANT ALL PRIVILEGES ON rag2_dialogue.* TO 'rag2_user'@'%';
FLUSH PRIVILEGES;
