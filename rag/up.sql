-- RAG系统MySQL数据库建表语句
-- 创建时间: 2024年
-- 说明: 该文件包含RAG系统所需的所有MySQL表结构

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS `rag_system`
DEFAULT CHARACTER SET utf8mb4
DEFAULT COLLATE utf8mb4_unicode_ci;

-- 使用数据库
USE `rag_system`;

-- ============================================================================
-- 会话管理表
-- ============================================================================

-- 会话记录表
CREATE TABLE IF NOT EXISTS `sessions` (
    `id` VARCHAR(36) NOT NULL COMMENT '会话ID，使用UUID',
    `user_id` VARCHAR(36) NULL COMMENT '用户ID',
    `title` VARCHAR(255) NULL COMMENT '会话标题',
    `status` VARCHAR(20) NOT NULL DEFAULT 'active' COMMENT '会话状态: active, archived, deleted',
    `metadata` JSON NULL COMMENT '会话元数据，存储额外信息',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `message_count` INT NOT NULL DEFAULT 0 COMMENT '消息数量',
    `total_tokens` INT NOT NULL DEFAULT 0 COMMENT '总Token数量',
    PRIMARY KEY (`id`),
    INDEX `idx_user_id` (`user_id`),
    INDEX `idx_status` (`status`),
    INDEX `idx_created_at` (`created_at`),
    INDEX `idx_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='会话记录表';

-- ============================================================================
-- 对话管理表
-- ============================================================================

-- 对话记录表
CREATE TABLE IF NOT EXISTS `conversations` (
    `id` VARCHAR(36) NOT NULL COMMENT '对话记录ID，使用UUID',
    `session_id` VARCHAR(36) NOT NULL COMMENT '会话ID，关联sessions表',
    `user_id` VARCHAR(36) NULL COMMENT '用户ID',
    `role` VARCHAR(20) NOT NULL COMMENT '消息角色: user, assistant, system',
    `content` TEXT NOT NULL COMMENT '消息内容',
    `metadata` JSON NULL COMMENT '消息元数据，存储额外信息',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `token_count` INT NULL COMMENT '消息Token数量',
    `processing_time` FLOAT NULL COMMENT '处理时间（秒）',
    PRIMARY KEY (`id`),
    INDEX `idx_session_id` (`session_id`),
    INDEX `idx_user_id` (`user_id`),
    INDEX `idx_role` (`role`),
    INDEX `idx_created_at` (`created_at`),
    FOREIGN KEY (`session_id`) REFERENCES `sessions`(`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='对话记录表';

-- ============================================================================
-- 文档管理表
-- ============================================================================

-- 文档记录表
CREATE TABLE IF NOT EXISTS `documents` (
    `id` VARCHAR(36) NOT NULL COMMENT '文档ID，使用UUID',
    `title` VARCHAR(500) NOT NULL COMMENT '文档标题',
    `content` LONGTEXT NOT NULL COMMENT '文档内容',
    `source` VARCHAR(255) NOT NULL COMMENT '文档来源',
    `doc_type` VARCHAR(50) NOT NULL COMMENT '文档类型: pdf, txt, md, html等',
    `file_path` VARCHAR(1000) NULL COMMENT '文件路径',
    `file_size` BIGINT NULL COMMENT '文件大小（字节）',
    `content_length` INT NOT NULL COMMENT '内容长度（字符数）',
    `keywords` JSON NULL COMMENT '关键词列表',
    `metadata` JSON NULL COMMENT '文档元数据',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `indexed_at` DATETIME NULL COMMENT '索引时间',
    `status` VARCHAR(20) NOT NULL DEFAULT 'active' COMMENT '文档状态: active, deleted, processing',
    PRIMARY KEY (`id`),
    INDEX `idx_source` (`source`),
    INDEX `idx_doc_type` (`doc_type`),
    INDEX `idx_status` (`status`),
    INDEX `idx_created_at` (`created_at`),
    INDEX `idx_indexed_at` (`indexed_at`),
    FULLTEXT INDEX `ft_title_content` (`title`, `content`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='文档记录表';

-- ============================================================================
-- 检索统计表
-- ============================================================================

-- 检索日志表
CREATE TABLE IF NOT EXISTS `retrieval_logs` (
    `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '日志ID',
    `session_id` VARCHAR(36) NOT NULL COMMENT '会话ID',
    `user_id` VARCHAR(36) NULL COMMENT '用户ID',
    `query` TEXT NOT NULL COMMENT '查询内容',
    `expanded_query` TEXT NULL COMMENT '扩展后的查询',
    `retrieval_method` VARCHAR(50) NOT NULL COMMENT '检索方法: vector_only, hybrid, es_only',
    `es_results_count` INT NULL COMMENT 'ES检索结果数量',
    `vector_results_count` INT NULL COMMENT '向量检索结果数量',
    `final_results_count` INT NOT NULL COMMENT '最终结果数量',
    `processing_time` FLOAT NOT NULL COMMENT '处理时间（秒）',
    `es_time` FLOAT NULL COMMENT 'ES检索时间（秒）',
    `vector_time` FLOAT NULL COMMENT '向量检索时间（秒）',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    INDEX `idx_session_id` (`session_id`),
    INDEX `idx_user_id` (`user_id`),
    INDEX `idx_retrieval_method` (`retrieval_method`),
    INDEX `idx_created_at` (`created_at`),
    FOREIGN KEY (`session_id`) REFERENCES `sessions`(`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='检索日志表';

-- ============================================================================
-- 系统监控表
-- ============================================================================

-- 系统状态监控表
CREATE TABLE IF NOT EXISTS `system_status` (
    `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '状态ID',
    `component` VARCHAR(50) NOT NULL COMMENT '组件名称: redis, mysql, elasticsearch, vector_store',
    `status` VARCHAR(20) NOT NULL COMMENT '状态: healthy, warning, error',
    `message` TEXT NULL COMMENT '状态消息',
    `metrics` JSON NULL COMMENT '监控指标',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    INDEX `idx_component` (`component`),
    INDEX `idx_status` (`status`),
    INDEX `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统状态监控表';

-- ============================================================================
-- 用户管理表（可选）
-- ============================================================================

-- 用户表
CREATE TABLE IF NOT EXISTS `users` (
    `id` VARCHAR(36) NOT NULL COMMENT '用户ID，使用UUID',
    `username` VARCHAR(100) NOT NULL COMMENT '用户名',
    `email` VARCHAR(255) NULL COMMENT '邮箱',
    `password_hash` VARCHAR(255) NULL COMMENT '密码哈希',
    `display_name` VARCHAR(100) NULL COMMENT '显示名称',
    `avatar_url` VARCHAR(500) NULL COMMENT '头像URL',
    `status` VARCHAR(20) NOT NULL DEFAULT 'active' COMMENT '用户状态: active, inactive, banned',
    `metadata` JSON NULL COMMENT '用户元数据',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    `last_login_at` DATETIME NULL COMMENT '最后登录时间',
    PRIMARY KEY (`id`),
    UNIQUE INDEX `uk_username` (`username`),
    UNIQUE INDEX `uk_email` (`email`),
    INDEX `idx_status` (`status`),
    INDEX `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- ============================================================================
-- 配置管理表
-- ============================================================================

-- 系统配置表
CREATE TABLE IF NOT EXISTS `system_config` (
    `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '配置ID',
    `config_key` VARCHAR(100) NOT NULL COMMENT '配置键',
    `config_value` TEXT NOT NULL COMMENT '配置值',
    `config_type` VARCHAR(20) NOT NULL DEFAULT 'string' COMMENT '配置类型: string, int, float, bool, json',
    `description` VARCHAR(500) NULL COMMENT '配置描述',
    `is_public` BOOLEAN NOT NULL DEFAULT FALSE COMMENT '是否公开配置',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`),
    UNIQUE INDEX `uk_config_key` (`config_key`),
    INDEX `idx_is_public` (`is_public`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统配置表';

-- ============================================================================
-- 初始化数据
-- ============================================================================

-- 插入默认系统配置
INSERT INTO `system_config` (`config_key`, `config_value`, `config_type`, `description`, `is_public`) VALUES
('system_name', 'RAG智能问答系统', 'string', '系统名称', TRUE),
('version', '1.0.0', 'string', '系统版本', TRUE),
('max_session_duration', '3600', 'int', '最大会话持续时间（秒）', FALSE),
('max_context_length', '4000', 'int', '最大上下文长度（tokens）', FALSE),
('default_retrieval_count', '5', 'int', '默认检索结果数量', FALSE),
('enable_query_expansion', 'true', 'bool', '是否启用查询扩展', FALSE),
('enable_context_compression', 'true', 'bool', '是否启用上下文压缩', FALSE),
('compression_threshold', '0.8', 'float', '上下文压缩阈值', FALSE)
ON DUPLICATE KEY UPDATE
    `config_value` = VALUES(`config_value`),
    `updated_at` = CURRENT_TIMESTAMP;

-- ============================================================================
-- 创建视图
-- ============================================================================

-- 会话统计视图
CREATE OR REPLACE VIEW `v_session_stats` AS
SELECT
    s.id as session_id,
    s.user_id,
    s.title,
    s.status,
    s.created_at,
    s.updated_at,
    s.message_count,
    s.total_tokens,
    COUNT(c.id) as actual_message_count,
    SUM(CASE WHEN c.role = 'user' THEN 1 ELSE 0 END) as user_message_count,
    SUM(CASE WHEN c.role = 'assistant' THEN 1 ELSE 0 END) as assistant_message_count,
    AVG(c.processing_time) as avg_processing_time,
    MAX(c.created_at) as last_message_at
FROM sessions s
LEFT JOIN conversations c ON s.id = c.session_id
GROUP BY s.id, s.user_id, s.title, s.status, s.created_at, s.updated_at, s.message_count, s.total_tokens;

-- 用户活跃度统计视图
CREATE OR REPLACE VIEW `v_user_activity` AS
SELECT
    u.id as user_id,
    u.username,
    u.display_name,
    u.status,
    COUNT(DISTINCT s.id) as session_count,
    COUNT(c.id) as total_messages,
    SUM(CASE WHEN c.role = 'user' THEN 1 ELSE 0 END) as user_messages,
    SUM(CASE WHEN c.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages,
    SUM(s.total_tokens) as total_tokens,
    MAX(c.created_at) as last_activity_at,
    DATEDIFF(CURDATE(), DATE(MAX(c.created_at))) as days_since_last_activity
FROM users u
LEFT JOIN sessions s ON u.id = s.user_id
LEFT JOIN conversations c ON s.id = c.session_id
GROUP BY u.id, u.username, u.display_name, u.status;

-- ============================================================================
-- 创建存储过程
-- ============================================================================

-- 清理过期会话的存储过程
DELIMITER //
CREATE PROCEDURE CleanExpiredSessions(IN days_to_keep INT)
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE session_count INT DEFAULT 0;

    -- 开始事务
    START TRANSACTION;

    -- 删除过期的会话（级联删除相关对话记录）
    DELETE FROM sessions
    WHERE status = 'active'
    AND updated_at < DATE_SUB(NOW(), INTERVAL days_to_keep DAY);

    -- 获取删除的会话数量
    SET session_count = ROW_COUNT();

    -- 提交事务
    COMMIT;

    -- 返回清理结果
    SELECT session_count as cleaned_sessions, NOW() as cleaned_at;

END //
DELIMITER ;

-- ============================================================================
-- 创建触发器
-- ============================================================================

-- 会话消息计数更新触发器
DELIMITER //
CREATE TRIGGER tr_conversations_insert_update_session
AFTER INSERT ON conversations
FOR EACH ROW
BEGIN
    UPDATE sessions
    SET message_count = message_count + 1,
        total_tokens = total_tokens + IFNULL(NEW.token_count, 0),
        updated_at = NOW()
    WHERE id = NEW.session_id;
END //
DELIMITER ;

-- 会话消息删除计数更新触发器
DELIMITER //
CREATE TRIGGER tr_conversations_delete_update_session
AFTER DELETE ON conversations
FOR EACH ROW
BEGIN
    UPDATE sessions
    SET message_count = GREATEST(message_count - 1, 0),
        total_tokens = GREATEST(total_tokens - IFNULL(OLD.token_count, 0), 0),
        updated_at = NOW()
    WHERE id = OLD.session_id;
END //
DELIMITER ;

-- ============================================================================
-- 创建索引优化
-- ============================================================================

-- 为大表创建复合索引
CREATE INDEX `idx_conversations_session_created` ON `conversations` (`session_id`, `created_at`);
CREATE INDEX `idx_conversations_user_role` ON `conversations` (`user_id`, `role`);
CREATE INDEX `idx_retrieval_logs_session_created` ON `retrieval_logs` (`session_id`, `created_at`);

-- ============================================================================
-- 权限设置（可选）
-- ============================================================================

-- 创建RAG系统专用用户（请根据实际情况修改密码）
-- CREATE USER IF NOT EXISTS 'rag_user'@'localhost' IDENTIFIED BY 'your_secure_password';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON rag_system.* TO 'rag_user'@'localhost';
-- FLUSH PRIVILEGES;

-- ============================================================================
-- 完成提示
-- ============================================================================

SELECT 'RAG系统数据库初始化完成！' as message, NOW() as completed_at;