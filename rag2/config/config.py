"""
RAG2项目主配置文件
统一管理所有配置项
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .environment_config import get_environment_config, get_model_config_dict

@dataclass
class DatabaseConfig:
    """数据库配置"""
    # PostgreSQL配置
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag2_db"
    postgres_user: str = "rag2_user"
    postgres_password: str = "rag2_password"
    
    # MySQL配置
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_db: str = "rag2_dialogue"
    mysql_user: str = "rag2_user"
    mysql_password: str = "rag2_password"
    
    # Redis配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "rag2_redis_password"
    redis_db: int = 0
    
    # Elasticsearch配置
    es_host: str = "localhost"
    es_port: int = 9200
    es_index_prefix: str = "rag2"
    
    # Neo4j配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "rag2_password"

@dataclass
class APIConfig:
    """API服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    log_level: str = "info"
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]

@dataclass
class RAGConfig:
    """RAG系统配置"""
    # 检索配置
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    similarity_threshold: float = 0.7
    
    # 上下文配置
    max_context_length: int = 4000
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # 会话配置
    max_conversation_turns: int = 20
    session_timeout_hours: int = 24
    
    # 压缩配置
    enable_context_compression: bool = True
    compression_ratio: float = 0.7
    
    # 自适应RAG配置
    enable_adaptive_rag: bool = True
    enable_self_rag: bool = True
    confidence_threshold: float = 0.8

class Config:
    """主配置类"""
    
    def __init__(self):
        self.environment = os.getenv("RAG_ENV", "development")
        self.debug = self.environment == "development"
        
        # 加载各模块配置
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.rag = RAGConfig()
        
        # 加载模型配置
        self.models = get_model_config_dict()
        
        # 加载环境变量覆盖
        self._load_env_overrides()
    
    def _load_env_overrides(self):
        """从环境变量加载配置覆盖"""
        # 数据库配置覆盖
        if os.getenv("POSTGRES_HOST"):
            self.database.postgres_host = os.getenv("POSTGRES_HOST")
        if os.getenv("POSTGRES_PORT"):
            self.database.postgres_port = int(os.getenv("POSTGRES_PORT"))
        if os.getenv("POSTGRES_PASSWORD"):
            self.database.postgres_password = os.getenv("POSTGRES_PASSWORD")
            
        if os.getenv("MYSQL_HOST"):
            self.database.mysql_host = os.getenv("MYSQL_HOST")
        if os.getenv("MYSQL_PORT"):
            self.database.mysql_port = int(os.getenv("MYSQL_PORT"))
        if os.getenv("MYSQL_PASSWORD"):
            self.database.mysql_password = os.getenv("MYSQL_PASSWORD")
            
        if os.getenv("REDIS_HOST"):
            self.database.redis_host = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            self.database.redis_port = int(os.getenv("REDIS_PORT"))
        if os.getenv("REDIS_PASSWORD"):
            self.database.redis_password = os.getenv("REDIS_PASSWORD")
            
        if os.getenv("ES_HOST"):
            self.database.es_host = os.getenv("ES_HOST")
        if os.getenv("ES_PORT"):
            self.database.es_port = int(os.getenv("ES_PORT"))
            
        if os.getenv("NEO4J_URI"):
            self.database.neo4j_uri = os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_PASSWORD"):
            self.database.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        # API配置覆盖
        if os.getenv("API_HOST"):
            self.api.host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            self.api.port = int(os.getenv("API_PORT"))
        if os.getenv("API_DEBUG"):
            self.api.debug = os.getenv("API_DEBUG").lower() == "true"
        if os.getenv("API_WORKERS"):
            self.api.workers = int(os.getenv("API_WORKERS"))
    
    def get_postgres_url(self) -> str:
        """获取PostgreSQL连接URL"""
        return f"postgresql://{self.database.postgres_user}:{self.database.postgres_password}@{self.database.postgres_host}:{self.database.postgres_port}/{self.database.postgres_db}"
    
    def get_mysql_url(self) -> str:
        """获取MySQL连接URL"""
        return f"mysql+pymysql://{self.database.mysql_user}:{self.database.mysql_password}@{self.database.mysql_host}:{self.database.mysql_port}/{self.database.mysql_db}"
    
    def get_redis_url(self) -> str:
        """获取Redis连接URL"""
        return f"redis://:{self.database.redis_password}@{self.database.redis_host}:{self.database.redis_port}/{self.database.redis_db}"
    
    def get_es_url(self) -> str:
        """获取Elasticsearch连接URL"""
        return f"http://{self.database.es_host}:{self.database.es_port}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "database": {
                "postgres_url": self.get_postgres_url(),
                "mysql_url": self.get_mysql_url(),
                "redis_url": self.get_redis_url(),
                "es_url": self.get_es_url(),
                "neo4j_uri": self.database.neo4j_uri,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "debug": self.api.debug,
                "workers": self.api.workers,
            },
            "rag": {
                "retrieval_top_k": self.rag.retrieval_top_k,
                "rerank_top_k": self.rag.rerank_top_k,
                "max_context_length": self.rag.max_context_length,
                "chunk_size": self.rag.chunk_size,
            },
            "models": self.models
        }

# 全局配置实例
config = Config()

# 便捷访问函数
def get_config() -> Config:
    """获取配置实例"""
    return config

def get_model_config() -> Dict[str, Any]:
    """获取模型配置"""
    return config.models

def get_database_config() -> DatabaseConfig:
    """获取数据库配置"""
    return config.database

def get_api_config() -> APIConfig:
    """获取API配置"""
    return config.api

def get_rag_config() -> RAGConfig:
    """获取RAG配置"""
    return config.rag

def is_development() -> bool:
    """是否为开发环境"""
    return config.environment == "development"

def is_production() -> bool:
    """是否为生产环境"""
    return config.environment == "production"
