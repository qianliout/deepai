"""
RAG个人知识库配置模块
参考BERT项目的简化配置方式，所有参数都在这里定义，运行时不需要手动传参
"""

import os
import logging
from typing import List
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from datetime import datetime


class EmbeddingConfig(BaseModel):
    """文本嵌入配置"""
    
    model_name: str = Field(default="BAAI/bge-small-zh-v1.5", description="嵌入模型名称")
    device: str = Field(default="mps", description="计算设备")
    max_length: int = Field(default=512, description="最大文本长度")
    batch_size: int = Field(default=32, description="批处理大小")
    cache_dir: str = Field(default="/Users/liuqianli/.cache/huggingface/hub", description="模型缓存目录")
     
    class Config:
        extra = "forbid"


class LLMConfig(BaseModel):
    """大语言模型配置"""
    
    api_key: str = Field(default="", description="通义百炼API密钥")
    model_name: str = Field(default="qwen-plus", description="模型名称")
    temperature: float = Field(default=0.7, description="采样温度")
    top_p: float = Field(default=0.9, description="核采样概率")
    max_tokens: int = Field(default=2048, description="最大生成token数")
    timeout: int = Field(default=60, description="请求超时时间")
    max_retries: int = Field(default=3, description="最大重试次数")
    
    class Config:
        extra = "forbid"


class VectorStoreConfig(BaseModel):
    """向量存储配置"""

    # 通用配置
    backend: str = Field(default="chromadb", description="向量数据库后端: chromadb, postgresql")
    collection_name: str = Field(default="knowledge_base", description="集合名称")
    top_k: int = Field(default=5, description="检索返回数量")
    score_threshold: float = Field(default=0.3, description="相似度阈值")

    # ChromaDB配置
    persist_directory: str = Field(default="data/vectorstore", description="ChromaDB数据目录")

    class Config:
        extra = "forbid"


class TextSplitterConfig(BaseModel):
    """文本分割配置"""
    
    chunk_size: int = Field(default=500, description="文本块大小")
    chunk_overlap: int = Field(default=50, description="文本块重叠大小")
    separators: List[str] = Field(
        default=["\n\n", "\n", "。", "！", "？", "；", " ", ""], 
        description="分割符列表"
    )
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v, info):
        chunk_size = info.data.get("chunk_size", 500)
        if v >= chunk_size:
            raise ValueError("chunk_overlap必须小于chunk_size")
        return v
    
    class Config:
        extra = "forbid"


class TokenizerConfig(BaseModel):
    """分词器配置"""
    
    tokenizer_type: str = Field(default="jieba", description="分词器类型")
    remove_stop_words: bool = Field(default=True, description="是否移除停用词")
    
    class Config:
        extra = "forbid"


class PathConfig(BaseModel):
    """路径配置"""
    
    data_dir: str = Field(default="data", description="数据根目录")
    documents_dir: str = Field(default="data/documents", description="文档目录")
    vectorstore_dir: str = Field(default="data/vectorstore", description="向量存储目录")
    log_dir: str = Field(default="/Users/liuqianli/work/python/deepai/logs/rag", description="日志目录")
    
    class Config:
        extra = "forbid"


class LoggingConfig(BaseModel):
    """日志配置"""

    log_level: str = Field(default="INFO", description="日志级别")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日志格式"
    )

    class Config:
        extra = "forbid"


class RedisConfig(BaseModel):
    """Redis配置"""

    host: str = Field(default="localhost", description="Redis主机地址")
    port: int = Field(default=6379, description="Redis端口")
    db: int = Field(default=0, description="Redis数据库编号")
    password: str = Field(default="", description="Redis密码")
    decode_responses: bool = Field(default=True, description="是否解码响应")
    socket_timeout: int = Field(default=5, description="连接超时时间")
    session_expire: int = Field(default=3600, description="会话过期时间(秒)")
    key_prefix: str = Field(default="rag:session:", description="会话键前缀")
    context_window_key: str = Field(default="rag:context:", description="上下文窗口键前缀")
    max_context_length: int = Field(default=4000, description="最大上下文长度(tokens)")

    class Config:
        extra = "forbid"


class ElasticsearchConfig(BaseModel):
    """Elasticsearch配置"""

    host: str = Field(default="localhost", description="ES主机地址")
    port: int = Field(default=9200, description="ES端口")
    username: str = Field(default="", description="ES用户名")
    password: str = Field(default="", description="ES密码")
    use_ssl: bool = Field(default=False, description="是否使用SSL")
    verify_certs: bool = Field(default=False, description="是否验证证书")
    timeout: int = Field(default=30, description="连接超时时间")
    max_retries: int = Field(default=3, description="最大重试次数")
    index_name: str = Field(default="rag_documents", description="文档索引名称")

    class Config:
        extra = "forbid"


class MySQLConfig(BaseModel):
    """MySQL配置"""

    host: str = Field(default="localhost", description="MySQL主机地址")
    port: int = Field(default=3306, description="MySQL端口")
    username: str = Field(default="root", description="MySQL用户名")
    password: str = Field(default="root", description="MySQL密码")
    database: str = Field(default="rag_system", description="数据库名称")
    charset: str = Field(default="utf8mb4", description="字符集")
    autocommit: bool = Field(default=True, description="是否自动提交")
    pool_size: int = Field(default=10, description="连接池大小")
    max_overflow: int = Field(default=20, description="连接池最大溢出")
    pool_timeout: int = Field(default=30, description="连接池超时时间")

    class Config:
        extra = "forbid"


class PostgreSQLConfig(BaseModel):
    """PostgreSQL向量数据库配置"""

    host: str = Field(default="localhost", description="PostgreSQL主机地址")
    port: int = Field(default=5432, description="PostgreSQL端口")
    username: str = Field(default="postgres", description="PostgreSQL用户名")
    password: str = Field(default="postgres", description="PostgreSQL密码")
    database: str = Field(default="rag_vectordb", description="向量数据库名称")
    table_name: str = Field(default="documents", description="文档表名称")
    vector_dimension: int = Field(default=512, description="向量维度")
    pool_size: int = Field(default=10, description="连接池大小")
    max_overflow: int = Field(default=20, description="连接池最大溢出")
    pool_timeout: int = Field(default=30, description="连接池超时时间")

    class Config:
        extra = "forbid"


class RerankerConfig(BaseModel):
    """重排序器配置"""

    # 双编码器配置
    bi_encoder_model: str = Field(default="BAAI/bge-reranker-base", description="双编码器模型名称")
    bi_encoder_enabled: bool = Field(default=True, description="是否启用双编码器")
    bi_encoder_top_k: int = Field(default=20, description="双编码器筛选的候选数量")

    # 交叉编码器配置
    cross_encoder_model: str = Field(default="BAAI/bge-reranker-v2-m3", description="交叉编码器模型名称")
    cross_encoder_enabled: bool = Field(default=True, description="是否启用交叉编码器")
    cross_encoder_top_k: int = Field(default=10, description="交叉编码器最终返回数量")

    # 通用配置
    device: str = Field(default="auto", description="计算设备")
    batch_size: int = Field(default=16, description="批处理大小")
    cache_dir: str = Field(default="/Users/liuqianli/.cache/huggingface/hub", description="模型缓存目录")

    # 分数融合配置
    fusion_method: str = Field(default="weighted", description="分数融合方法: weighted, rrf, max")
    bi_encoder_weight: float = Field(default=0.3, description="双编码器权重")
    cross_encoder_weight: float = Field(default=0.7, description="交叉编码器权重")
    original_weight: float = Field(default=0.2, description="原始检索分数权重")

    class Config:
        extra = "forbid"



def get_device() -> str:
    """自动检测设备"""
    try:
        import torch
        if defaultConfig.embedding.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return defaultConfig.embedding.device
    except ImportError:
        return "cpu"


def setup_logging():
    """设置日志系统"""
    log_dir = Path(defaultConfig.path.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = log_dir / log_filename

    logging.basicConfig(
        level=getattr(logging, defaultConfig.logging.log_level),
        format=defaultConfig.logging.log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filepath, encoding="utf-8"),
        ],
    )

    logger = logging.getLogger("RAG")
    logger.setLevel(getattr(logging, defaultConfig.logging.log_level))
    return logger


def create_directories():
    """创建所有必要的目录"""
    directories = [
        defaultConfig.path.data_dir,
        defaultConfig.path.documents_dir,
        defaultConfig.path.vectorstore_dir,
        defaultConfig.path.log_dir,
        defaultConfig.embedding.cache_dir,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")


def load_env_config():
    """从环境变量加载配置"""
    if api_key := os.getenv("DASHSCOPE_API_KEY"):
        defaultConfig.llm.api_key = api_key

    if device := os.getenv("RAG_DEVICE"):
        defaultConfig.embedding.device = device


def print_config():
    """打印配置信息"""
    print("=" * 50)
    print("RAG个人知识库配置信息")
    print("=" * 50)

    print(f"\n🤖 LLM配置:")
    print(f"  模型: {defaultConfig.llm.model_name}")
    print(f"  温度: {defaultConfig.llm.temperature}")
    print(f"  最大tokens: {defaultConfig.llm.max_tokens}")
    print(f"  API密钥: {'已设置' if defaultConfig.llm.api_key else '未设置'}")

    print(f"\n📊 嵌入配置:")
    print(f"  模型: {defaultConfig.embedding.model_name}")
    print(f"  设备: {get_device()}")
    print(f"  最大长度: {defaultConfig.embedding.max_length}")
    print(f"  批次大小: {defaultConfig.embedding.batch_size}")

    print(f"\n🗂️ 向量存储配置:")
    print(f"  存储目录: {defaultConfig.vector_store.persist_directory}")
    print(f"  集合名称: {defaultConfig.vector_store.collection_name}")
    print(f"  检索数量: {defaultConfig.vector_store.top_k}")
    print(f"  相似度阈值: {defaultConfig.vector_store.score_threshold}")

    print(f"\n📝 文本分割配置:")
    print(f"  块大小: {defaultConfig.text_splitter.chunk_size}")
    print(f"  重叠大小: {defaultConfig.text_splitter.chunk_overlap}")

    print(f"\n📁 路径配置:")
    print(f"  数据目录: {defaultConfig.path.data_dir}")
    print(f"  文档目录: {defaultConfig.path.documents_dir}")
    print(f"  日志目录: {defaultConfig.path.log_dir}")

    print("=" * 50)


class Config(BaseModel):
    """总配置类"""
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    text_splitter: TextSplitterConfig = Field(default_factory=TextSplitterConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    path: PathConfig = Field(default_factory=PathConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    mysql: MySQLConfig = Field(default_factory=MySQLConfig)
    postgresql: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)

# 默认配置实例
defaultConfig = Config()


if __name__ == "__main__":
    load_env_config()
    print_config()
    
    device = get_device()
    print(f"\n检测到的设备: {device}")
    
    logger = setup_logging()
    logger.info("配置模块测试成功！")
