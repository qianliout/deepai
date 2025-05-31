"""
RAG个人知识库配置管理模块

简化版配置，专注于核心RAG功能：
- 文本嵌入配置 (BGE模型)
- ChromaDB向量存储配置
- 通义百炼LLM配置
- 文本分割配置
- Redis会话存储配置

数据流：配置加载 -> 环境变量覆盖 -> 目录创建
"""

import os
from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path
from enum import Enum


class DeviceType(str, Enum):
    """计算设备类型"""

    CPU = "cpu"
    MPS = "mps"  # Apple Silicon GPU
    CUDA = "cuda"


class ChromaDBConfig(BaseModel):
    """ChromaDB向量存储配置

    使用ChromaDB作为向量数据库，底层使用SQLite存储
    """

    persist_directory: str = Field(default="data/vectorstore", description="ChromaDB数据持久化目录")
    collection_name: str = Field(default="knowledge_base", description="集合名称")
    top_k: int = Field(default=5, description="检索返回的文档数量")
    score_threshold: float = Field(default=0.7, description="相似度阈值")

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v <= 0:
            raise ValueError("top_k必须大于0")
        return v


class EmbeddingConfig(BaseModel):
    """文本嵌入模型配置

    使用BGE中文嵌入模型进行文本向量化
    """

    model_name: str = Field(default="BAAI/bge-small-zh-v1.5", description="文本嵌入模型名称，支持中英文")
    device: DeviceType = Field(default=DeviceType.MPS, description="计算设备类型")
    max_length: int = Field(default=512, description="最大文本长度")
    batch_size: int = Field(default=32, description="批处理大小")
    cache_dir: str = Field(default="/Users/liuqianli/.cache/huggingface/hub", description="模型缓存目录")
    normalize_embeddings: bool = Field(default=True, description="是否标准化嵌入向量")

    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, v):
        if v <= 0 or v > 8192:
            raise ValueError("max_length必须在1-8192之间")
        return v


class LLMConfig(BaseModel):
    """大语言模型配置

    配置通义百炼大模型的API参数和生成配置
    """

    # API配置
    api_key: str = Field(default="", description="通义百炼API密钥")
    model_name: str = Field(default="qwen-max", description="模型名称")

    # 生成参数
    temperature: float = Field(default=0.7, description="采样温度，控制生成的随机性")
    top_p: float = Field(default=0.9, description="核采样概率")
    max_tokens: int = Field(default=2048, description="最大生成token数")

    # 请求配置
    timeout: int = Field(default=60, description="请求超时时间(秒)")
    max_retries: int = Field(default=3, description="最大重试次数")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError("temperature必须在0-2之间")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("top_p必须在0-1之间")
        return v


class RedisConfig(BaseModel):
    """Redis会话存储配置

    用于存储对话历史和会话上下文
    """

    host: str = Field(default="localhost", description="Redis主机地址")
    port: int = Field(default=6379, description="Redis端口")
    password: str = Field(default="", description="Redis密码")
    db: int = Field(default=0, description="Redis数据库编号")
    session_ttl: int = Field(default=3600, description="会话过期时间(秒)")
    max_history: int = Field(default=20, description="最大历史记录数")


class ChineseTokenizerConfig(BaseModel):
    """中文分词配置"""
    tokenizer_type: str = Field(default="jieba", description="分词器类型: manual/jieba")
    remove_stop_words: bool = Field(default=True, description="是否移除停用词")
    user_dict_path: str = Field(default="", description="用户词典路径")
    enable_pos_tagging: bool = Field(default=False, description="是否启用词性标注")

class QueryExpansionConfig(BaseModel):
    """查询扩展配置"""
    enable_synonyms: bool = Field(default=True, description="是否启用同义词扩展")
    max_synonyms_per_word: int = Field(default=2, description="每个词的最大同义词数量")
    similarity_threshold: float = Field(default=0.7, description="同义词相似度阈值")
    max_expansion_ratio: float = Field(default=2.0, description="最大扩展比例")

class RetrieverConfig(BaseModel):
    """检索器配置"""

    enable_hybrid_search: bool = Field(default=False, description="是否启用混合检索")
    enable_reranking: bool = Field(default=False, description="是否启用重排序")
    enable_query_expansion: bool = Field(default=True, description="是否启用查询扩展")
    keyword_weight: float = Field(default=0.3, description="关键词检索权重")
    semantic_weight: float = Field(default=0.7, description="语义检索权重")
    rerank_top_k: int = Field(default=10, description="重排序候选数量")


class TextSplitterConfig(BaseModel):
    """文本分割配置

    配置文档分块的策略和参数，影响检索的粒度和效果
    """

    chunk_size: int = Field(default=500, description="文本块大小(字符数)")
    chunk_overlap: int = Field(default=50, description="文本块重叠大小")
    separators: List[str] = Field(default=["\n\n", "\n", "。", "！", "？", "；", " ", ""], description="文本分割符优先级列表")
    keep_separator: bool = Field(default=True, description="是否保留分隔符")

    @model_validator(mode="after")
    def validate_chunk_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap必须小于chunk_size")
        return self


class LoggingConfig(BaseModel):
    """日志配置"""

    level: str = Field(default="INFO", description="日志级别")
    log_dir: str = Field(default="logs", description="日志目录")
    max_file_size: str = Field(default="10 MB", description="单个日志文件最大大小")
    retention: str = Field(default="30 days", description="日志保留时间")


class Config(BaseModel):
    """RAG系统总配置类

    整合所有子配置，提供统一的配置管理接口
    """

    # 子配置模块
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    text_splitter: TextSplitterConfig = Field(default_factory=TextSplitterConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    chinese_tokenizer: ChineseTokenizerConfig = Field(default_factory=ChineseTokenizerConfig)
    query_expansion: QueryExpansionConfig = Field(default_factory=QueryExpansionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # 项目路径配置
    data_dir: str = Field(default="data", description="数据目录")
    documents_dir: str = Field(default="data/documents", description="文档目录")

    # 系统配置
    environment: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=True, description="是否开启调试模式")

    def __init__(self, **data):
        super().__init__(**data)
        self._load_from_env()
        self._create_directories()

    def _load_from_env(self):
        """从环境变量加载配置"""
        # 加载API密钥
        if api_key := os.getenv("DASHSCOPE_API_KEY"):
            self.llm.api_key = api_key

        # 加载Redis配置
        if redis_host := os.getenv("REDIS_HOST"):
            self.redis.host = redis_host
        if redis_port := os.getenv("REDIS_PORT"):
            self.redis.port = int(redis_port)
        if redis_password := os.getenv("REDIS_PASSWORD"):
            self.redis.password = redis_password

        # 加载环境配置
        if env := os.getenv("ENVIRONMENT"):
            self.environment = env
            if env == "production":
                self.debug = False
                self.logging.level = "WARNING"

    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.data_dir,
            self.documents_dir,
            self.logging.log_dir,
            self.chromadb.persist_directory,
            self.embedding.cache_dir,
        ]

        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get_device(self) -> str:
        """获取可用的计算设备"""
        try:
            import torch

            if self.embedding.device == DeviceType.MPS and torch.backends.mps.is_available():
                return "mps"
            elif self.embedding.device == DeviceType.CUDA and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"


# 全局配置实例
config = Config()
