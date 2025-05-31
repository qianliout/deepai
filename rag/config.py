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
    
    persist_directory: str = Field(default="data/vectorstore", description="ChromaDB数据目录")
    collection_name: str = Field(default="knowledge_base", description="集合名称")
    top_k: int = Field(default=5, description="检索返回数量")
    score_threshold: float = Field(default=0.7, description="相似度阈值")
    
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
    log_dir: str = Field(default="logs", description="日志目录")
    
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


# 全局配置实例 - 参考BERT项目的方式
EMBEDDING_CONFIG = EmbeddingConfig()
LLM_CONFIG = LLMConfig()
VECTORSTORE_CONFIG = VectorStoreConfig()
TEXT_SPLITTER_CONFIG = TextSplitterConfig()
TOKENIZER_CONFIG = TokenizerConfig()
PATH_CONFIG = PathConfig()
LOGGING_CONFIG = LoggingConfig()


def get_device() -> str:
    """自动检测设备"""
    try:
        import torch
        if EMBEDDING_CONFIG.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return EMBEDDING_CONFIG.device
    except ImportError:
        return "cpu"


def setup_logging():
    """设置日志系统"""
    log_dir = Path(PATH_CONFIG.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_filename = f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = log_dir / log_filename
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG.log_level),
        format=LOGGING_CONFIG.log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filepath, encoding="utf-8"),
        ],
    )
    
    logger = logging.getLogger("RAG")
    logger.setLevel(getattr(logging, LOGGING_CONFIG.log_level))
    return logger


def create_directories():
    """创建所有必要的目录"""
    directories = [
        PATH_CONFIG.data_dir,
        PATH_CONFIG.documents_dir,
        PATH_CONFIG.vectorstore_dir,
        PATH_CONFIG.log_dir,
        EMBEDDING_CONFIG.cache_dir,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")


def load_env_config():
    """从环境变量加载配置"""
    if api_key := os.getenv("DASHSCOPE_API_KEY"):
        LLM_CONFIG.api_key = api_key
    
    if device := os.getenv("RAG_DEVICE"):
        EMBEDDING_CONFIG.device = device


def print_config():
    """打印配置信息"""
    print("=" * 50)
    print("RAG个人知识库配置信息")
    print("=" * 50)
    
    print(f"\n🤖 LLM配置:")
    print(f"  模型: {LLM_CONFIG.model_name}")
    print(f"  温度: {LLM_CONFIG.temperature}")
    print(f"  最大tokens: {LLM_CONFIG.max_tokens}")
    print(f"  API密钥: {'已设置' if LLM_CONFIG.api_key else '未设置'}")
    
    print(f"\n📊 嵌入配置:")
    print(f"  模型: {EMBEDDING_CONFIG.model_name}")
    print(f"  设备: {get_device()}")
    print(f"  最大长度: {EMBEDDING_CONFIG.max_length}")
    print(f"  批次大小: {EMBEDDING_CONFIG.batch_size}")
    
    print(f"\n🗂️ 向量存储配置:")
    print(f"  存储目录: {VECTORSTORE_CONFIG.persist_directory}")
    print(f"  集合名称: {VECTORSTORE_CONFIG.collection_name}")
    print(f"  检索数量: {VECTORSTORE_CONFIG.top_k}")
    print(f"  相似度阈值: {VECTORSTORE_CONFIG.score_threshold}")
    
    print(f"\n📝 文本分割配置:")
    print(f"  块大小: {TEXT_SPLITTER_CONFIG.chunk_size}")
    print(f"  重叠大小: {TEXT_SPLITTER_CONFIG.chunk_overlap}")
    
    print(f"\n📁 路径配置:")
    print(f"  数据目录: {PATH_CONFIG.data_dir}")
    print(f"  文档目录: {PATH_CONFIG.documents_dir}")
    print(f"  日志目录: {PATH_CONFIG.log_dir}")
    
    print("=" * 50)



if __name__ == "__main__":
    load_env_config()
    print_config()
    
    device = get_device()
    print(f"\n检测到的设备: {device}")
    
    logger = setup_logging()
    logger.info("配置模块测试成功！")
