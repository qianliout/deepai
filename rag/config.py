"""
配置管理模块
"""
from typing import Optional, List
from pydantic import BaseModel, Field
from pathlib import Path

class VectorStoreConfig(BaseModel):
    """向量存储配置"""
    store_type: str = Field(default="chroma", description="向量存储类型: chroma/faiss")
    persist_directory: str = Field(default="vectorstore", description="向量数据持久化目录")
    collection_name: str = Field(default="knowledge_base", description="集合名称")

class EmbeddingConfig(BaseModel):
    """文本嵌入模型配置"""
    model_name: str = Field(
        default="BAAI/bge-small-zh-v1.5", 
        description="文本嵌入模型名称"
    )
    device: str = Field(default="mps", description="设备类型: cpu/mps")
    max_length: int = Field(default=512, description="最大文本长度")

class LLMConfig(BaseModel):
    """大语言模型配置"""
    api_key: str = Field(default="", description="通义千问API key")
    model_name: str = Field(
        default="qwen-max", 
        description="模型名称"
    )
    temperature: float = Field(default=0.7, description="采样温度")
    top_p: float = Field(default=0.9, description="核采样概率")
    max_tokens: int = Field(default=2048, description="最大生成token数")

class TextSplitterConfig(BaseModel):
    """文本分割配置"""
    chunk_size: int = Field(default=500, description="文本块大小")
    chunk_overlap: int = Field(default=50, description="文本块重叠大小")
    length_function: str = Field(default="char", description="长度计算函数: char/token")

class Config(BaseModel):
    """总配置类"""
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    text_splitter: TextSplitterConfig = Field(default_factory=TextSplitterConfig)
    
    data_dir: str = Field(default="data", description="数据目录")
    knowledge_dir: str = Field(default="knowledge", description="知识库文档目录")
    log_dir: str = Field(default="logs", description="日志目录")
    
    def __init__(self, **data):
        super().__init__(**data)
        # 确保目录存在
        for dir_path in [self.data_dir, self.knowledge_dir, self.log_dir,
                        self.vector_store.persist_directory]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

# 默认配置实例
config = Config()
