"""
RAG2项目环境配置管理
支持开发环境和生产环境的模型切换
"""

import os
from typing import Dict, Any
from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    """模型配置数据类"""
    provider: str
    model_name: str
    api_base: str
    temperature: float
    max_tokens: int
    device: str = "cpu"
    use_fp16: bool = False
    dimensions: int = 768

@dataclass
class EnvironmentConfig:
    """环境配置数据类"""
    llm: ModelConfig
    embedding: ModelConfig
    reranker: ModelConfig
    environment: str

# 开发环境配置 - 使用小模型，支持本地调试
DEV_CONFIG = EnvironmentConfig(
    environment="development",
    llm=ModelConfig(
        provider="ollama",
        model_name="qwen2.5:7b",
        api_base="http://localhost:11434/v1",
        temperature=0.1,
        max_tokens=2048,
        device="cpu"
    ),
    embedding=ModelConfig(
        provider="huggingface",
        model_name="BAAI/bge-base-zh-v1.5",
        api_base="",
        temperature=0.0,
        max_tokens=0,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        dimensions=768
    ),
    reranker=ModelConfig(
        provider="huggingface",
        model_name="BAAI/bge-reranker-base",
        api_base="",
        temperature=0.0,
        max_tokens=0,
        device="cpu",
        use_fp16=False
    )
)

# 生产环境配置 - 使用大模型，追求最佳性能
PROD_CONFIG = EnvironmentConfig(
    environment="production",
    llm=ModelConfig(
        provider="deepseek",
        model_name="deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        temperature=0.1,
        max_tokens=4096,
        device="cuda"
    ),
    embedding=ModelConfig(
        provider="huggingface",
        model_name="BAAI/bge-large-zh-v1.5",
        api_base="",
        temperature=0.0,
        max_tokens=0,
        device="cuda",
        dimensions=1024
    ),
    reranker=ModelConfig(
        provider="huggingface",
        model_name="BAAI/bge-reranker-v2-m3",
        api_base="",
        temperature=0.0,
        max_tokens=0,
        device="cuda",
        use_fp16=True
    )
)

def get_environment_config() -> EnvironmentConfig:
    """
    根据环境变量获取对应的配置
    
    Returns:
        EnvironmentConfig: 当前环境的配置
    """
    env = os.getenv("RAG_ENV", "development").lower()
    
    if env == "production":
        return PROD_CONFIG
    else:
        return DEV_CONFIG

def get_model_config_dict() -> Dict[str, Any]:
    """
    获取当前环境的模型配置字典
    
    Returns:
        Dict[str, Any]: 模型配置字典
    """
    config = get_environment_config()
    
    return {
        "environment": config.environment,
        "llm": {
            "provider": config.llm.provider,
            "model_name": config.llm.model_name,
            "api_base": config.llm.api_base,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
            "device": config.llm.device
        },
        "embedding": {
            "provider": config.embedding.provider,
            "model_name": config.embedding.model_name,
            "device": config.embedding.device,
            "dimensions": config.embedding.dimensions,
            "normalize_embeddings": True
        },
        "reranker": {
            "provider": config.reranker.provider,
            "model_name": config.reranker.model_name,
            "device": config.reranker.device,
            "use_fp16": config.reranker.use_fp16,
            "top_k": 5 if config.environment == "development" else 10
        }
    }

def print_current_config():
    """打印当前环境配置信息"""
    config = get_environment_config()
    print(f"当前环境: {config.environment}")
    print(f"LLM模型: {config.llm.model_name} ({config.llm.provider})")
    print(f"嵌入模型: {config.embedding.model_name}")
    print(f"重排序模型: {config.reranker.model_name}")
    print(f"计算设备: {config.llm.device}")

# 环境变量检查
def check_environment_variables():
    """检查必要的环境变量是否设置"""
    config = get_environment_config()
    missing_vars = []
    
    if config.environment == "production":
        if not os.getenv("DEEPSEEK_API_KEY"):
            missing_vars.append("DEEPSEEK_API_KEY")
    
    if config.llm.device == "cuda":
        if not os.getenv("CUDA_VISIBLE_DEVICES"):
            print("警告: 未设置CUDA_VISIBLE_DEVICES环境变量")
    
    if missing_vars:
        raise ValueError(f"缺少必要的环境变量: {', '.join(missing_vars)}")
    
    return True

# 模型资源需求估算
def estimate_resource_requirements():
    """估算当前配置的资源需求"""
    config = get_environment_config()
    
    if config.environment == "development":
        return {
            "cpu_cores": 4,
            "memory_gb": 8,
            "gpu_memory_gb": 0,  # 开发环境使用CPU
            "disk_space_gb": 10
        }
    else:
        return {
            "cpu_cores": 8,
            "memory_gb": 16,
            "gpu_memory_gb": 12,  # 生产环境需要GPU
            "disk_space_gb": 50
        }

if __name__ == "__main__":
    # 测试配置加载
    print("=== RAG2 环境配置测试 ===")
    print_current_config()
    print("\n=== 资源需求估算 ===")
    resources = estimate_resource_requirements()
    for key, value in resources.items():
        print(f"{key}: {value}")
    
    print("\n=== 环境变量检查 ===")
    try:
        check_environment_variables()
        print("环境变量检查通过")
    except ValueError as e:
        print(f"环境变量检查失败: {e}")
