"""
T5配置模块 - 统一的参数管理
所有超参数都在这里定义，运行时不需要手动传参
基于pydantic进行数据结构定义和验证
"""

import torch
from pydantic import BaseModel, Field
from typing import Optional, List
import logging


class T5Config(BaseModel):
    """T5模型配置"""

    # 模型架构参数
    vocab_size: int = Field(default=32128, description="词汇表大小")
    d_model: int = Field(default=512, description="模型维度")
    d_kv: int = Field(default=64, description="键值维度")
    d_ff: int = Field(default=2048, description="前馈网络维度")
    num_layers: int = Field(default=6, description="编码器和解码器层数")
    num_heads: int = Field(default=8, description="注意力头数")
    
    # 相对位置编码
    relative_attention_num_buckets: int = Field(default=32, description="相对位置编码桶数")
    relative_attention_max_distance: int = Field(default=128, description="相对位置编码最大距离")
    
    # 正则化参数
    dropout_rate: float = Field(default=0.1, description="Dropout概率")
    layer_norm_epsilon: float = Field(default=1e-6, description="LayerNorm的epsilon")
    
    # 初始化参数
    initializer_factor: float = Field(default=1.0, description="权重初始化因子")
    use_custom_init: bool = Field(default=True, description="是否使用自定义权重初始化")

    # 序列长度
    max_length: int = Field(default=512, description="最大序列长度")
    
    # 任务特定参数
    decoder_start_token_id: int = Field(default=0, description="解码器开始token id")
    eos_token_id: int = Field(default=1, description="结束token id")
    pad_token_id: int = Field(default=0, description="padding token id")
    
    class Config:
        """Pydantic配置"""
        extra = "forbid"  # 禁止额外字段


class TrainingConfig(BaseModel):
    """训练配置"""

    # 基础训练参数
    batch_size: int = Field(default=8, description="批次大小")
    learning_rate: float = Field(default=5e-5, description="学习率")
    num_epochs: int = Field(default=3, description="训练轮数")
    warmup_steps: int = Field(default=100, description="预热步数")
    
    # 优化器参数
    weight_decay: float = Field(default=0.01, description="权重衰减")
    adam_epsilon: float = Field(default=1e-8, description="Adam优化器epsilon")
    adam_beta1: float = Field(default=0.9, description="Adam优化器beta1")
    adam_beta2: float = Field(default=0.999, description="Adam优化器beta2")
    max_grad_norm: float = Field(default=1.0, description="梯度裁剪阈值")
    
    # 数据集参数
    dataset_name: str = Field(default="simple_test", description="数据集名称")
    dataset_config: str = Field(default="3.0.0", description="数据集配置版本")
    max_samples: Optional[int] = Field(default=1000, description="最大样本数（用于快速测试）")
    
    # 设备和并行
    device: str = Field(default="auto", description="训练设备")
    num_workers: int = Field(default=4, description="数据加载器工作进程数")
    
    # 日志和保存
    logging_steps: int = Field(default=100, description="日志记录步数")
    save_steps: int = Field(default=1000, description="模型保存步数")
    eval_steps: int = Field(default=500, description="评估步数")
    
    # 目录配置 - 所有路径都在这里统一定义
    # 预训练相关目录
    pretrain_checkpoints_dir: str = Field(
        default="/Users/liuqianli/work/python/deepai/saved_model/t5/pretrain/checkpoints",
        description="预训练过程中的检查点保存目录",
    )
    pretrain_best_dir: str = Field(
        default="/Users/liuqianli/work/python/deepai/saved_model/t5/pretrain/best",
        description="预训练最佳模型保存目录",
    )
    pretrain_final_dir: str = Field(
        default="/Users/liuqianli/work/python/deepai/saved_model/t5/pretrain/final",
        description="预训练最终模型保存目录",
    )
    
    # 微调相关目录
    finetuning_checkpoints_dir: str = Field(
        default="/Users/liuqianli/work/python/deepai/saved_model/t5/finetuning/checkpoints",
        description="微调过程中的检查点保存目录",
    )
    finetuning_best_dir: str = Field(
        default="/Users/liuqianli/work/python/deepai/saved_model/t5/finetuning/best",
        description="微调最佳模型保存目录",
    )
    finetuning_final_dir: str = Field(
        default="/Users/liuqianli/work/python/deepai/saved_model/t5/finetuning/final",
        description="微调最终模型保存目录",
    )
    
    # 其他目录
    log_dir: str = Field(
        default="/Users/liuqianli/work/python/deepai/logs/t5",
        description="日志保存目录",
    )
    cache_dir: str = Field(
        default="/Users/liuqianli/.cache/huggingface/datasets",
        description="HuggingFace数据集缓存目录",
    )
    
    class Config:
        """Pydantic配置"""
        extra = "forbid"


class DataConfig(BaseModel):
    """数据配置"""

    # Tokenizer配置
    tokenizer_name: str = Field(default="t5-small", description="tokenizer名称")

    # 序列长度
    max_source_length: int = Field(default=512, description="源序列最大长度")
    max_target_length: int = Field(default=128, description="目标序列最大长度")

    # 任务前缀
    task_prefix: str = Field(default="translate English to German: ", description="任务前缀")

    class Config:
        """Pydantic配置"""
        extra = "forbid"


class LoggingConfig(BaseModel):
    """日志配置"""

    log_level: str = Field(default="INFO", description="日志级别")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日志格式",
    )
    log_file: Optional[str] = Field(
        default=None, description="日志文件路径，将使用TRAINING_CONFIG.log_dir"
    )
    
    class Config:
        """Pydantic配置"""
        extra = "forbid"


# 全局配置实例 - 运行时直接使用这些配置，不需要手动传参
T5_CONFIG = T5Config()
TRAINING_CONFIG = TrainingConfig()
DATA_CONFIG = DataConfig()
LOGGING_CONFIG = LoggingConfig()


def get_device() -> torch.device:
    """
    自动检测并返回最佳设备
    
    Returns:
        torch.device: 设备对象
    """
    if TRAINING_CONFIG.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(TRAINING_CONFIG.device)
    
    return device


def setup_logging():
    """设置日志系统"""
    import os
    from datetime import datetime
    
    # 创建日志目录
    log_dir = TRAINING_CONFIG.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    log_filename = f"t5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG.log_level),
        format=LOGGING_CONFIG.log_format,
        handlers=[
            logging.StreamHandler(),  # 控制台输出
            logging.FileHandler(log_filepath, encoding="utf-8"),  # 文件输出
        ],
    )
    
    # 创建T5专用logger
    logger = logging.getLogger("T5")
    logger.setLevel(getattr(logging, LOGGING_CONFIG.log_level))
    
    return logger


def create_directories():
    """创建所有必要的目录"""
    import os
    
    directories = [
        # 预训练相关目录
        TRAINING_CONFIG.pretrain_checkpoints_dir,
        TRAINING_CONFIG.pretrain_best_dir,
        TRAINING_CONFIG.pretrain_final_dir,
        # 微调相关目录
        TRAINING_CONFIG.finetuning_checkpoints_dir,
        TRAINING_CONFIG.finetuning_best_dir,
        TRAINING_CONFIG.finetuning_final_dir,
        # 其他目录
        TRAINING_CONFIG.log_dir,
        TRAINING_CONFIG.cache_dir,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 创建目录: {directory}")


def print_config():
    """打印所有配置信息"""
    print("=" * 50)
    print("T5框架配置信息")
    print("=" * 50)
    
    print("\n🏗️ 模型配置:")
    print(f"  词汇表大小: {T5_CONFIG.vocab_size:,}")
    print(f"  模型维度: {T5_CONFIG.d_model}")
    print(f"  层数: {T5_CONFIG.num_layers}")
    print(f"  注意力头数: {T5_CONFIG.num_heads}")
    print(f"  最大序列长度: {T5_CONFIG.max_length}")
    
    print("\n🚀 训练配置:")
    print(f"  批次大小: {TRAINING_CONFIG.batch_size}")
    print(f"  学习率: {TRAINING_CONFIG.learning_rate}")
    print(f"  训练轮数: {TRAINING_CONFIG.num_epochs}")
    print(f"  最大样本数: {TRAINING_CONFIG.max_samples}")
    print(f"  设备: {get_device()}")
    
    print("\n📊 数据配置:")
    print(f"  数据集: {TRAINING_CONFIG.dataset_name}")
    print(f"  Tokenizer: {DATA_CONFIG.tokenizer_name}")
    print(f"  源序列长度: {DATA_CONFIG.max_source_length}")
    print(f"  目标序列长度: {DATA_CONFIG.max_target_length}")
    
    print("=" * 50)


if __name__ == "__main__":
    # 测试配置
    print_config()
    
    # 测试设备检测
    device = get_device()
    print(f"\n检测到的设备: {device}")
    
    # 测试日志
    logger = setup_logging()
    logger.info("T5配置模块测试成功！")
