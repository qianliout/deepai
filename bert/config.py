"""
配置模块 - 统一的参数管理
所有超参数都在这里定义，运行时不需要手动传参
"""

import torch
from pydantic import BaseModel, Field
from typing import Optional
import logging


class BertConfig(BaseModel):
    """BERT模型配置"""

    # 模型架构参数
    vocab_size: int = Field(default=30522, description="词汇表大小")
    hidden_size: int = Field(default=768, description="隐藏层维度")
    num_hidden_layers: int = Field(default=12, description="Transformer层数")
    num_attention_heads: int = Field(default=12, description="注意力头数")
    intermediate_size: int = Field(default=3072, description="前馈网络中间层维度")
    hidden_act: str = Field(default="gelu", description="激活函数")

    # 位置和类型嵌入
    max_position_embeddings: int = Field(default=512, description="最大位置嵌入数")
    type_vocab_size: int = Field(default=2, description="token类型词汇表大小")

    # 正则化参数
    hidden_dropout_prob: float = Field(default=0.1, description="隐藏层dropout概率")
    attention_probs_dropout_prob: float = Field(default=0.1, description="注意力dropout概率")
    layer_norm_eps: float = Field(default=1e-12, description="LayerNorm的epsilon")

    # 初始化参数
    initializer_range: float = Field(default=0.02, description="权重初始化范围")

    # 任务类型（用于分类任务）
    problem_type: Optional[str] = Field(default=None, description="问题类型")

    class Config:
        """Pydantic配置"""

        extra = "forbid"  # 禁止额外字段


class TrainingConfig(BaseModel):
    """训练配置"""

    # 基础训练参数
    batch_size: int = Field(default=16, description="批次大小")
    learning_rate: float = Field(default=1e-4, description="学习率")
    num_epochs: int = Field(default=3, description="训练轮数")
    warmup_steps: int = Field(default=1000, description="预热步数")

    # 优化器参数
    weight_decay: float = Field(default=0.01, description="权重衰减")
    adam_epsilon: float = Field(default=1e-8, description="Adam优化器epsilon")
    adam_beta1: float = Field(default=0.9, description="Adam优化器beta1")
    adam_beta2: float = Field(default=0.999, description="Adam优化器beta2")
    max_grad_norm: float = Field(default=1.0, description="梯度裁剪阈值")

    # 数据集参数
    dataset_name: str = Field(default="Salesforce/wikitext", description="数据集名称")
    dataset_config: str = Field(default="wikitext-2-raw-v1", description="数据集配置")
    max_samples: Optional[int] = Field(default=1000, description="最大样本数（用于快速测试）")

    # 设备和并行
    device: str = Field(default="auto", description="训练设备")
    num_workers: int = Field(default=4, description="数据加载器工作进程数")

    # 日志和保存
    logging_steps: int = Field(default=100, description="日志记录步数")
    save_steps: int = Field(default=1000, description="模型保存步数")

    # 目录配置 - 所有路径都在这里统一定义
    # 预训练相关目录
    pretrain_checkpoints_dir: str = Field(
        default="/Users/liuqianli/work/python/deepai/saved_model/bert/pretrain/checkpoints", description="预训练过程中的检查点保存目录"
    )
    pretrain_best_dir: str = Field(
        default="/Users/liuqianli/work/python/deepai/saved_model/bert/pretrain/best", description="预训练最佳模型保存目录"
    )
    pretrain_final_dir: str = Field(
        default="/Users/liuqianli/work/python/deepai/saved_model/bert/pretrain/final", description="预训练最终模型保存目录"
    )

    # 微调相关目录
    finetuning_checkpoints_dir: str = Field(
        default="/Users/liuqianli/work/python/deepai/saved_model/bert/finetuning/checkpoints", description="微调过程中的检查点保存目录"
    )
    finetuning_best_dir: str = Field(
        default="/Users/liuqianli/work/python/deepai/saved_model/bert/finetuning/best", description="微调最佳模型保存目录"
    )
    finetuning_final_dir: str = Field(
        default="/Users/liuqianli/work/python/deepai/saved_model/bert/finetuning/final", description="微调最终模型保存目录"
    )

    # 其他目录
    log_dir: str = Field(default="/Users/liuqianli/work/python/deepai/logs/bert", description="日志保存目录")
    cache_dir: str = Field(default="/Users/liuqianli/.cache/huggingface/datasets", description="HuggingFace数据集缓存目录")

    class Config:
        """Pydantic配置"""

        extra = "forbid"


class DataConfig(BaseModel):
    """数据配置"""

    # Tokenizer配置
    tokenizer_name: str = Field(default="bert-base-uncased", description="tokenizer名称")
    do_lower_case: bool = Field(default=True, description="是否转换为小写")

    # 序列长度
    max_length: int = Field(default=128, description="最大序列长度")

    # MLM配置
    mlm_probability: float = Field(default=0.15, description="MLM掩码概率")

    # NSP配置
    nsp_probability: float = Field(default=0.5, description="NSP负样本概率")

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
    log_file: Optional[str] = Field(default=None, description="日志文件路径，将使用TRAINING_CONFIG.log_dir")

    class Config:
        """Pydantic配置"""

        extra = "forbid"


# 全局配置实例 - 运行时直接使用这些配置，不需要手动传参
BERT_CONFIG = BertConfig()
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
    log_filename = f"bert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG.log_level),
        format=LOGGING_CONFIG.log_format,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_filepath, encoding="utf-8")],  # 控制台输出  # 文件输出
    )

    # 创建BERT专用logger
    logger = logging.getLogger("BERT")
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
    print("BERT框架配置信息")
    print("=" * 50)

    print("\n🏗️ 模型配置:")
    print(f"  词汇表大小: {BERT_CONFIG.vocab_size:,}")
    print(f"  隐藏层维度: {BERT_CONFIG.hidden_size}")
    print(f"  Transformer层数: {BERT_CONFIG.num_hidden_layers}")
    print(f"  注意力头数: {BERT_CONFIG.num_attention_heads}")
    print(f"  最大序列长度: {DATA_CONFIG.max_length}")

    print("\n🚀 训练配置:")
    print(f"  批次大小: {TRAINING_CONFIG.batch_size}")
    print(f"  学习率: {TRAINING_CONFIG.learning_rate}")
    print(f"  训练轮数: {TRAINING_CONFIG.num_epochs}")
    print(f"  最大样本数: {TRAINING_CONFIG.max_samples}")
    print(f"  设备: {get_device()}")

    print("\n📊 数据配置:")
    print(f"  数据集: {TRAINING_CONFIG.dataset_name}")
    print(f"  配置: {TRAINING_CONFIG.dataset_config}")
    print(f"  Tokenizer: {DATA_CONFIG.tokenizer_name}")
    print(f"  MLM概率: {DATA_CONFIG.mlm_probability}")

    print("\n📁 目录配置:")
    print("  预训练相关目录:")
    print(f"    检查点目录: {TRAINING_CONFIG.pretrain_checkpoints_dir}")
    print(f"    最佳模型目录: {TRAINING_CONFIG.pretrain_best_dir}")
    print(f"    最终模型目录: {TRAINING_CONFIG.pretrain_final_dir}")
    print("  微调相关目录:")
    print(f"    检查点目录: {TRAINING_CONFIG.finetuning_checkpoints_dir}")
    print(f"    最佳模型目录: {TRAINING_CONFIG.finetuning_best_dir}")
    print(f"    最终模型目录: {TRAINING_CONFIG.finetuning_final_dir}")
    print("  其他目录:")
    print(f"    日志保存目录: {TRAINING_CONFIG.log_dir}")
    print(f"    数据缓存目录: {TRAINING_CONFIG.cache_dir}")
    print(f"    日志级别: {LOGGING_CONFIG.log_level}")

    print("=" * 50)


if __name__ == "__main__":
    # 测试配置
    print_config()

    # 测试设备检测
    device = get_device()
    print(f"\n检测到的设备: {device}")

    # 测试日志
    logger = setup_logging()
    logger.info("配置模块测试成功！")
