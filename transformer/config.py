"""
配置文件 - 使用Pydantic定义所有数据结构和超参数
"""

from pydantic import BaseModel, Field
from typing import Optional, List
import torch


class ModelConfig(BaseModel):
    """Transformer模型配置"""

    # 模型维度
    d_model: int = Field(default=512, description="模型隐藏层维度")
    d_ff: int = Field(default=2048, description="前馈网络维度")
    n_heads: int = Field(default=8, description="多头注意力头数")
    n_layers: int = Field(default=6, description="编码器/解码器层数")

    # 词汇表
    vocab_size_en: int = Field(default=10000, description="英语词汇表大小")
    vocab_size_it: int = Field(default=10000, description="意大利语词汇表大小")
    max_seq_len: int = Field(default=128, description="最大序列长度") # 也就transformer中的seq_len

    # 正则化
    dropout: float = Field(default=0.1, description="Dropout概率")

    # 特殊token
    pad_token: str = Field(default="<PAD>", description="填充token")
    unk_token: str = Field(default="<UNK>", description="未知token")
    bos_token: str = Field(default="<BOS>", description="句子开始token")
    eos_token: str = Field(default="<EOS>", description="句子结束token")


class TrainingConfig(BaseModel):
    """训练配置"""

    # 数据
    train_size: int = Field(default=10000, description="训练数据大小")
    val_size: int = Field(default=2000, description="验证数据大小")
    batch_size: int = Field(default=32, description="批次大小")

    # 训练参数
    learning_rate: float = Field(default=1e-4, description="学习率")
    num_epochs: int = Field(default=10, description="训练轮数")
    warmup_steps: int = Field(default=4000, description="学习率预热步数")

    # 设备配置
    device: str = Field(
        default="mps" if torch.backends.mps.is_available() else "cpu",
        description="训练设备",
    )

    # 目录配置 - 所有路径都在这里统一定义
    # 预训练相关目录
    pretrain_checkpoints_dir: str = Field(default="/Users/liuqianli/work/python/deepai/saved_model/transformer/pretrain/checkpoints", description="预训练过程中的模型保存目录")
    pretrain_best_dir: str = Field(default="/Users/liuqianli/work/python/deepai/saved_model/transformer/pretrain/best", description="预训练最佳模型保存目录")
    pretrain_final_dir: str = Field(default="/Users/liuqianli/work/python/deepai/saved_model/transformer/pretrain/final", description="预训练完成后最终模型保存目录")
    pretrain_vocab_dir: str = Field(default="/Users/liuqianli/work/python/deepai/saved_model/transformer/pretrain/vocab", description="字典存放目录")

    # 其他目录
    log_dir: str = Field(default="/Users/liuqianli/work/python/deepai/logs/transformer", description="日志保存目录")
    cache_dir: str = Field(default="/Users/liuqianli/.cache/huggingface/datasets", description="HuggingFace数据集缓存目录")

    # 日志和保存
    log_interval: int = Field(default=100, description="日志打印间隔")
    save_interval: int = Field(default=1000, description="模型保存间隔")


class DataConfig(BaseModel):
    """数据配置"""

    dataset_name: str = Field(
        default="Helsinki-NLP/opus_books", description="数据集名称"
    )
    language_pair: str = Field(default="en-it", description="语言对")

    # 分词配置
    min_freq: int = Field(default=4, description="词汇最小频率")
    max_vocab_size: int = Field(default=10000, description="最大词汇表大小")


# 全局配置实例
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
DATA_CONFIG = DataConfig()


def create_directories():
    """创建所有必要的目录"""
    import os

    directories = [
        TRAINING_CONFIG.pretrain_checkpoints_dir,
        TRAINING_CONFIG.pretrain_best_dir,
        TRAINING_CONFIG.pretrain_final_dir,
        TRAINING_CONFIG.pretrain_vocab_dir,
        TRAINING_CONFIG.log_dir,
        TRAINING_CONFIG.cache_dir,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 创建目录: {directory}")


def get_device():
    """获取设备"""
    if TRAINING_CONFIG.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(TRAINING_CONFIG.device)


def print_config():
    """打印配置信息"""
    print("=" * 60)
    print("Transformer配置信息")
    print("=" * 60)

    print("\n🏗️ 模型配置:")
    print(f"  模型维度: {MODEL_CONFIG.d_model}")
    print(f"  注意力头数: {MODEL_CONFIG.n_heads}")
    print(f"  编码器/解码器层数: {MODEL_CONFIG.n_layers}")
    print(f"  前馈网络维度: {MODEL_CONFIG.d_ff}")
    print(f"  最大序列长度: {MODEL_CONFIG.max_seq_len}")
    print(f"  Dropout: {MODEL_CONFIG.dropout}")

    print("\n🚀 训练配置:")
    print(f"  训练数据大小: {TRAINING_CONFIG.train_size}")
    print(f"  验证数据大小: {TRAINING_CONFIG.val_size}")
    print(f"  批次大小: {TRAINING_CONFIG.batch_size}")
    print(f"  学习率: {TRAINING_CONFIG.learning_rate}")
    print(f"  训练轮数: {TRAINING_CONFIG.num_epochs}")
    print(f"  设备: {get_device()}")

    print("\n📊 数据配置:")
    print(f"  数据集: {DATA_CONFIG.dataset_name}")
    print(f"  语言对: {DATA_CONFIG.language_pair}")
    print(f"  最小词频: {DATA_CONFIG.min_freq}")
    print(f"  最大词汇表大小: {DATA_CONFIG.max_vocab_size}")

    print("\n📁 目录配置:")
    print(f"  预训练检查点目录: {TRAINING_CONFIG.pretrain_checkpoints_dir}")
    print(f"  预训练最佳模型目录: {TRAINING_CONFIG.pretrain_best_dir}")
    print(f"  预训练最终模型目录: {TRAINING_CONFIG.pretrain_final_dir}")
    print(f"  词典保存目录: {TRAINING_CONFIG.pretrain_vocab_dir}")
    print(f"  日志保存目录: {TRAINING_CONFIG.log_dir}")
    print(f"  数据缓存目录: {TRAINING_CONFIG.cache_dir}")

    print("=" * 60)


if __name__ == "__main__":
    # 测试配置
    print_config()
    print("\n测试目录创建:")
    create_directories()