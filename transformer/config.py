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
    num_epochs: int = Field(default=1, description="训练轮数")
    warmup_steps: int = Field(default=4000, description="学习率预热步数")

    # 设备配置
    device: str = Field(
        default="mps" if torch.backends.mps.is_available() else "cpu",
        description="训练设备",
    )

    # 目录配置 - 所有路径都在这里统一定义
    model_save_dir: str = Field(default="/Users/liuqianli/work/python/deepai/saved_model/transformer", description="模型保存目录")
    vocab_save_dir: str = Field(default="/Users/liuqianli/work/python/deepai/saved_model/transformer/vocab", description="词典保存目录")
    log_dir: str = Field(default="/Users/liuqianli/work/python/deepai/logs/transformer", description="日志保存目录")
    cache_dir: str = Field(default="/Users/liuqianli/.cache/huggingface/datasets", description="HuggingFace数据集缓存目录")

    # 日志
    log_interval: int = Field(default=100, description="日志打印间隔")
    save_interval: int = Field(default=1000, description="模型保存间隔")


class DataConfig(BaseModel):
    """数据配置"""

    dataset_name: str = Field(
        default="Helsinki-NLP/opus_books", description="数据集名称"
    )
    language_pair: str = Field(default="en-it", description="语言对")
    # cache_dir 已移动到 TrainingConfig 中统一管理

    # 分词配置
    min_freq: int = Field(default=4, description="词汇最小频率")
    max_vocab_size: int = Field(default=10000, description="最大词汇表大小")


class Config(BaseModel):
    """总配置类"""

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    def save_config(self, path: str):
        """保存配置到文件"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_config(cls, path: str):
        """从文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())


# 默认配置实例
default_config = Config()


def create_directories():
    """创建所有必要的目录"""
    import os

    directories = [
        default_config.training.model_save_dir,
        default_config.training.vocab_save_dir,
        default_config.training.log_dir,
        default_config.training.cache_dir,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 创建目录: {directory}")


def get_device():
    """获取设备"""
    return default_config.training.device
