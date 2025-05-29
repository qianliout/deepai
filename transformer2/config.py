"""
Transformer2配置文件 - 使用Pydantic定义所有数据结构和超参数
重构版本，消除重复代码，优化结构，参考bert2实现方式
详细的数据流转注释和shape说明
"""

import torch
from pydantic import BaseModel, Field
from typing import Optional
import logging
import os
from datetime import datetime


class TransformerConfig(BaseModel):
    """Transformer模型配置

    定义模型架构的所有超参数，包括：
    - 模型维度和层数配置
    - 注意力机制配置
    - 正则化参数
    - 特殊token定义

    数据流转说明：
    输入序列 [batch_size, seq_len] ->
    Embedding [batch_size, seq_len, d_model] ->
    Encoder/Decoder [batch_size, seq_len, d_model] ->
    输出投影 [batch_size, seq_len, vocab_size]
    """

    # 模型架构参数
    vocab_size_src: int = Field(default=10000, description="源语言词汇表大小")
    vocab_size_tgt: int = Field(default=10000, description="目标语言词汇表大小")
    d_model: int = Field(default=512, description="模型维度 (embedding维度)")
    n_heads: int = Field(default=8, description="多头注意力头数，d_model必须能被n_heads整除")
    n_layers: int = Field(default=6, description="编码器/解码器层数")
    d_ff: int = Field(default=2048, description="前馈网络中间层维度，通常是d_model的4倍")

    # 序列长度配置
    max_seq_len: int = Field(default=512, description="最大序列长度，影响位置编码")

    # 正则化参数
    dropout: float = Field(default=0.1, description="Dropout概率，应用于注意力和前馈网络")
    layer_norm_eps: float = Field(default=1e-6, description="LayerNorm的epsilon值，防止除零")

    # 权重初始化参数
    init_range: float = Field(default=0.02, description="权重初始化范围，使用正态分布")

    # 特殊token ID定义
    pad_token_id: int = Field(default=0, description="填充token ID")
    bos_token_id: int = Field(default=1, description="序列开始token ID")
    eos_token_id: int = Field(default=2, description="序列结束token ID")
    unk_token_id: int = Field(default=3, description="未知token ID")

    class Config:
        """Pydantic配置"""

        extra = "forbid"  # 禁止额外字段


class TrainingConfig(BaseModel):
    """训练配置

    定义训练过程的所有参数，包括：
    - 基础训练参数（batch size, learning rate等）
    - 优化器配置
    - 数据集配置
    - 设备和并行配置
    - 日志和保存配置

    数据流转说明：
    训练批次 [batch_size, seq_len] ->
    模型前向传播 [batch_size, seq_len, vocab_size] ->
    损失计算 [batch_size, seq_len] ->
    反向传播和参数更新
    """

    # 基础训练参数
    batch_size: int = Field(default=32, description="训练批次大小，影响内存使用和梯度稳定性")
    learning_rate: float = Field(default=1e-4, description="初始学习率，会通过warmup调度")
    num_epochs: int = Field(default=1, description="训练轮数")
    warmup_steps: int = Field(default=4000, description="学习率预热步数，论文建议4000")

    # 优化器参数
    weight_decay: float = Field(default=0.01, description="权重衰减系数，L2正则化")
    adam_beta1: float = Field(default=0.9, description="Adam优化器beta1参数，一阶矩估计")
    adam_beta2: float = Field(default=0.98, description="Adam优化器beta2参数，二阶矩估计")
    adam_eps: float = Field(default=1e-9, description="Adam优化器epsilon参数，防止除零")
    max_grad_norm: float = Field(default=1.0, description="梯度裁剪阈值，防止梯度爆炸")

    # 数据集参数
    dataset_name: str = Field(default="Helsinki-NLP/opus_books", description="HuggingFace数据集名称")
    language_pair: str = Field(default="en-it", description="语言对 (源语言-目标语言)")
    max_samples: Optional[int] = Field(default=1000, description="最大样本数（用于快速测试，None表示使用全部数据）")
    train_split: str = Field(default="train", description="训练集分割名称")
    val_split: str = Field(default="validation", description="验证集分割名称")

    # 设备和并行配置
    device: str = Field(default="auto", description="训练设备 (auto/cpu/cuda/mps)")
    num_workers: int = Field(default=0, description="数据加载器工作进程数 (Mac M1建议设为0)")

    # 日志和保存配置
    logging_steps: int = Field(default=100, description="每多少步记录一次日志")
    save_steps: int = Field(default=1000, description="每多少步保存一次模型")
    eval_steps: int = Field(default=500, description="每多少步进行一次验证")

    # 目录配置 - 所有路径都在这里统一定义
    model_save_dir: str = Field(default="/Users/liuqianli/work/python/deepai/saved_model/transformer2", description="模型保存目录")
    log_dir: str = Field(default="/Users/liuqianli/work/python/deepai/logs/transformer2", description="日志保存目录")
    cache_dir: str = Field(default="/Users/liuqianli/.cache/huggingface/datasets", description="HuggingFace数据集缓存目录")
    vocab_save_dir: str = Field(default="/Users/liuqianli/work/python/deepai/saved_model/transformer2/vocab", description="词汇表保存目录")

    class Config:
        """Pydantic配置"""

        extra = "forbid"


class DataConfig(BaseModel):
    """数据配置

    定义数据预处理的所有参数，包括：
    - 分词器配置
    - 数据预处理参数
    - 数据增强配置

    数据流转说明：
    原始文本 -> 分词 -> token序列 [seq_len] ->
    添加特殊token -> 填充/截断 [max_length] ->
    转换为ID序列 [max_length] -> 批次 [batch_size, max_length]
    """

    # 分词器配置
    tokenizer_type: str = Field(default="simple", description="分词器类型 (simple/sentencepiece)")
    vocab_file_src: Optional[str] = Field(default=None, description="源语言词汇表文件路径")
    vocab_file_tgt: Optional[str] = Field(default=None, description="目标语言词汇表文件路径")
    min_freq: int = Field(default=2, description="词汇最小频率阈值，低于此频率的词被替换为UNK")

    # 数据预处理参数
    max_length: int = Field(default=128, description="序列最大长度，超过会被截断")
    min_length: int = Field(default=1, description="序列最小长度，过短的序列会被过滤")

    # 数据增强配置
    use_data_augmentation: bool = Field(default=False, description="是否使用数据增强")

    class Config:
        """Pydantic配置"""

        extra = "forbid"


# 创建全局配置实例，所有模块都使用这些实例，无需手动传参
TRANSFORMER_CONFIG = TransformerConfig()
TRAINING_CONFIG = TrainingConfig()
DATA_CONFIG = DataConfig()


# ============================================================================
# 工具函数
# ============================================================================


def get_device() -> str:
    """自动检测可用的计算设备

    Returns:
        str: 设备名称 ('mps'/'cuda'/'cpu')
    """
    if TRAINING_CONFIG.device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    else:
        return TRAINING_CONFIG.device


def setup_logging():
    """设置日志系统

    创建统一的日志配置，包括：
    - 控制台输出
    - 文件输出
    - 第三方库日志级别控制
    """
    # 创建日志目录
    log_dir = TRAINING_CONFIG.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # 设置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 设置根日志器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # 控制台输出
            logging.FileHandler(
                os.path.join(
                    log_dir,
                    f"transformer2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                ),
                encoding="utf-8",
            ),
        ],
    )

    # 设置第三方库日志级别，避免过多输出
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

    return logging.getLogger("Transformer2")


def create_directories():
    """创建所有必要的目录"""
    directories = [
        TRAINING_CONFIG.model_save_dir,
        TRAINING_CONFIG.log_dir,
        TRAINING_CONFIG.cache_dir,
        TRAINING_CONFIG.vocab_save_dir,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 创建目录: {directory}")


def print_config():
    """打印当前配置信息

    以美观的格式显示所有配置参数，便于调试和确认
    """
    print("\n" + "=" * 60)
    print("🔧 Transformer2 配置信息")
    print("=" * 60)

    print(f"\n📊 模型配置:")
    print(f"  源语言词汇表大小: {TRANSFORMER_CONFIG.vocab_size_src:,}")
    print(f"  目标语言词汇表大小: {TRANSFORMER_CONFIG.vocab_size_tgt:,}")
    print(f"  模型维度 (d_model): {TRANSFORMER_CONFIG.d_model}")
    print(f"  注意力头数: {TRANSFORMER_CONFIG.n_heads}")
    print(f"  编码器/解码器层数: {TRANSFORMER_CONFIG.n_layers}")
    print(f"  前馈网络维度: {TRANSFORMER_CONFIG.d_ff}")
    print(f"  最大序列长度: {TRANSFORMER_CONFIG.max_seq_len}")
    print(f"  Dropout概率: {TRANSFORMER_CONFIG.dropout}")

    print(f"\n🚀 训练配置:")
    print(f"  批次大小: {TRAINING_CONFIG.batch_size}")
    print(f"  学习率: {TRAINING_CONFIG.learning_rate}")
    print(f"  训练轮数: {TRAINING_CONFIG.num_epochs}")
    print(f"  预热步数: {TRAINING_CONFIG.warmup_steps}")
    print(f"  最大样本数: {TRAINING_CONFIG.max_samples}")
    print(f"  计算设备: {get_device()}")
    print(f"  模型保存目录: {TRAINING_CONFIG.model_save_dir}")
    print(f"  日志保存目录: {TRAINING_CONFIG.log_dir}")
    print(f"  数据缓存目录: {TRAINING_CONFIG.cache_dir}")

    print(f"\n📁 数据配置:")
    print(f"  数据集: {TRAINING_CONFIG.dataset_name}")
    print(f"  语言对: {TRAINING_CONFIG.language_pair}")
    print(f"  最大序列长度: {DATA_CONFIG.max_length}")
    print(f"  最小词频: {DATA_CONFIG.min_freq}")
    print(f"  分词器类型: {DATA_CONFIG.tokenizer_type}")

    print("=" * 60 + "\n")


def update_config_for_quick_test():
    """更新配置为快速测试模式

    将配置修改为小规模设置，用于快速验证代码正确性：
    - 减小模型规模
    - 减少训练数据量
    - 缩短序列长度
    """
    print("⚡ 切换到快速测试模式...")

    # 小规模模型配置
    TRANSFORMER_CONFIG.d_model = 256
    TRANSFORMER_CONFIG.n_heads = 4
    TRANSFORMER_CONFIG.n_layers = 3
    TRANSFORMER_CONFIG.d_ff = 1024
    TRANSFORMER_CONFIG.vocab_size_src = 5000
    TRANSFORMER_CONFIG.vocab_size_tgt = 5000

    # 快速训练配置
    TRAINING_CONFIG.batch_size = 16
    TRAINING_CONFIG.num_epochs = 2
    TRAINING_CONFIG.max_samples = 100
    TRAINING_CONFIG.model_save_dir = "/Users/liuqianli/work/python/deepai/saved_model/transformer2_quick_test"
    TRAINING_CONFIG.log_dir = "/Users/liuqianli/work/python/deepai/logs/transformer2_quick_test"
    TRAINING_CONFIG.logging_steps = 10
    TRAINING_CONFIG.save_steps = 50
    TRAINING_CONFIG.eval_steps = 25

    # 短序列配置
    DATA_CONFIG.max_length = 64
    TRANSFORMER_CONFIG.max_seq_len = 64

    print("✅ 已切换到快速测试模式")
    print(f"  - 模型维度: {TRANSFORMER_CONFIG.d_model}")
    print(f"  - 训练样本: {TRAINING_CONFIG.max_samples}")
    print(f"  - 序列长度: {DATA_CONFIG.max_length}")


def validate_config():
    """验证配置的合理性

    检查配置参数是否符合要求，防止运行时错误
    """
    # 检查模型配置
    assert (
        TRANSFORMER_CONFIG.d_model % TRANSFORMER_CONFIG.n_heads == 0
    ), f"d_model({TRANSFORMER_CONFIG.d_model})必须能被n_heads({TRANSFORMER_CONFIG.n_heads})整除"

    assert TRANSFORMER_CONFIG.d_model > 0, "d_model必须大于0"
    assert TRANSFORMER_CONFIG.n_heads > 0, "n_heads必须大于0"
    assert TRANSFORMER_CONFIG.n_layers > 0, "n_layers必须大于0"
    assert TRANSFORMER_CONFIG.d_ff > 0, "d_ff必须大于0"

    # 检查训练配置
    assert TRAINING_CONFIG.batch_size > 0, "batch_size必须大于0"
    assert TRAINING_CONFIG.learning_rate > 0, "learning_rate必须大于0"
    assert TRAINING_CONFIG.num_epochs > 0, "num_epochs必须大于0"

    # 检查数据配置
    assert DATA_CONFIG.max_length > 0, "max_length必须大于0"
    assert DATA_CONFIG.min_length > 0, "min_length必须大于0"
    assert DATA_CONFIG.max_length >= DATA_CONFIG.min_length, "max_length必须大于等于min_length"

    print("✅ 配置验证通过")


# ============================================================================
# 自动初始化
# ============================================================================

# 自动设置设备
if TRAINING_CONFIG.device == "auto":
    TRAINING_CONFIG.device = get_device()
    print(f"🔍 自动检测到设备: {TRAINING_CONFIG.device}")

# 验证配置
validate_config()


if __name__ == "__main__":
    # 测试配置
    print_config()

    # 测试设备检测
    device = get_device()
    print(f"\n检测到的设备: {device}")

    # 测试日志
    logger = setup_logging()
    logger.info("配置模块测试成功！")

    # 测试快速模式
    print("\n测试快速模式:")
    update_config_for_quick_test()
    print_config()
