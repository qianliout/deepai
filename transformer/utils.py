"""
工具函数 - 日志、文件操作等辅助功能
"""

import os
import json
import logging
import torch
import math
from typing import Dict, List, Any
from datetime import datetime


def setup_logging(log_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    设置日志系统

    Args:
        log_dir: 日志目录
        log_level: 日志级别

    Returns:
        配置好的logger
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 创建logger
    logger = logging.getLogger("transformer")
    logger.setLevel(log_level)

    # 清除已有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 文件handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"training_{timestamp}.log"), encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_vocab(vocab: Dict[str, int], path: str, lang: str):
    """
    保存词典到文件

    Args:
        vocab: 词典字典
        path: 保存路径
        lang: 语言标识
    """
    logger = logging.getLogger("transformer.utils")
    os.makedirs(path, exist_ok=True)
    vocab_file = os.path.join(path, f"{lang}_vocab.json")

    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    logger.info(f"词典已保存到: {vocab_file}")


def load_vocab(path: str, lang: str) -> Dict[str, int]:
    """
    从文件加载词典

    Args:
        path: 词典路径
        lang: 语言标识

    Returns:
        词典字典
    """
    vocab_file = os.path.join(path, f"{lang}_vocab.json")

    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"词典文件不存在: {vocab_file}")

    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    return vocab


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量

    Args:
        model: PyTorch模型

    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
):
    """
    保存模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮数
        loss: 当前损失
        path: 保存路径
    """
    logger = logging.getLogger("transformer.utils")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(checkpoint, path)
    logger.info(f"模型已保存到: {path}")


def load_model(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str, device: str
):
    """
    加载模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        path: 模型路径
        device: 设备

    Returns:
        (epoch, loss)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    logger = logging.getLogger("transformer.utils")
    logger.info(f"模型已从 {path} 加载，轮数: {epoch}, 损失: {loss:.4f}")

    return epoch, loss


def get_device() -> str:
    """
    获取可用的设备

    Returns:
        设备字符串
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def format_time(seconds: float) -> str:
    """
    格式化时间显示

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    创建填充掩码

    Args:
        seq: 输入序列 [batch_size, seq_len]
        pad_idx: 填充token的索引

    Returns:
        掩码张量 [batch_size, 1, 1, seq_len]
    """
    # seq == pad_idx 的位置为True，需要被掩码
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size: int) -> torch.Tensor:
    """
    创建前瞻掩码（用于解码器自注意力）

    Args:
        size: 序列长度

    Returns:
        掩码张量 [size, size]
    """
    # 上三角矩阵，对角线及以下为False，上三角为True
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask
