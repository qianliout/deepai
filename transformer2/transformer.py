"""
Transformer核心组件模块 - 重构版本
包含所有transformer的基础组件，详细的数据流转注释和shape说明
消除重复代码，优化结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple

from config import TRANSFORMER_CONFIG

logger = logging.getLogger("Transformer2")


class PositionalEncoding(nn.Module):
    """位置编码模块

    为输入序列添加位置信息，使模型能够理解token的位置关系

    数据流转：
    输入: [batch_size, seq_len, d_model]
    输出: [batch_size, seq_len, d_model] (添加了位置编码)
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000):
        """
        初始化位置编码

        Args:
            d_model: 模型维度
            max_seq_len: 最大序列长度
        """
        super().__init__()
        self.d_model = d_model

        # 创建位置编码矩阵 [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # [max_seq_len, 1]

        # 计算div_term用于sin/cos函数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model//2]

        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_seq_len, d_model//2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_seq_len, d_model//2]

        # 添加batch维度并注册为buffer
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer("pe", pe)

        logger.debug(f"位置编码初始化完成: d_model={d_model}, max_seq_len={max_seq_len}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入tensor [batch_size, seq_len, d_model]

        Returns:
            添加位置编码后的tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # 确保输入维度正确
        assert d_model == self.d_model, f"输入d_model({d_model})与初始化d_model({self.d_model})不匹配"
        assert seq_len <= self.pe.size(1), f"序列长度({seq_len})超过最大长度({self.pe.size(1)})"

        # 添加位置编码
        # x: [batch_size, seq_len, d_model]
        # self.pe[:, :seq_len]: [1, seq_len, d_model]
        # 结果: [batch_size, seq_len, d_model]
        result = x + self.pe[:, :seq_len]

        logger.debug(f"位置编码前向传播: 输入shape={x.shape}, 输出shape={result.shape}")
        return result


class MultiHeadAttention(nn.Module):
    """多头注意力机制

    实现scaled dot-product attention的多头版本

    数据流转：
    输入: query [batch_size, seq_len, d_model]
         key   [batch_size, seq_len, d_model]
         value [batch_size, seq_len, d_model]

    1. 线性变换: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
    2. 重塑为多头: [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
    3. 计算注意力: [batch_size, n_heads, seq_len, d_k] -> [batch_size, n_heads, seq_len, seq_len]
    4. 应用注意力: [batch_size, n_heads, seq_len, seq_len] @ [batch_size, n_heads, seq_len, d_k]
                  -> [batch_size, n_heads, seq_len, d_k]
    5. 合并多头: [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
    6. 输出投影: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力

        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: dropout概率
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model({d_model})必须能被n_heads({n_heads})整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)  # Query投影
        self.w_k = nn.Linear(d_model, d_model)  # Key投影
        self.w_v = nn.Linear(d_model, d_model)  # Value投影
        self.w_o = nn.Linear(d_model, d_model)  # 输出投影

        self.dropout = nn.Dropout(dropout)

        # 缩放因子
        self.scale = math.sqrt(self.d_k)

        logger.debug(f"多头注意力初始化完成: d_model={d_model}, n_heads={n_heads}, d_k={self.d_k}")

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            query: 查询tensor [batch_size, seq_len, d_model]
            key: 键tensor [batch_size, seq_len, d_model]
            value: 值tensor [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len] 或 [seq_len, seq_len]

        Returns:
            output: 输出tensor [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, query_seq_len, d_model = query.shape
        key_batch_size, key_seq_len, key_d_model = key.shape
        value_batch_size, value_seq_len, value_d_model = value.shape

        # 确保输入维度正确
        assert d_model == self.d_model, f"输入d_model({d_model})与初始化d_model({self.d_model})不匹配"
        assert key_d_model == self.d_model, f"key d_model({key_d_model})与初始化d_model({self.d_model})不匹配"
        assert value_d_model == self.d_model, f"value d_model({value_d_model})与初始化d_model({self.d_model})不匹配"
        assert (
            batch_size == key_batch_size == value_batch_size
        ), f"batch_size不匹配: query={batch_size}, key={key_batch_size}, value={value_batch_size}"
        assert key_seq_len == value_seq_len, f"key和value的seq_len必须相同: key={key_seq_len}, value={value_seq_len}"

        # 1. 线性变换
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 2. 重塑为多头
        # Q: [batch_size, query_seq_len, d_model] -> [batch_size, query_seq_len, n_heads, d_k] -> [batch_size, n_heads, query_seq_len, d_k]
        Q = Q.view(batch_size, query_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # K,V: [batch_size, key_seq_len, d_model] -> [batch_size, key_seq_len, n_heads, d_k] -> [batch_size, n_heads, key_seq_len, d_k]
        K = K.view(batch_size, key_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, key_seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 3. 计算注意力
        attention_output, attention_weights = self._scaled_dot_product_attention(Q, K, V, mask)

        # 4. 合并多头
        # [batch_size, n_heads, query_seq_len, d_k] -> [batch_size, query_seq_len, n_heads, d_k] -> [batch_size, query_seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, query_seq_len, self.d_model)

        # 5. 输出投影
        # [batch_size, query_seq_len, d_model] -> [batch_size, query_seq_len, d_model]
        output = self.w_o(attention_output)

        logger.debug(f"多头注意力前向传播: query_shape={query.shape}, key_shape={key.shape}, 输出shape={output.shape}")
        return output, attention_weights

    def _scaled_dot_product_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        缩放点积注意力

        Args:
            Q: 查询 [batch_size, n_heads, query_seq_len, d_k]
            K: 键 [batch_size, n_heads, key_seq_len, d_k]
            V: 值 [batch_size, n_heads, key_seq_len, d_k]
            mask: 掩码 [batch_size, query_seq_len, key_seq_len] 或 [query_seq_len, key_seq_len]

        Returns:
            output: [batch_size, n_heads, query_seq_len, d_k]
            attention_weights: [batch_size, n_heads, query_seq_len, key_seq_len]
        """
        # 计算注意力分数
        # Q: [batch_size, n_heads, query_seq_len, d_k]
        # K.transpose(-2, -1): [batch_size, n_heads, d_k, key_seq_len]
        # scores: [batch_size, n_heads, query_seq_len, key_seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用掩码
        if mask is not None:
            # 确保mask维度正确
            if mask.dim() == 2:  # [query_seq_len, key_seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, query_seq_len, key_seq_len]
            elif mask.dim() == 3:  # [batch_size, query_seq_len, key_seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, query_seq_len, key_seq_len]

            # 将掩码位置设为很小的值
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        # [batch_size, n_heads, query_seq_len, key_seq_len]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        # attention_weights: [batch_size, n_heads, query_seq_len, key_seq_len]
        # V: [batch_size, n_heads, key_seq_len, d_k]
        # output: [batch_size, n_heads, query_seq_len, d_k]
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class FeedForward(nn.Module):
    """前馈网络

    两层全连接网络，中间使用ReLU激活

    数据流转：
    输入: [batch_size, seq_len, d_model]
    第一层: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_ff]
    激活: ReLU
    第二层: [batch_size, seq_len, d_ff] -> [batch_size, seq_len, d_model]
    输出: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        初始化前馈网络

        Args:
            d_model: 模型维度
            d_ff: 前馈网络中间层维度
            dropout: dropout概率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        logger.debug(f"前馈网络初始化完成: d_model={d_model}, d_ff={d_ff}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入tensor [batch_size, seq_len, d_model]

        Returns:
            输出tensor [batch_size, seq_len, d_model]
        """
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_ff]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # [batch_size, seq_len, d_ff] -> [batch_size, seq_len, d_model]
        x = self.linear2(x)

        logger.debug(f"前馈网络前向传播: 输入shape到输出shape保持不变")
        return x


class LayerNorm(nn.Module):
    """层归一化

    对最后一个维度进行归一化，稳定训练过程

    数据流转：
    输入: [batch_size, seq_len, d_model]
    归一化: 对d_model维度计算均值和方差
    输出: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        初始化层归一化

        Args:
            d_model: 模型维度
            eps: 防止除零的小值
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(d_model))  # 偏移参数
        self.eps = eps

        logger.debug(f"层归一化初始化完成: d_model={d_model}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入tensor [batch_size, seq_len, d_model]

        Returns:
            归一化后的tensor [batch_size, seq_len, d_model]
        """
        # 计算最后一个维度的均值和方差
        mean = x.mean(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [batch_size, seq_len, 1]

        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # [batch_size, seq_len, d_model]

        # 应用缩放和偏移
        output = self.gamma * x_norm + self.beta  # [batch_size, seq_len, d_model]

        logger.debug(f"层归一化前向传播: shape保持不变")
        return output


class EncoderLayer(nn.Module):
    """Transformer编码器层

    包含多头自注意力和前馈网络，每个子层都有残差连接和层归一化

    数据流转：
    输入: [batch_size, seq_len, d_model]

    1. 多头自注意力:
       输入 -> 多头注意力 -> [batch_size, seq_len, d_model]
       残差连接: 输入 + 注意力输出 -> [batch_size, seq_len, d_model]
       层归一化: [batch_size, seq_len, d_model]

    2. 前馈网络:
       输入 -> 前馈网络 -> [batch_size, seq_len, d_model]
       残差连接: 输入 + 前馈输出 -> [batch_size, seq_len, d_model]
       层归一化: [batch_size, seq_len, d_model]

    输出: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        初始化编码器层

        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络中间层维度
            dropout: dropout概率
        """
        super().__init__()

        # 多头自注意力
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # 层归一化
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        logger.debug(f"编码器层初始化完成: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入tensor [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len] 或 [seq_len, seq_len]

        Returns:
            输出tensor [batch_size, seq_len, d_model]
        """
        # 1. 多头自注意力 + 残差连接 + 层归一化
        # 自注意力: query, key, value都是同一个输入x
        attn_output, _ = self.self_attention(x, x, x, mask)  # [batch_size, seq_len, d_model]
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)  # 残差连接 + 层归一化

        # 2. 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)  # [batch_size, seq_len, d_model]
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)  # 残差连接 + 层归一化

        logger.debug(f"编码器层前向传播: shape保持不变")
        return x


class DecoderLayer(nn.Module):
    """Transformer解码器层

    包含掩码多头自注意力、编码器-解码器注意力和前馈网络
    每个子层都有残差连接和层归一化

    数据流转：
    输入:
    - x: [batch_size, tgt_seq_len, d_model] (解码器输入)
    - encoder_output: [batch_size, src_seq_len, d_model] (编码器输出)

    1. 掩码多头自注意力:
       x -> 自注意力(带掩码) -> [batch_size, tgt_seq_len, d_model]
       残差连接 + 层归一化

    2. 编码器-解码器注意力:
       query: x, key/value: encoder_output
       -> 交叉注意力 -> [batch_size, tgt_seq_len, d_model]
       残差连接 + 层归一化

    3. 前馈网络:
       -> 前馈网络 -> [batch_size, tgt_seq_len, d_model]
       残差连接 + 层归一化

    输出: [batch_size, tgt_seq_len, d_model]
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        初始化解码器层

        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络中间层维度
            dropout: dropout概率
        """
        super().__init__()

        # 掩码多头自注意力
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 编码器-解码器注意力
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # 层归一化
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        logger.debug(f"解码器层初始化完成: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 解码器输入 [batch_size, tgt_seq_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
            self_attn_mask: 自注意力掩码 [tgt_seq_len, tgt_seq_len] (下三角掩码)
            cross_attn_mask: 交叉注意力掩码 [batch_size, src_seq_len] (padding掩码)

        Returns:
            输出tensor [batch_size, tgt_seq_len, d_model]
        """
        # 1. 掩码多头自注意力 + 残差连接 + 层归一化
        # 自注意力: query, key, value都是解码器输入x
        self_attn_output, _ = self.self_attention(x, x, x, self_attn_mask)
        self_attn_output = self.dropout(self_attn_output)
        x = self.norm1(x + self_attn_output)  # [batch_size, tgt_seq_len, d_model]

        # 2. 编码器-解码器注意力 + 残差连接 + 层归一化
        # query: 解码器输出, key/value: 编码器输出
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask)
        cross_attn_output = self.dropout(cross_attn_output)
        x = self.norm2(x + cross_attn_output)  # [batch_size, tgt_seq_len, d_model]

        # 3. 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm3(x + ff_output)  # [batch_size, tgt_seq_len, d_model]

        logger.debug(f"解码器层前向传播: 输入shape={x.shape}, 输出shape保持不变")
        return x


def create_padding_mask(seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    创建padding掩码

    Args:
        seq: 输入序列 [batch_size, seq_len]
        pad_token_id: padding token的ID

    Returns:
        掩码 [batch_size, seq_len] (1表示有效token，0表示padding)
    """
    return (seq != pad_token_id).long()


def create_look_ahead_mask(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    创建前瞻掩码(下三角掩码)，用于解码器自注意力

    Args:
        size: 序列长度
        device: 设备

    Returns:
        掩码 [size, size] (1表示可见，0表示掩码)
    """
    mask = torch.tril(torch.ones(size, size, device=device))
    return mask  # [size, size]


def create_combined_mask(tgt_seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    创建解码器的组合掩码(padding掩码 + 前瞻掩码)

    Args:
        tgt_seq: 目标序列 [batch_size, seq_len]
        pad_token_id: padding token的ID

    Returns:
        组合掩码 [batch_size, seq_len, seq_len]
    """
    batch_size, seq_len = tgt_seq.shape
    device = tgt_seq.device

    # 创建padding掩码 [batch_size, seq_len]
    padding_mask = create_padding_mask(tgt_seq, pad_token_id)

    # 创建前瞻掩码 [seq_len, seq_len]
    look_ahead_mask = create_look_ahead_mask(seq_len, device)

    # 组合掩码: 两个掩码都为1时才为1
    # padding_mask.unsqueeze(1): [batch_size, 1, seq_len]
    # look_ahead_mask.unsqueeze(0): [1, seq_len, seq_len]
    # 广播后: [batch_size, seq_len, seq_len]
    combined_mask = padding_mask.unsqueeze(1) * look_ahead_mask.unsqueeze(0)

    return combined_mask
