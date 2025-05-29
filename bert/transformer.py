"""
Transformer基础组件模块
包含多头自注意力、前馈网络、层归一化等核心组件
重点关注mask的创建和使用逻辑，包含详细的shape注释
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger("BERT")


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制 - BERT的核心组件

    数据流：
    input: (batch_size, seq_len, d_model)
    -> Q,K,V: (batch_size, seq_len, d_model)
    -> reshape: (batch_size, n_heads, seq_len, d_k)
    -> attention: (batch_size, n_heads, seq_len, seq_len)
    -> output: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # Q、K、V线性变换层
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _create_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        创建注意力掩码 - 重点解释mask逻辑

        Args:
            attention_mask: 原始掩码 (batch_size, seq_len)
                           1表示真实token，0表示padding token

        Returns:
            extended_mask: 扩展掩码 (batch_size, n_heads, seq_len, seq_len)
                          0表示可以注意，-10000表示不能注意

        Mask逻辑详解：
        1. 原始mask: [1, 1, 1, 0, 0] (前3个是真实token，后2个是padding)
        2. 扩展为4D: (batch_size, 1, 1, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
        3. 转换值: 1->0(可注意), 0->-10000(不可注意)
        4. 在softmax中，-10000会变成接近0的概率
        """
        batch_size, seq_len = attention_mask.shape

        # 步骤1: 扩展维度 (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 步骤2: 广播到所有注意力头 (batch_size, n_heads, seq_len, seq_len)
        extended_mask = extended_mask.expand(batch_size, self.n_heads, seq_len, seq_len)

        # 步骤3: 转换掩码值
        # 原来: 1(真实token) -> 0(可以注意)
        # 原来: 0(padding) -> -10000(不能注意，softmax后接近0)
        extended_mask = (1.0 - extended_mask) * -10000.0

        return extended_mask

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,  # (batch_size, n_heads, seq_len, d_k)
        k: torch.Tensor,  # (batch_size, n_heads, seq_len, d_k)
        v: torch.Tensor,  # (batch_size, n_heads, seq_len, d_k)
        mask: Optional[torch.Tensor] = None,  # (batch_size, n_heads, seq_len, seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        缩放点积注意力计算

        Args:
            q: 查询矩阵 (batch_size, n_heads, seq_len, d_k)
            k: 键矩阵 (batch_size, n_heads, seq_len, d_k)
            v: 值矩阵 (batch_size, n_heads, seq_len, d_k)
            mask: 注意力掩码 (batch_size, n_heads, seq_len, seq_len)

        Returns:
            output: 注意力输出 (batch_size, n_heads, seq_len, d_k)
            attention_weights: 注意力权重 (batch_size, n_heads, seq_len, seq_len)

        计算步骤：
        1. scores = Q @ K^T / sqrt(d_k)  # 计算相似度分数
        2. scores += mask               # 应用掩码（padding位置设为-10000）
        3. weights = softmax(scores)    # 计算注意力权重
        4. output = weights @ V         # 加权求和得到输出
        """
        d_k = q.size(-1)

        # 步骤1: 计算注意力分数 Q @ K^T / sqrt(d_k)
        # (batch_size, n_heads, seq_len, d_k) @ (batch_size, n_heads, d_k, seq_len)
        # -> (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # 步骤2: 应用掩码（如果有）
        if mask is not None:
            scores = scores + mask  # padding位置变成-10000

        # 步骤3: 计算注意力权重
        # softmax会将-10000变成接近0的概率，实现掩码效果
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, n_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)

        # 步骤4: 加权求和
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_k)
        # -> (batch_size, n_heads, seq_len, d_k)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, seq_len, d_model)
        attention_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            output_attentions: 是否输出注意力权重

        Returns:
            output: 注意力输出 (batch_size, seq_len, d_model)
            attention_probs: 注意力权重 (batch_size, n_heads, seq_len, seq_len) 或 None
        """
        batch_size, seq_len, _ = x.shape

        # 步骤1: 线性变换得到Q、K、V
        Q = self.w_q(x)  # (batch_size, seq_len, d_model)
        K = self.w_k(x)  # (batch_size, seq_len, d_model)
        V = self.w_v(x)  # (batch_size, seq_len, d_model)

        # 步骤2: 重塑为多头形式
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 步骤3: 处理注意力掩码
        if attention_mask is not None:
            attention_mask = self._create_attention_mask(attention_mask)

        # 步骤4: 计算注意力
        attention_output, attention_probs = self.scaled_dot_product_attention(Q, K, V, attention_mask)

        # 步骤5: 重塑回原始形状
        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 步骤6: 最终线性变换
        output = self.w_o(attention_output)  # (batch_size, seq_len, d_model)

        if output_attentions:
            return output, attention_probs
        else:
            return output, None


class FeedForward(nn.Module):
    """
    前馈网络 - Transformer中的位置前馈网络

    数据流：
    input: (batch_size, seq_len, d_model)
    -> linear1: (batch_size, seq_len, d_ff)
    -> activation: (batch_size, seq_len, d_ff)
    -> dropout: (batch_size, seq_len, d_ff)
    -> linear2: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        # 初始化权重
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, d_model)

        Returns:
            output: 输出张量 (batch_size, seq_len, d_model)
        """
        # FFN(x) = linear2(dropout(activation(linear1(x))))
        x = self.linear1(x)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = self.activation(x)  # (batch_size, seq_len, d_ff)
        x = self.dropout(x)  # (batch_size, seq_len, d_ff)
        x = self.linear2(x)  # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return x


class LayerNorm(nn.Module):
    """
    层归一化 - 自定义实现以便更好地理解

    数据流：
    input: (batch_size, seq_len, d_model)
    -> normalize: (batch_size, seq_len, d_model)
    -> scale_shift: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model: int, eps: float = 1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(d_model))  # 偏移参数
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, d_model)

        Returns:
            output: 归一化后的张量 (batch_size, seq_len, d_model)
        """
        # 在最后一个维度上计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (batch_size, seq_len, 1)

        # 归一化
        normalized = (x - mean) / torch.sqrt(var + self.eps)  # (batch_size, seq_len, d_model)

        # 缩放和平移
        return self.gamma * normalized + self.beta  # (batch_size, seq_len, d_model)


class AddNorm(nn.Module):
    """
    残差连接 + 层归一化 (Add & Norm)

    数据流：
    residual: (batch_size, seq_len, d_model)
    sublayer_output: (batch_size, seq_len, d_model)
    -> add: (batch_size, seq_len, d_model)
    -> norm: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-12):
        super().__init__()
        self.layer_norm = LayerNorm(d_model, eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        前向传播：残差连接 + 层归一化

        Args:
            x: 原始输入张量 (batch_size, seq_len, d_model)
            sublayer_output: 子层输出张量 (batch_size, seq_len, d_model)

        Returns:
            output: AddNorm输出 (batch_size, seq_len, d_model)
        """
        # 残差连接 + dropout + 层归一化
        return self.layer_norm(x + self.dropout(sublayer_output))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层 - BERT的基础构建块

    数据流：
    input: (batch_size, seq_len, d_model)
    -> self_attention: (batch_size, seq_len, d_model)
    -> add_norm1: (batch_size, seq_len, d_model)
    -> feed_forward: (batch_size, seq_len, d_model)
    -> add_norm2: (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        # 使用AddNorm模块
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, seq_len, d_model)
        attention_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            output_attentions: 是否输出注意力权重

        Returns:
            output: 编码器层输出 (batch_size, seq_len, d_model)
            attention_probs: 注意力权重 (batch_size, n_heads, seq_len, seq_len) 或 None
        """
        # 步骤1: 自注意力 + AddNorm
        attn_output, attention_probs = self.self_attention(x, attention_mask, output_attentions)
        x = self.add_norm1(x, attn_output)  # (batch_size, seq_len, d_model)

        # 步骤2: 前馈网络 + AddNorm
        ff_output = self.feed_forward(x)
        x = self.add_norm2(x, ff_output)  # (batch_size, seq_len, d_model)

        return x, attention_probs
