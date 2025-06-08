"""
Transformer模型 - 完整的Transformer架构实现
包括：编码器、解码器、注意力机制、前馈网络等
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from config import ModelConfig


class PositionalEncoding(nn.Module):
    """
    位置编码 - 为序列添加位置信息

    数据流转过程：
    1. 输入: token embeddings [batch_size, seq_len, d_model]
    2. 获取预计算的位置编码 [1, max_seq_len, d_model]
    3. 切片位置编码到当前序列长度 [1, seq_len, d_model]
    4. 广播相加: embeddings + position_encoding
    5. 输出: 带位置信息的embeddings [batch_size, seq_len, d_model]
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
        self.max_seq_len = max_seq_len

        # 创建位置编码矩阵 [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)

        # 位置索引 [max_seq_len, 1] - 每个位置的索引值
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 计算频率项 [d_model//2] - 用于sin/cos的频率
        # div_term = 1 / (10000^(2i/d_model)) for i in [0, 1, 2, ..., d_model//2-1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 应用sin和cos函数生成位置编码
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos

        # 添加batch维度 [1, max_seq_len, d_model] 并注册为buffer（不参与梯度更新）
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 将位置编码添加到输入embeddings

        Args:
            x: 输入张量 [batch_size, seq_len, d_model] - token embeddings

        Returns:
            添加位置编码后的张量 [batch_size, seq_len, d_model]

        数据流转详解：
        1. x.shape = [batch_size, seq_len, d_model] (输入embeddings)
        2. pe.shape = [1, max_seq_len, d_model] (预计算的位置编码)
        3. pe_slice.shape = [1, seq_len, d_model] (切片到当前序列长度)
        4. 广播相加: x + pe_slice -> [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)

        # 获取位置编码并切片到当前序列长度
        # pe: [1, max_seq_len, d_model] -> pe_slice: [1, seq_len, d_model]
        pe = getattr(self, "pe")
        pe_slice = pe[:, :seq_len, :]

        # 广播相加: [batch_size, seq_len, d_model] + [1, seq_len, d_model]
        # -> [batch_size, seq_len, d_model]
        return x + pe_slice


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    数据流转过程：
    1. 输入: query, key, value [batch_size, seq_len, d_model]
    2. 线性变换: Q, K, V [batch_size, seq_len, d_model]
    3. 重塑为多头: [batch_size, n_heads, seq_len, d_k]
    4. 计算注意力: scaled dot-product attention
    5. 合并多头: [batch_size, seq_len, d_model]
    6. 输出投影: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力

        Args:
            d_model: 模型维度 (例如: 512)
            n_heads: 注意力头数 (例如: 8)
            dropout: dropout概率
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度 (例如: 512//8=64)

        # 线性变换层 - 将输入投影到Q, K, V空间
        # 输入: [batch_size, seq_len, d_model] -> 输出: [batch_size, seq_len, d_model]
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Query投影
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Key投影
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Value投影
        self.w_o = nn.Linear(d_model, d_model)              # 输出投影

        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        缩放点积注意力 - 注意力机制的核心计算

        公式: Attention(Q,K,V) = softmax(QK^T/√d_k)V

        Args:
            q: 查询张量 [batch_size, n_heads, seq_len_q, d_k]
            k: 键张量 [batch_size, n_heads, seq_len_k, d_k]
            v: 值张量 [batch_size, n_heads, seq_len_v, d_k]
            mask: 注意力掩码 [batch_size, n_heads, seq_len_q, seq_len_k] (可选)

        Returns:
            (注意力输出 [batch_size, n_heads, seq_len_q, d_k],
             注意力权重 [batch_size, n_heads, seq_len_q, seq_len_k])

        数据流转详解:
        1. q: [batch_size, n_heads, seq_len_q, d_k] (查询)
        2. k: [batch_size, n_heads, seq_len_k, d_k] (键)
        3. k.transpose(-2,-1): [batch_size, n_heads, d_k, seq_len_k] (转置键)
        4. scores = q @ k^T: [batch_size, n_heads, seq_len_q, seq_len_k] (注意力分数)
        5. scores / √d_k: 缩放防止梯度消失
        6. 应用mask: 将无效位置设为-∞
        7. softmax(scores): [batch_size, n_heads, seq_len_q, seq_len_k] (注意力权重)
        8. attention_weights @ v: [batch_size, n_heads, seq_len_q, d_k] (加权值)
        """
        d_k = q.size(-1)  # 获取键的维度用于缩放

        # 步骤1: 计算注意力分数 QK^T/√d_k
        # q: [batch_size, n_heads, seq_len_q, d_k]
        # k.transpose(-2, -1): [batch_size, n_heads, d_k, seq_len_k]
        # scores: [batch_size, n_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # 步骤2: 应用掩码 (如果提供)
        if mask is not None:
            # mask中True的位置表示需要被掩盖的位置
            # 将这些位置设为很小的负数，softmax后会接近0
            scores = scores.masked_fill(mask == True, -1e9)

        # 步骤3: 计算注意力权重 (softmax归一化)
        # scores: [batch_size, n_heads, seq_len_q, seq_len_k]
        # attention_weights: [batch_size, n_heads, seq_len_q, seq_len_k]
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 步骤4: 计算加权输出
        # attention_weights: [batch_size, n_heads, seq_len_q, seq_len_k]
        # v: [batch_size, n_heads, seq_len_v, d_k] (注意: seq_len_v = seq_len_k)
        # output: [batch_size, n_heads, seq_len_q, d_k]
        output = torch.matmul(attention_weights, v)

        return output, attention_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        多头注意力前向传播

        Args:
            query: 查询张量 [batch_size, seq_len_q, d_model]
            key: 键张量 [batch_size, seq_len_k, d_model]
            value: 值张量 [batch_size, seq_len_v, d_model]
            mask: 注意力掩码 [batch_size, 1, seq_len_q, seq_len_k] (可选)

        Returns:
            多头注意力输出 [batch_size, seq_len_q, d_model]

        数据流转详解:
        1. 输入: query, key, value [batch_size, seq_len, d_model]
        2. 线性变换: Q, K, V [batch_size, seq_len, d_model]
        3. 重塑为多头: [batch_size, n_heads, seq_len, d_k]
        4. 计算注意力: [batch_size, n_heads, seq_len_q, d_k]
        5. 合并多头: [batch_size, seq_len_q, d_model]
        6. 输出投影: [batch_size, seq_len_q, d_model]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)

        # 步骤1: 线性变换生成Q, K, V
        # 通过可学习的权重矩阵将输入投影到查询、键、值空间
        Q = self.w_q(query)  # [batch_size, seq_len_q, d_model]
        K = self.w_k(key)    # [batch_size, seq_len_k, d_model]
        V = self.w_v(value)  # [batch_size, seq_len_v, d_model]

        # 步骤2: 重塑为多头形式
        # 将d_model维度分割为n_heads个d_k维度的头
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k] -> [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_v, self.n_heads, self.d_k).transpose(1, 2)

        # 步骤3: 调整掩码维度以匹配多头注意力
        if mask is not None:
            # 输入mask通常是 [batch_size, 1, seq_len_q, seq_len_k]
            # 需要扩展为 [batch_size, n_heads, seq_len_q, seq_len_k]
            if mask.dim() == 4:
                # 如果已经是4维，为每个头复制掩码
                mask = mask.expand(batch_size, self.n_heads, -1, -1)
            else:
                # 否则添加必要的维度
                mask = mask.unsqueeze(1).expand(batch_size, self.n_heads, -1, -1)

        # 步骤4: 计算缩放点积注意力
        # Q, K, V: [batch_size, n_heads, seq_len, d_k]
        # attention_output: [batch_size, n_heads, seq_len_q, d_k]
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        # 步骤5: 合并多头输出
        # [batch_size, n_heads, seq_len_q, d_k] -> [batch_size, seq_len_q, n_heads, d_k] -> [batch_size, seq_len_q, d_model]
        attention_output = (
            attention_output.transpose(1, 2)  # 交换n_heads和seq_len_q维度
            .contiguous()                     # 确保内存连续
            .view(batch_size, seq_len_q, self.d_model)  # 重塑为原始形状
        )

        # 步骤6: 最终线性变换
        # [batch_size, seq_len_q, d_model] -> [batch_size, seq_len_q, d_model]
        output = self.w_o(attention_output)

        return output


class FeedForward(nn.Module):
    """
    前馈网络
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        初始化前馈网络

        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度
            dropout: dropout概率
        """
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前馈网络前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            输出张量 [batch_size, seq_len, d_model]

        数据流转详解:
        1. x: [batch_size, seq_len, d_model] (输入)
        2. linear1(x): [batch_size, seq_len, d_ff] (第一层线性变换，扩展维度)
        3. relu(): [batch_size, seq_len, d_ff] (ReLU激活函数)
        4. dropout(): [batch_size, seq_len, d_ff] (随机失活)
        5. linear2(): [batch_size, seq_len, d_model] (第二层线性变换，恢复维度)

        公式: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
        """
        # 步骤1: 第一层线性变换 + ReLU激活
        # x: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_ff]
        hidden = torch.relu(self.linear1(x))

        # 步骤2: Dropout
        # [batch_size, seq_len, d_ff] -> [batch_size, seq_len, d_ff]
        hidden = self.dropout(hidden)

        # 步骤3: 第二层线性变换
        # [batch_size, seq_len, d_ff] -> [batch_size, seq_len, d_model]
        output = self.linear2(hidden)

        return output


class LayerNorm(nn.Module):
    """
    层归一化
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        初始化层归一化

        Args:
            d_model: 模型维度
            eps: 数值稳定性参数
        """
        super().__init__()
        # 在nn.LayerNorm这个方法中对应参数：elementwise_affine进行控制，默认为True
        # 会创建两个可学习参数weight和bias,就是这里的gamma和beta
        self.gamma = nn.Parameter(torch.ones(d_model))  # 可以理解成每个特征维总体均值
        self.beta = nn.Parameter(torch.zeros(d_model))  # 可以理解成每个特征维总体方差
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            归一化后的张量
        """
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 归一化
        normalized = (x - mean) / torch.sqrt(var + self.eps)

        # 缩放和平移
        return self.gamma * normalized + self.beta


class AddNorm(nn.Module):
    """
    残差连接 + 层归一化 (Add & Norm)
    实现 Transformer 中的 AddNorm 模块
    """

    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-6):
        """
        初始化 AddNorm 模块

        Args:
            d_model: 模型维度
            dropout: dropout概率
            eps: 层归一化的数值稳定性参数
        """
        super().__init__()

        self.layer_norm = LayerNorm(d_model, eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        前向传播：残差连接 + 层归一化

        Args:
            x: 原始输入张量 [batch_size, seq_len, d_model]
            sublayer_output: 子层输出张量 [batch_size, seq_len, d_model]
            # TODO 如何理解这里的sublayer_output

        Returns:
            AddNorm 输出 [batch_size, seq_len, d_model]
        """
        # 残差连接 + dropout + 层归一化
        # 使用 Post-LN 方式: LayerNorm(x + Sublayer(x))
        return self.layer_norm(x + self.dropout(sublayer_output))


class EncoderLayer(nn.Module):
    """
    编码器层 - 使用 AddNorm 模块
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        初始化编码器层

        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: dropout概率
        """
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # 使用 AddNorm 模块替代单独的 LayerNorm 和残差连接
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码

        Returns:
            编码器层输出
        """
        # 自注意力 + AddNorm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.add_norm1(x, attn_output)

        # 前馈网络 + AddNorm
        ff_output = self.feed_forward(x)
        x = self.add_norm2(x, ff_output)

        return x


class DecoderLayer(nn.Module):
    """
    解码器层 - 使用 AddNorm 模块
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        初始化解码器层

        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: dropout概率
        """
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # 使用 AddNorm 模块替代单独的 LayerNorm 和残差连接
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

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
            x: 解码器输入 [batch_size, seq_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            self_attn_mask: 自注意力掩码
            cross_attn_mask: 交叉注意力掩码

        Returns:
            解码器层输出  [batch_size, seq_len, d_model]
        """
        # 自注意力 + AddNorm
        self_attn_output = self.self_attention(x, x, x, self_attn_mask)
        x = self.add_norm1(x, self_attn_output)

        # 交叉注意力 + AddNorm
        cross_attn_output = self.cross_attention(
            x, encoder_output, encoder_output, cross_attn_mask
        )
        x = self.add_norm2(x, cross_attn_output)

        # 前馈网络 + AddNorm
        ff_output = self.feed_forward(x)
        x = self.add_norm3(x, ff_output)

        return x


class Encoder(nn.Module):
    """
    编码器
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        """
        初始化编码器

        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: 编码器层数
            d_ff: 前馈网络维度
            max_seq_len: 最大序列长度
            dropout: dropout概率
        """
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # 初始化embedding权重
        # TODO 一定要在这里手动初始化吗
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入token序列 [batch_size, seq_len]
            mask: 填充掩码

        Returns:
            编码器输出 [batch_size, seq_len, d_model]
        """
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 通过编码器层
        for layer in self.layers:
            x = layer(x, mask)

        return x


class Decoder(nn.Module):
    """
    解码器
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        """
        初始化解码器

        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: 解码器层数
            d_ff: 前馈网络维度
            max_seq_len: 最大序列长度
            dropout: dropout概率
        """
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # 初始化embedding权重
        nn.init.xavier_uniform_(self.embedding.weight)

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
            x: 解码器输入token序列 [batch_size, seq_len]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            self_attn_mask: 自注意力掩码
            cross_attn_mask: 交叉注意力掩码

        Returns:
            解码器输出 [batch_size, seq_len, d_model]
        """
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 通过解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)

        return x


class Transformer(nn.Module):
    """
    完整的Transformer模型
    """

    def __init__(self, config: ModelConfig):
        """
        初始化Transformer模型

        Args:
            config: 模型配置
        """
        super().__init__()

        self.config = config

        # 编码器和解码器
        self.encoder = Encoder(
            vocab_size=config.vocab_size_en,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        self.decoder = Decoder(
            vocab_size=config.vocab_size_it,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # 输出投影层
        self.output_projection = nn.Linear(config.d_model, config.vocab_size_it)

        # 初始化输出层权重
        nn.init.xavier_uniform_(self.output_projection.weight)

        # 特殊token ID
        self.pad_id = 0

    def create_masks(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建注意力掩码 - 详细解释各种mask的计算过程

        Args:
            src: 源序列 [batch_size, src_len] - 编码器输入token序列
            tgt: 目标序列 [batch_size, tgt_len] - 解码器输入token序列

        Returns:
            (编码器掩码 [batch_size, 1, 1, src_len],
             解码器自注意力掩码 [batch_size, 1, tgt_len, tgt_len],
             解码器交叉注意力掩码 [batch_size, 1, 1, src_len])

        Mask计算详解:

        1. 编码器填充掩码 (encoder_mask):
           - 目的: 防止注意力机制关注到填充token (PAD)
           - 计算: src == pad_id -> [batch_size, src_len] -> [batch_size, 1, 1, src_len]
           - 作用: 在编码器自注意力中屏蔽PAD位置

        2. 解码器自注意力掩码 (decoder_self_mask):
           - 包含两部分: 填充掩码 + 前瞻掩码
           - 填充掩码: 屏蔽PAD token
           - 前瞻掩码: 防止解码器看到未来的token (因果性)
           - 计算过程详见下方

        3. 解码器交叉注意力掩码 (decoder_cross_mask):
           - 目的: 防止解码器关注编码器输出中的PAD位置
           - 与编码器掩码相同
        """
        batch_size = src.size(0)
        src_len = src.size(1)
        tgt_len = tgt.size(1)

        # ========== 1. 编码器填充掩码 ==========
        # 找出源序列中的填充位置
        # src: [batch_size, src_len] -> encoder_mask: [batch_size, 1, 1, src_len]
        encoder_mask = (src == self.pad_id).unsqueeze(1).unsqueeze(2)

        # 示例: 如果src = [[1, 2, 0, 0], [3, 4, 5, 0]] (0是PAD)
        # 则encoder_mask = [[[False, False, True, True]], [[False, False, False, True]]]

        # ========== 2. 解码器填充掩码 ==========
        # 找出目标序列中的填充位置
        # tgt: [batch_size, tgt_len] -> decoder_padding_mask: [batch_size, 1, 1, tgt_len]
        decoder_padding_mask = (tgt == self.pad_id).unsqueeze(1).unsqueeze(2)

        # ========== 3. 解码器前瞻掩码 (Look-ahead Mask) ==========
        # 创建上三角矩阵，防止解码器看到未来的token
        # torch.triu创建上三角矩阵，diagonal=1表示主对角线上方的元素为1
        look_ahead_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1
        ).bool()

        # 示例: 如果tgt_len=4，则look_ahead_mask为:
        # [[False, True,  True,  True ],
        #  [False, False, True,  True ],
        #  [False, False, False, True ],
        #  [False, False, False, False]]
        # 这确保了位置i只能看到位置<=i的token

        # 添加batch和head维度: [tgt_len, tgt_len] -> [1, 1, tgt_len, tgt_len]
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)

        # ========== 4. 解码器自注意力掩码 ==========
        # 结合填充掩码和前瞻掩码
        # 需要广播: decoder_padding_mask [batch_size, 1, 1, tgt_len]
        #          + look_ahead_mask [1, 1, tgt_len, tgt_len]
        #          -> decoder_self_mask [batch_size, 1, tgt_len, tgt_len]
        decoder_self_mask = decoder_padding_mask | look_ahead_mask

        # 解释:
        # - decoder_padding_mask会在每一行的PAD位置设为True
        # - look_ahead_mask会在上三角位置设为True
        # - 两者OR操作确保既屏蔽PAD又屏蔽未来token

        # ========== 5. 解码器交叉注意力掩码 ==========
        # 解码器关注编码器输出时，只需要屏蔽编码器的PAD位置
        decoder_cross_mask = encoder_mask

        return encoder_mask, decoder_self_mask, decoder_cross_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Transformer前向传播 - 完整的编码-解码过程

        Args:
            src: 源序列 [batch_size, src_len] - 编码器输入token ID序列
            tgt: 目标序列 [batch_size, tgt_len] - 解码器输入token ID序列

        Returns:
            输出logits [batch_size, tgt_len, vocab_size_it] - 每个位置的词汇表概率分布

        数据流转详解:
        1. 输入: src [batch_size, src_len], tgt [batch_size, tgt_len]
        2. 创建掩码: encoder_mask, decoder_self_mask, decoder_cross_mask
        3. 编码阶段: src -> encoder_output [batch_size, src_len, d_model]
        4. 解码阶段: tgt + encoder_output -> decoder_output [batch_size, tgt_len, d_model]
        5. 输出投影: decoder_output -> logits [batch_size, tgt_len, vocab_size_it]
        """
        # 步骤1: 创建注意力掩码
        # encoder_mask: [batch_size, 1, 1, src_len] - 编码器填充掩码
        # decoder_self_mask: [batch_size, 1, tgt_len, tgt_len] - 解码器自注意力掩码
        # decoder_cross_mask: [batch_size, 1, 1, src_len] - 解码器交叉注意力掩码
        encoder_mask, decoder_self_mask, decoder_cross_mask = self.create_masks(
            src, tgt
        )

        # 步骤2: 编码阶段
        # src: [batch_size, src_len] -> encoder_output: [batch_size, src_len, d_model]
        encoder_output = self.encoder(src, encoder_mask)

        # 步骤3: 解码阶段
        # tgt: [batch_size, tgt_len] + encoder_output: [batch_size, src_len, d_model]
        # -> decoder_output: [batch_size, tgt_len, d_model]
        decoder_output = self.decoder(
            tgt, encoder_output, decoder_self_mask, decoder_cross_mask
        )

        # 步骤4: 输出投影
        # decoder_output: [batch_size, tgt_len, d_model] -> output: [batch_size, tgt_len, vocab_size_it]
        output = self.output_projection(decoder_output)

        return output

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        仅编码（用于推理）

        Args:
            src: 源序列 [batch_size, src_len]

        Returns:
            编码器输出 [batch_size, src_len, d_model]
        """
        encoder_mask = (src == self.pad_id).unsqueeze(1).unsqueeze(2)
        return self.encoder(src, encoder_mask)

    def decode_step(
        self, tgt: torch.Tensor, encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        解码一步（用于推理）

        Args:
            tgt: 目标序列 [batch_size, tgt_len]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]

        Returns:
            输出logits [batch_size, tgt_len, vocab_size_it]
        """
        batch_size, tgt_len = tgt.size()
        src_len = encoder_output.size(1)

        # 创建掩码
        decoder_padding_mask = (tgt == self.pad_id).unsqueeze(1).unsqueeze(2)
        look_ahead_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1
        ).bool()
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)
        decoder_self_mask = decoder_padding_mask | look_ahead_mask

        # 交叉注意力掩码（假设编码器输出没有填充）
        decoder_cross_mask = torch.zeros(
            batch_size, 1, 1, src_len, device=tgt.device
        ).bool()

        # 解码
        decoder_output = self.decoder(
            tgt, encoder_output, decoder_self_mask, decoder_cross_mask
        )

        # 输出投影
        output = self.output_projection(decoder_output)

        return output
