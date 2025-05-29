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

        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 计算div_term，使用基本数学运算
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 应用sin和cos函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置

        # 添加batch维度并注册为buffer
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            添加位置编码后的张量
        """
        seq_len = x.size(1)
        # 使用getattr获取位置编码并切片
        pe = getattr(self, "pe")
        pe_slice = pe[:, :seq_len, :]
        return x + pe_slice


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
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

        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

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
        缩放点积注意力

        Args:
            q: 查询张量 [batch_size, n_heads, seq_len_q, d_k]
            k: 键张量 [batch_size, n_heads, seq_len_k, d_k]
            v: 值张量 [batch_size, n_heads, seq_len_v, d_k]
            mask: 注意力掩码

        Returns:
            (注意力输出, 注意力权重)
        """
        d_k = q.size(-1)

        # 计算注意力分数: Q * K^T / sqrt(d_k)
        # 使用torch.matmul进行矩阵乘法
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # 应用掩码
        if mask is not None:
            # 将掩码位置设为很小的负数
            scores = scores.masked_fill(mask == True, -1e9)

        # 计算注意力权重
        # softmax不会改变shape
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 计算注意力输出
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
        前向传播

        Args:
            query: 查询张量 [batch_size, seq_len_q, d_model]
            key: 键张量 [batch_size, seq_len_k, d_model]
            value: 值张量 [batch_size, seq_len_v, d_model]
            mask: 注意力掩码

        Returns:
            多头注意力输出 [batch_size, seq_len_q, d_model]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)

        # 线性变换
        # 这里做这们的变换主要是为了通过qkv学习w_q,w_k,w_v 这几可学的参数
        Q = self.w_q(query)  # [batch_size, seq_len_q, d_model]
        K = self.w_k(key)  # [batch_size, seq_len_k, d_model]
        V = self.w_v(value)  # [batch_size, seq_len_v, d_model]

        # 重塑为多头形式: [batch_size, seq_len, n_heads, d_k] -> [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_v, self.n_heads, self.d_k).transpose(1, 2)

        # 调整掩码维度
        if mask is not None:
            # 确保掩码有正确的维度用于多头注意力
            # 输入掩码应该是 [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
            if mask.dim() == 4:
                # 如果已经是4维，为多头添加一个维度
                mask = mask.expand(batch_size, self.n_heads, -1, -1)
            else:
                # 否则添加必要的维度
                mask = mask.unsqueeze(1).expand(batch_size, self.n_heads, -1, -1)

        # 计算注意力
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        # attention_output: [batch_size, n_heads, seq_len_q, d_k]

        # 重塑回原始形状: [batch_size, n_heads, seq_len_q, d_k] -> [batch_size, seq_len_q, d_model]
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, self.d_model)
        )

        # 最终线性变换
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
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            前馈网络输出
        """
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


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
            解码器层输出
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
        创建注意力掩码

        Args:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]

        Returns:
            (编码器掩码, 解码器自注意力掩码, 解码器交叉注意力掩码)
        """
        tgt_len = tgt.size(1)

        # 编码器填充掩码
        encoder_mask = (
            (src == self.pad_id).unsqueeze(1).unsqueeze(2)
        )  # [batch_size, 1, 1, src_len]

        # 解码器填充掩码
        decoder_padding_mask = (
            (tgt == self.pad_id).unsqueeze(1).unsqueeze(2)
        )  # [batch_size, 1, 1, tgt_len]

        # 解码器前瞻掩码
        look_ahead_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1
        ).bool()
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(
            0
        )  # [1, 1, tgt_len, tgt_len]

        # 解码器自注意力掩码 = 填充掩码 + 前瞻掩码
        decoder_self_mask = decoder_padding_mask | look_ahead_mask

        # 解码器交叉注意力掩码（只有编码器的填充掩码）
        decoder_cross_mask = encoder_mask

        return encoder_mask, decoder_self_mask, decoder_cross_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]

        Returns:
            输出logits [batch_size, tgt_len, vocab_size_it]
        """
        # 创建掩码
        encoder_mask, decoder_self_mask, decoder_cross_mask = self.create_masks(
            src, tgt
        )

        # 编码器
        encoder_output = self.encoder(src, encoder_mask)

        # 解码器
        decoder_output = self.decoder(
            tgt, encoder_output, decoder_self_mask, decoder_cross_mask
        )

        # 输出投影
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
