"""
Transformer模型定义 - 重构版本
包含完整的Transformer编码器-解码器架构
详细的数据流转注释和shape说明
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple

from config import TRANSFORMER_CONFIG
from transformer import (
    PositionalEncoding,
    EncoderLayer,
    DecoderLayer,
    create_padding_mask,
    create_combined_mask,
)

logger = logging.getLogger("Transformer2")


class TransformerEncoder(nn.Module):
    """Transformer编码器

    由多个编码器层堆叠而成，处理源序列

    数据流转：
    输入: [batch_size, src_seq_len] (token IDs)

    1. 词嵌入: [batch_size, src_seq_len] -> [batch_size, src_seq_len, d_model]
    2. 位置编码: [batch_size, src_seq_len, d_model] -> [batch_size, src_seq_len, d_model]
    3. 编码器层(N层): [batch_size, src_seq_len, d_model] -> [batch_size, src_seq_len, d_model]

    输出: [batch_size, src_seq_len, d_model]
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

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # 编码器层堆叠
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        # 权重初始化
        self._init_weights()

        logger.info(f"编码器初始化完成: vocab_size={vocab_size}, d_model={d_model}, " f"n_layers={n_layers}, n_heads={n_heads}")

    def _init_weights(self):
        """初始化权重"""
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.embedding.weight)

        # 缩放embedding权重
        self.embedding.weight.data.mul_(self.d_model**0.5)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            src: 源序列 [batch_size, src_seq_len]
            src_mask: 源序列掩码 [batch_size, src_seq_len] (可选)

        Returns:
            编码器输出 [batch_size, src_seq_len, d_model]
        """
        batch_size, src_seq_len = src.shape

        # 1. 词嵌入
        # [batch_size, src_seq_len] -> [batch_size, src_seq_len, d_model]
        x = self.embedding(src)

        # 2. 位置编码
        # [batch_size, src_seq_len, d_model] -> [batch_size, src_seq_len, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 3. 转换掩码格式用于自注意力
        # 编码器的自注意力不需要掩码（或者使用padding掩码）
        # 这里我们暂时不使用掩码，因为编码器可以看到所有位置
        encoder_mask = None

        # 4. 通过编码器层
        for i, layer in enumerate(self.layers):
            x = layer(x, encoder_mask)  # [batch_size, src_seq_len, d_model]
            logger.debug(f"编码器第{i+1}层输出shape: {x.shape}")

        logger.debug(f"编码器前向传播完成: 输入shape={src.shape}, 输出shape={x.shape}")
        return x


class TransformerDecoder(nn.Module):
    """Transformer解码器

    由多个解码器层堆叠而成，处理目标序列并结合编码器输出

    数据流转：
    输入:
    - tgt: [batch_size, tgt_seq_len] (目标序列token IDs)
    - encoder_output: [batch_size, src_seq_len, d_model] (编码器输出)

    1. 词嵌入: [batch_size, tgt_seq_len] -> [batch_size, tgt_seq_len, d_model]
    2. 位置编码: [batch_size, tgt_seq_len, d_model] -> [batch_size, tgt_seq_len, d_model]
    3. 解码器层(N层): [batch_size, tgt_seq_len, d_model] -> [batch_size, tgt_seq_len, d_model]

    输出: [batch_size, tgt_seq_len, d_model]
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

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # 解码器层堆叠
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        # 权重初始化
        self._init_weights()

        logger.info(f"解码器初始化完成: vocab_size={vocab_size}, d_model={d_model}, " f"n_layers={n_layers}, n_heads={n_heads}")

    def _init_weights(self):
        """初始化权重"""
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.embedding.weight)

        # 缩放embedding权重
        self.embedding.weight.data.mul_(self.d_model**0.5)

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            tgt: 目标序列 [batch_size, tgt_seq_len]
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
            tgt_mask: 目标序列掩码 [batch_size, tgt_seq_len, tgt_seq_len] (可选)
            src_mask: 源序列掩码 [batch_size, src_seq_len] (可选)

        Returns:
            解码器输出 [batch_size, tgt_seq_len, d_model]
        """
        batch_size, tgt_seq_len = tgt.shape

        # 1. 词嵌入
        # [batch_size, tgt_seq_len] -> [batch_size, tgt_seq_len, d_model]
        x = self.embedding(tgt)

        # 2. 位置编码
        # [batch_size, tgt_seq_len, d_model] -> [batch_size, tgt_seq_len, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 3. 处理交叉注意力掩码
        # src_mask用于交叉注意力，需要转换格式
        cross_attn_mask = None  # 暂时不使用交叉注意力掩码

        # 4. 通过解码器层
        for i, layer in enumerate(self.layers):
            x = layer(x, encoder_output, tgt_mask, cross_attn_mask)  # [batch_size, tgt_seq_len, d_model]
            logger.debug(f"解码器第{i+1}层输出shape: {x.shape}")

        logger.debug(f"解码器前向传播完成: 输入shape={tgt.shape}, 输出shape={x.shape}")
        return x


class Transformer(nn.Module):
    """完整的Transformer模型

    包含编码器、解码器和输出投影层

    数据流转：
    输入:
    - src: [batch_size, src_seq_len] (源序列)
    - tgt: [batch_size, tgt_seq_len] (目标序列)

    1. 编码器: src -> [batch_size, src_seq_len, d_model]
    2. 解码器: tgt + encoder_output -> [batch_size, tgt_seq_len, d_model]
    3. 输出投影: [batch_size, tgt_seq_len, d_model] -> [batch_size, tgt_seq_len, vocab_size]

    输出: [batch_size, tgt_seq_len, vocab_size] (logits)
    """

    def __init__(self):
        """
        使用全局配置初始化Transformer模型
        """
        super().__init__()

        config = TRANSFORMER_CONFIG

        # 编码器
        self.encoder = TransformerEncoder(
            vocab_size=config.vocab_size_src,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # 解码器
        self.decoder = TransformerDecoder(
            vocab_size=config.vocab_size_tgt,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # 输出投影层
        self.output_projection = nn.Linear(config.d_model, config.vocab_size_tgt)

        # 权重初始化
        self._init_weights()

        # 计算参数数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Transformer模型初始化完成:")
        logger.info(f"  - 总参数数量: {total_params:,}")
        logger.info(f"  - 可训练参数: {trainable_params:,}")
        logger.info(f"  - 模型配置: d_model={config.d_model}, n_layers={config.n_layers}, " f"n_heads={config.n_heads}")

    def _init_weights(self):
        """初始化权重"""
        # 输出投影层使用Xavier初始化
        nn.init.xavier_uniform_(self.output_projection.weight)

        # 如果编码器和解码器共享词汇表，可以共享embedding权重
        if TRANSFORMER_CONFIG.vocab_size_src == TRANSFORMER_CONFIG.vocab_size_tgt:
            self.decoder.embedding.weight = self.encoder.embedding.weight
            logger.info("编码器和解码器共享embedding权重")

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            src: 源序列 [batch_size, src_seq_len]
            tgt: 目标序列 [batch_size, tgt_seq_len]
            src_mask: 源序列掩码 (暂时不使用)
            tgt_mask: 目标序列掩码 [batch_size, tgt_seq_len, tgt_seq_len] (可选)

        Returns:
            输出logits [batch_size, tgt_seq_len, vocab_size]
        """
        # 1. 编码器
        # [batch_size, src_seq_len] -> [batch_size, src_seq_len, d_model]
        encoder_output = self.encoder(src, None)  # 暂时不使用源序列掩码

        # 2. 解码器
        # [batch_size, tgt_seq_len] -> [batch_size, tgt_seq_len, d_model]
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, None)  # 暂时不使用交叉注意力掩码

        # 3. 输出投影
        # [batch_size, tgt_seq_len, d_model] -> [batch_size, tgt_seq_len, vocab_size]
        logits = self.output_projection(decoder_output)

        logger.debug(f"Transformer前向传播完成: src_shape={src.shape}, tgt_shape={tgt.shape}, " f"output_shape={logits.shape}")
        return logits

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        仅编码，用于推理时的增量解码

        Args:
            src: 源序列 [batch_size, src_seq_len]
            src_mask: 源序列掩码 [batch_size, src_seq_len] (可选)

        Returns:
            编码器输出 [batch_size, src_seq_len, d_model]
        """
        return self.encoder(src, src_mask)

    def decode_step(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        单步解码，用于推理时的增量解码

        Args:
            tgt: 目标序列 [batch_size, tgt_seq_len]
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
            tgt_mask: 目标序列掩码 [batch_size, tgt_seq_len, tgt_seq_len] (可选)
            src_mask: 源序列掩码 [batch_size, src_seq_len] (可选)

        Returns:
            输出logits [batch_size, tgt_seq_len, vocab_size]
        """
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        logits = self.output_projection(decoder_output)
        return logits
