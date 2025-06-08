"""
T5模型模块 - 完整的T5实现
包含编码器、解码器和完整的T5模型
重点关注数据流转和shape变化，包含详细的shape注释
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import logging

from config import T5_CONFIG
from transformer import MultiHeadAttention, FeedForward, LayerNorm

logger = logging.getLogger("T5")


class T5EncoderBlock(nn.Module):
    """
    T5编码器块 - 只包含自注意力和前馈网络

    数据流：
    input: (batch_size, seq_len, d_model)
    -> self_attention: (batch_size, seq_len, d_model)
    -> add & norm: (batch_size, seq_len, d_model)
    -> feed_forward: (batch_size, seq_len, d_model)
    -> add & norm: (batch_size, seq_len, d_model)
    """

    def __init__(self, has_relative_attention_bias: bool = False):
        super().__init__()

        # 自注意力层
        self.self_attention = MultiHeadAttention(
            is_decoder=False,
            has_relative_attention_bias=has_relative_attention_bias
        )
        self.self_attn_layer_norm = LayerNorm(T5_CONFIG.d_model, eps=T5_CONFIG.layer_norm_epsilon)
        self.self_attn_dropout = nn.Dropout(T5_CONFIG.dropout_rate)

        # 前馈网络层
        self.feed_forward = FeedForward()
        self.ff_layer_norm = LayerNorm(T5_CONFIG.d_model, eps=T5_CONFIG.layer_norm_epsilon)
        self.ff_dropout = nn.Dropout(T5_CONFIG.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (batch_size, seq_len, d_model)
        attention_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len, seq_len)
        position_bias: Optional[torch.Tensor] = None,  # (1, num_heads, seq_len, seq_len)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        编码器块前向传播

        Args:
            hidden_states: 输入隐藏状态 (batch_size, seq_len, d_model)
            attention_mask: 注意力掩码 (batch_size, seq_len, seq_len)
            position_bias: 位置偏置 (1, num_heads, seq_len, seq_len)

        Returns:
            hidden_states: 输出隐藏状态 (batch_size, seq_len, d_model)
            position_bias: 更新的位置偏置 (1, num_heads, seq_len, seq_len)
        """
        # 1. 自注意力层
        normed_hidden_states = self.self_attn_layer_norm(hidden_states)
        attention_output, position_bias, _ = self.self_attention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
        )
        attention_output = self.self_attn_dropout(attention_output)
        hidden_states = hidden_states + attention_output  # 残差连接

        # 2. 前馈网络层
        normed_hidden_states = self.ff_layer_norm(hidden_states)
        ff_output = self.feed_forward(normed_hidden_states)
        ff_output = self.ff_dropout(ff_output)
        hidden_states = hidden_states + ff_output  # 残差连接

        return hidden_states, position_bias


class T5DecoderBlock(nn.Module):
    """
    T5解码器块 - 包含自注意力、交叉注意力和前馈网络

    数据流：
    input: (batch_size, seq_len, d_model)
    -> self_attention: (batch_size, seq_len, d_model)
    -> cross_attention: (batch_size, seq_len, d_model)
    -> feed_forward: (batch_size, seq_len, d_model)
    """

    def __init__(self, has_relative_attention_bias: bool = False):
        super().__init__()

        # 自注意力层
        self.self_attention = MultiHeadAttention(
            is_decoder=True,
            has_relative_attention_bias=has_relative_attention_bias
        )
        self.self_attn_layer_norm = LayerNorm(T5_CONFIG.d_model, eps=T5_CONFIG.layer_norm_epsilon)
        self.self_attn_dropout = nn.Dropout(T5_CONFIG.dropout_rate)

        # 交叉注意力层
        self.cross_attention = MultiHeadAttention(is_decoder=True)
        self.cross_attn_layer_norm = LayerNorm(T5_CONFIG.d_model, eps=T5_CONFIG.layer_norm_epsilon)
        self.cross_attn_dropout = nn.Dropout(T5_CONFIG.dropout_rate)

        # 前馈网络层
        self.feed_forward = FeedForward()
        self.ff_layer_norm = LayerNorm(T5_CONFIG.d_model, eps=T5_CONFIG.layer_norm_epsilon)
        self.ff_dropout = nn.Dropout(T5_CONFIG.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (batch_size, seq_len, d_model)
        encoder_hidden_states: torch.Tensor,  # (batch_size, encoder_seq_len, d_model)
        self_attention_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len, seq_len)
        cross_attention_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len, encoder_seq_len)
        self_position_bias: Optional[torch.Tensor] = None,  # (1, num_heads, seq_len, seq_len)
        cross_position_bias: Optional[torch.Tensor] = None,  # (1, num_heads, seq_len, encoder_seq_len)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        解码器块前向传播

        Args:
            hidden_states: 输入隐藏状态 (batch_size, seq_len, d_model)
            encoder_hidden_states: 编码器隐藏状态 (batch_size, encoder_seq_len, d_model)
            self_attention_mask: 自注意力掩码 (batch_size, seq_len, seq_len)
            cross_attention_mask: 交叉注意力掩码 (batch_size, seq_len, encoder_seq_len)
            self_position_bias: 自注意力位置偏置 (1, num_heads, seq_len, seq_len)
            cross_position_bias: 交叉注意力位置偏置 (1, num_heads, seq_len, encoder_seq_len)

        Returns:
            hidden_states: 输出隐藏状态 (batch_size, seq_len, d_model)
            self_position_bias: 更新的自注意力位置偏置
            cross_position_bias: 更新的交叉注意力位置偏置
        """
        # 1. 自注意力层
        normed_hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attention_output, self_position_bias, _ = self.self_attention(
            normed_hidden_states,
            mask=self_attention_mask,
            position_bias=self_position_bias,
        )
        self_attention_output = self.self_attn_dropout(self_attention_output)
        hidden_states = hidden_states + self_attention_output  # 残差连接

        # 2. 交叉注意力层
        normed_hidden_states = self.cross_attn_layer_norm(hidden_states)
        cross_attention_output, cross_position_bias, _ = self.cross_attention(
            normed_hidden_states,
            key_value_states=encoder_hidden_states,
            mask=cross_attention_mask,
            position_bias=cross_position_bias,
        )
        cross_attention_output = self.cross_attn_dropout(cross_attention_output)
        hidden_states = hidden_states + cross_attention_output  # 残差连接

        # 3. 前馈网络层
        normed_hidden_states = self.ff_layer_norm(hidden_states)
        ff_output = self.feed_forward(normed_hidden_states)
        ff_output = self.ff_dropout(ff_output)
        hidden_states = hidden_states + ff_output  # 残差连接

        return hidden_states, self_position_bias, cross_position_bias


class T5Encoder(nn.Module):
    """
    T5编码器 - 多个编码器块的堆叠

    数据流：
    input_ids: (batch_size, seq_len)
    -> embedding: (batch_size, seq_len, d_model)
    -> encoder_block1: (batch_size, seq_len, d_model)
    -> encoder_block2: (batch_size, seq_len, d_model)
    -> ...
    -> final_layer_norm: (batch_size, seq_len, d_model)
    """

    def __init__(self, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_tokens = embed_tokens

        # 创建多个编码器块
        self.blocks = nn.ModuleList([
            T5EncoderBlock(has_relative_attention_bias=(i == 0))
            for i in range(T5_CONFIG.num_layers)
        ])

        # 最终层归一化
        self.final_layer_norm = LayerNorm(T5_CONFIG.d_model, eps=T5_CONFIG.layer_norm_epsilon)
        self.dropout = nn.Dropout(T5_CONFIG.dropout_rate)

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch_size, seq_len)
        attention_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
    ) -> torch.Tensor:
        """
        编码器前向传播

        Args:
            input_ids: 输入token ids (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, seq_len)

        Returns:
            hidden_states: 编码器输出 (batch_size, seq_len, d_model)
        """
        # 获取输入嵌入
        hidden_states = self.embed_tokens(input_ids)  # (batch_size, seq_len, d_model)
        hidden_states = self.dropout(hidden_states)

        # 创建扩展的注意力掩码
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)

        extended_attention_mask = self._get_extended_attention_mask(attention_mask)

        # 通过所有编码器块
        position_bias = None
        for i, block in enumerate(self.blocks):
            hidden_states, position_bias = block(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias if i > 0 else None,  # 第一层计算位置偏置
            )

        # 最终层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

    def _get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        创建扩展的注意力掩码

        Args:
            attention_mask: 原始注意力掩码 (batch_size, seq_len)

        Returns:
            extended_attention_mask: 扩展的注意力掩码 (batch_size, 1, seq_len, seq_len)
        """
        # 扩展维度: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
        extended_attention_mask = attention_mask[:, None, None, :]

        # 转换为适合注意力计算的格式
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


class T5Decoder(nn.Module):
    """
    T5解码器 - 多个解码器块的堆叠

    数据流：
    decoder_input_ids: (batch_size, decoder_seq_len)
    -> embedding: (batch_size, decoder_seq_len, d_model)
    -> decoder_block1: (batch_size, decoder_seq_len, d_model)
    -> decoder_block2: (batch_size, decoder_seq_len, d_model)
    -> ...
    -> final_layer_norm: (batch_size, decoder_seq_len, d_model)
    """

    def __init__(self, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_tokens = embed_tokens

        # 创建多个解码器块
        self.blocks = nn.ModuleList([
            T5DecoderBlock(has_relative_attention_bias=(i == 0))
            for i in range(T5_CONFIG.num_layers)
        ])

        # 最终层归一化
        self.final_layer_norm = LayerNorm(T5_CONFIG.d_model, eps=T5_CONFIG.layer_norm_epsilon)
        self.dropout = nn.Dropout(T5_CONFIG.dropout_rate)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,  # (batch_size, decoder_seq_len)
        encoder_hidden_states: torch.Tensor,  # (batch_size, encoder_seq_len, d_model)
        decoder_attention_mask: Optional[torch.Tensor] = None,  # (batch_size, decoder_seq_len)
        encoder_attention_mask: Optional[torch.Tensor] = None,  # (batch_size, encoder_seq_len)
    ) -> torch.Tensor:
        """
        解码器前向传播

        Args:
            decoder_input_ids: 解码器输入token ids (batch_size, decoder_seq_len)
            encoder_hidden_states: 编码器隐藏状态 (batch_size, encoder_seq_len, d_model)
            decoder_attention_mask: 解码器注意力掩码 (batch_size, decoder_seq_len)
            encoder_attention_mask: 编码器注意力掩码 (batch_size, encoder_seq_len)

        Returns:
            hidden_states: 解码器输出 (batch_size, decoder_seq_len, d_model)
        """
        # 获取输入嵌入
        hidden_states = self.embed_tokens(decoder_input_ids)  # (batch_size, decoder_seq_len, d_model)
        hidden_states = self.dropout(hidden_states)

        # 创建注意力掩码
        if decoder_attention_mask is None:
            batch_size, decoder_seq_len = decoder_input_ids.shape
            decoder_attention_mask = torch.ones(batch_size, decoder_seq_len, device=decoder_input_ids.device)

        if encoder_attention_mask is None:
            batch_size, encoder_seq_len = encoder_hidden_states.shape[:2]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_len, device=encoder_hidden_states.device)

        # 创建扩展的注意力掩码
        self_attention_mask = self._get_decoder_attention_mask(decoder_attention_mask)
        cross_attention_mask = self._get_cross_attention_mask(decoder_attention_mask, encoder_attention_mask)

        # 通过所有解码器块
        self_position_bias = None
        cross_position_bias = None

        for i, block in enumerate(self.blocks):
            hidden_states, self_position_bias, cross_position_bias = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
                self_position_bias=self_position_bias if i > 0 else None,  # 第一层计算位置偏置
                cross_position_bias=cross_position_bias if i > 0 else None,
            )

        # 最终层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

    def _get_decoder_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        创建解码器的因果注意力掩码

        Args:
            attention_mask: 原始注意力掩码 (batch_size, seq_len)

        Returns:
            causal_mask: 因果注意力掩码 (batch_size, 1, seq_len, seq_len)
        """
        batch_size, seq_len = attention_mask.shape

        # 创建因果掩码（下三角矩阵）
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=attention_mask.device))

        # 结合padding掩码和因果掩码
        extended_attention_mask = attention_mask[:, None, None, :] * causal_mask[None, None, :, :]

        # 转换为适合注意力计算的格式
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def _get_cross_attention_mask(
        self,
        decoder_attention_mask: torch.Tensor,
        encoder_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        创建交叉注意力掩码

        Args:
            decoder_attention_mask: 解码器注意力掩码 (batch_size, decoder_seq_len)
            encoder_attention_mask: 编码器注意力掩码 (batch_size, encoder_seq_len)

        Returns:
            cross_attention_mask: 交叉注意力掩码 (batch_size, 1, decoder_seq_len, encoder_seq_len)
        """
        # 扩展维度
        extended_attention_mask = encoder_attention_mask[:, None, None, :]

        # 转换为适合注意力计算的格式
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask



class T5Model(nn.Module):
    """
    完整的T5模型 - 包含编码器和解码器

    数据流：
    encoder_input_ids: (batch_size, encoder_seq_len)
    decoder_input_ids: (batch_size, decoder_seq_len)
    -> encoder: (batch_size, encoder_seq_len, d_model)
    -> decoder: (batch_size, decoder_seq_len, d_model)
    """

    def __init__(self):
        super().__init__()

        # 共享的词嵌入
        self.shared = nn.Embedding(T5_CONFIG.vocab_size, T5_CONFIG.d_model)

        # 编码器和解码器
        self.encoder = T5Encoder(self.shared)
        self.decoder = T5Decoder(self.shared)

        # 方案1: 使用PyTorch内置初始化（可选）
        if T5_CONFIG.use_custom_init:
            self.apply(self._init_weights)
        else:
            # 使用PyTorch默认初始化，通常也能工作
            pass

        logger.info(f"T5模型初始化完成，参数数量: {self.count_parameters():,}")

    def _init_weights(self, module):
        """自定义权重初始化（仅在use_custom_init=True时使用）"""
        factor = T5_CONFIG.initializer_factor
        if isinstance(module, LayerNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()

    def count_parameters(self) -> int:
        """计算参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        """设置输入嵌入层"""
        self.shared = new_embeddings
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch_size, encoder_seq_len)
        decoder_input_ids: torch.Tensor,  # (batch_size, decoder_seq_len)
        attention_mask: Optional[torch.Tensor] = None,  # (batch_size, encoder_seq_len)
        decoder_attention_mask: Optional[torch.Tensor] = None,  # (batch_size, decoder_seq_len)
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 编码器输入token ids (batch_size, encoder_seq_len)
            decoder_input_ids: 解码器输入token ids (batch_size, decoder_seq_len)
            attention_mask: 编码器注意力掩码 (batch_size, encoder_seq_len)
            decoder_attention_mask: 解码器注意力掩码 (batch_size, decoder_seq_len)

        Returns:
            包含模型输出的字典
        """
        # 编码器前向传播
        encoder_hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # (batch_size, encoder_seq_len, d_model)

        # 解码器前向传播
        decoder_hidden_states = self.decoder(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
        )  # (batch_size, decoder_seq_len, d_model)

        return {
            "last_hidden_state": decoder_hidden_states,
            "encoder_last_hidden_state": encoder_hidden_states,
        }


class T5ForConditionalGeneration(nn.Module):
    """
    用于条件生成的T5模型 - 包含语言模型头

    数据流：
    input -> T5Model -> lm_head -> logits: (batch_size, decoder_seq_len, vocab_size)
    """

    def __init__(self):
        super().__init__()

        self.model_dim = T5_CONFIG.d_model
        self.shared = nn.Embedding(T5_CONFIG.vocab_size, T5_CONFIG.d_model)

        # T5主模型
        self.encoder = T5Encoder(self.shared)
        self.decoder = T5Decoder(self.shared)

        # 语言模型头
        self.lm_head = nn.Linear(T5_CONFIG.d_model, T5_CONFIG.vocab_size, bias=False)

        # 方案1: 使用PyTorch内置初始化（可选）
        if T5_CONFIG.use_custom_init:
            self.apply(self._init_weights)
        else:
            # 使用PyTorch默认初始化，通常也能工作
            pass

        logger.info(f"T5ForConditionalGeneration初始化完成，参数数量: {self.count_parameters():,}")

    def _init_weights(self, module):
        """自定义权重初始化（仅在use_custom_init=True时使用）"""
        factor = T5_CONFIG.initializer_factor
        if isinstance(module, LayerNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()

    def count_parameters(self) -> int:
        """计算参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        """设置输入嵌入层"""
        self.shared = new_embeddings
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        """获取输出嵌入层"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """设置输出嵌入层"""
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch_size, encoder_seq_len)
        decoder_input_ids: torch.Tensor,  # (batch_size, decoder_seq_len)
        attention_mask: Optional[torch.Tensor] = None,  # (batch_size, encoder_seq_len)
        decoder_attention_mask: Optional[torch.Tensor] = None,  # (batch_size, decoder_seq_len)
        labels: Optional[torch.Tensor] = None,  # (batch_size, decoder_seq_len)
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 编码器输入 (batch_size, encoder_seq_len)
            decoder_input_ids: 解码器输入 (batch_size, decoder_seq_len)
            attention_mask: 编码器注意力掩码 (batch_size, encoder_seq_len)
            decoder_attention_mask: 解码器注意力掩码 (batch_size, decoder_seq_len)
            labels: 标签 (batch_size, decoder_seq_len)

        Returns:
            包含损失和logits的字典
        """
        # 编码器前向传播
        encoder_hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # (batch_size, encoder_seq_len, d_model)

        # 解码器前向传播
        decoder_hidden_states = self.decoder(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
        )  # (batch_size, decoder_seq_len, d_model)

        # 语言模型头
        lm_logits = self.lm_head(decoder_hidden_states)  # (batch_size, decoder_seq_len, vocab_size)

        loss = None
        if labels is not None:
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return {
            "loss": loss,
            "logits": lm_logits,
            "encoder_last_hidden_state": encoder_hidden_states,
        }
