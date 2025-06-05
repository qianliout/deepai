"""
T5 Transformer组件模块
包含注意力机制、前馈网络、层归一化等核心组件
重点关注数据流转和shape变化，包含详细的shape注释
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

from config import T5_CONFIG

logger = logging.getLogger("T5")


class LayerNorm(nn.Module):
    """
    T5风格的LayerNorm - 不包含bias项
    
    数据流：
    input: (batch_size, seq_len, d_model) -> output: (batch_size, seq_len, d_model)
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 输入张量 (batch_size, seq_len, d_model)
            
        Returns:
            normalized: 归一化后的张量 (batch_size, seq_len, d_model)
        """
        # 计算方差（沿最后一个维度）
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # (batch_size, seq_len, 1)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # (batch_size, seq_len, d_model)
        
        # 应用可学习的缩放参数
        return self.weight * hidden_states  # (batch_size, seq_len, d_model)


class RelativePositionBias(nn.Module):
    """
    T5相对位置偏置
    
    数据流：
    query_length, key_length -> relative_position_bucket -> bias: (1, num_heads, query_length, key_length)
    """
    
    def __init__(self, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, T5_CONFIG.num_heads)
    
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        将相对位置映射到桶索引
        
        Args:
            relative_position: 相对位置 (query_length, key_length)
            
        Returns:
            relative_buckets: 桶索引 (query_length, key_length)
        """
        relative_buckets = 0
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
            num_buckets = self.num_buckets
        
        # 现在relative_position在[0, inf)范围内
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # 对于大的相对位置，使用对数缩放
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    
    def forward(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        """
        计算相对位置偏置
        
        Args:
            query_length: 查询序列长度
            key_length: 键序列长度
            device: 设备
            
        Returns:
            values: 相对位置偏置 (1, num_heads, query_length, key_length)
        """
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]  # (query_length, 1)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]  # (1, key_length)
        relative_position = memory_position - context_position  # (query_length, key_length)
        
        relative_position_bucket = self._relative_position_bucket(relative_position)  # (query_length, key_length)
        values = self.relative_attention_bias(relative_position_bucket)  # (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # (1, num_heads, query_length, key_length)
        return values


class MultiHeadAttention(nn.Module):
    """
    T5多头注意力机制
    
    数据流：
    query: (batch_size, query_len, d_model)
    key: (batch_size, key_len, d_model)  
    value: (batch_size, key_len, d_model)
    -> attention_output: (batch_size, query_len, d_model)
    """
    
    def __init__(self, is_decoder: bool = False, has_relative_attention_bias: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.d_model = T5_CONFIG.d_model
        self.d_kv = T5_CONFIG.d_kv
        self.num_heads = T5_CONFIG.num_heads
        
        # 线性投影层
        self.q = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.k = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.v = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.o = nn.Linear(self.num_heads * self.d_kv, self.d_model, bias=False)
        
        # 相对位置偏置
        if self.has_relative_attention_bias:
            self.relative_attention_bias = RelativePositionBias(
                bidirectional=not is_decoder,
                num_buckets=T5_CONFIG.relative_attention_num_buckets,
                max_distance=T5_CONFIG.relative_attention_max_distance
            )
        
        self.dropout = nn.Dropout(T5_CONFIG.dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # (batch_size, seq_len, d_model)
        key_value_states: Optional[torch.Tensor] = None,  # (batch_size, kv_seq_len, d_model)
        mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len, kv_seq_len)
        position_bias: Optional[torch.Tensor] = None,  # (1, num_heads, seq_len, kv_seq_len)
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播
        
        Args:
            hidden_states: 查询输入 (batch_size, seq_len, d_model)
            key_value_states: 键值输入，如果为None则使用hidden_states (batch_size, kv_seq_len, d_model)
            mask: 注意力掩码 (batch_size, seq_len, kv_seq_len)
            position_bias: 位置偏置 (1, num_heads, seq_len, kv_seq_len)
            past_key_value: 缓存的键值对
            use_cache: 是否使用缓存
            
        Returns:
            attention_output: 注意力输出 (batch_size, seq_len, d_model)
            position_bias: 位置偏置 (1, num_heads, seq_len, kv_seq_len)
            present_key_value: 当前键值对（如果use_cache=True）
        """
        batch_size, seq_len = hidden_states.shape[:2]
        
        # 如果没有提供key_value_states，使用hidden_states（自注意力）
        if key_value_states is None:
            key_value_states = hidden_states
        
        kv_seq_len = key_value_states.shape[1]
        
        # 线性投影
        query_states = self.q(hidden_states)  # (batch_size, seq_len, num_heads * d_kv)
        key_states = self.k(key_value_states)  # (batch_size, kv_seq_len, num_heads * d_kv)
        value_states = self.v(key_value_states)  # (batch_size, kv_seq_len, num_heads * d_kv)
        
        # 重塑为多头格式
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.d_kv).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_kv)
        key_states = key_states.view(batch_size, kv_seq_len, self.num_heads, self.d_kv).transpose(1, 2)  # (batch_size, num_heads, kv_seq_len, d_kv)
        value_states = value_states.view(batch_size, kv_seq_len, self.num_heads, self.d_kv).transpose(1, 2)  # (batch_size, num_heads, kv_seq_len, d_kv)
        
        # 处理缓存
        present_key_value = None
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=-2)
            value_states = torch.cat([past_key_value[1], value_states], dim=-2)
        
        if use_cache:
            present_key_value = (key_states, value_states)
        
        # 计算注意力分数
        scores = torch.matmul(query_states, key_states.transpose(-1, -2))  # (batch_size, num_heads, seq_len, kv_seq_len)
        
        # 计算相对位置偏置
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.relative_attention_bias(seq_len, kv_seq_len, hidden_states.device)
            else:
                position_bias = torch.zeros((1, self.num_heads, seq_len, kv_seq_len), device=hidden_states.device, dtype=hidden_states.dtype)
        
        scores += position_bias  # (batch_size, num_heads, seq_len, kv_seq_len)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, kv_seq_len)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, value_states)  # (batch_size, num_heads, seq_len, d_kv)
        
        # 重塑回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_kv)  # (batch_size, seq_len, num_heads * d_kv)
        
        # 输出投影
        attn_output = self.o(attn_output)  # (batch_size, seq_len, d_model)
        
        return attn_output, position_bias, present_key_value


class FeedForward(nn.Module):
    """
    T5前馈网络
    
    数据流：
    input: (batch_size, seq_len, d_model)
    -> wi: (batch_size, seq_len, d_ff)
    -> activation: (batch_size, seq_len, d_ff)
    -> wo: (batch_size, seq_len, d_model)
    """
    
    def __init__(self):
        super().__init__()
        self.wi = nn.Linear(T5_CONFIG.d_model, T5_CONFIG.d_ff, bias=False)
        self.wo = nn.Linear(T5_CONFIG.d_ff, T5_CONFIG.d_model, bias=False)
        self.dropout = nn.Dropout(T5_CONFIG.dropout_rate)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 输入 (batch_size, seq_len, d_model)
            
        Returns:
            output: 输出 (batch_size, seq_len, d_model)
        """
        hidden_states = self.wi(hidden_states)  # (batch_size, seq_len, d_ff)
        hidden_states = F.relu(hidden_states)  # (batch_size, seq_len, d_ff)
        hidden_states = self.dropout(hidden_states)  # (batch_size, seq_len, d_ff)
        hidden_states = self.wo(hidden_states)  # (batch_size, seq_len, d_model)
        return hidden_states
