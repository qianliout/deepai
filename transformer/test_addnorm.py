"""
测试 AddNorm 模块的功能
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


# 直接导入需要的类，避免依赖config
class LayerNorm(nn.Module):
    """层归一化"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta


class AddNorm(nn.Module):
    """残差连接 + 层归一化 (Add & Norm)"""

    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.layer_norm = LayerNorm(d_model, eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x + self.dropout(sublayer_output))


class MultiHeadAttention(nn.Module):
    """简化的多头注意力机制"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        seq_len_q = query.size(1)

        Q = (
            self.w_q(query)
            .view(batch_size, seq_len_q, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == True, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, self.d_model)
        )
        return self.w_o(output)


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


def test_addnorm_basic():
    """测试基本的 AddNorm 功能"""
    print("=== 测试基本 AddNorm 功能 ===")

    batch_size, seq_len, d_model = 2, 10, 512
    dropout = 0.1

    # 创建 AddNorm 模块
    add_norm = AddNorm(d_model, dropout)

    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    sublayer_output = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    output = add_norm(x, sublayer_output)

    print(f"输入形状: {x.shape}")
    print(f"子层输出形状: {sublayer_output.shape}")
    print(f"AddNorm 输出形状: {output.shape}")
    print(f"输出均值: {output.mean().item():.6f}")
    print(f"输出标准差: {output.std().item():.6f}")
    print()


def test_addnorm_with_attention():
    """测试 AddNorm 与注意力机制的结合"""
    print("=== 测试 AddNorm 与注意力机制 ===")

    batch_size, seq_len, d_model = 2, 10, 512
    n_heads = 8
    dropout = 0.1

    # 创建模块
    attention = MultiHeadAttention(d_model, n_heads, dropout)
    add_norm = AddNorm(d_model, dropout)

    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)

    # 注意力计算
    attn_output = attention(x, x, x)

    # AddNorm
    output = add_norm(x, attn_output)

    print(f"输入形状: {x.shape}")
    print(f"注意力输出形状: {attn_output.shape}")
    print(f"AddNorm 输出形状: {output.shape}")
    print()


def test_addnorm_with_feedforward():
    """测试 AddNorm 与前馈网络的结合"""
    print("=== 测试 AddNorm 与前馈网络 ===")

    batch_size, seq_len, d_model = 2, 10, 512
    d_ff = 2048
    dropout = 0.1

    # 创建模块
    feed_forward = FeedForward(d_model, d_ff, dropout)
    add_norm = AddNorm(d_model, dropout)

    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)

    # 前馈网络计算
    ff_output = feed_forward(x)

    # AddNorm
    output = add_norm(x, ff_output)

    print(f"输入形状: {x.shape}")
    print(f"前馈网络输出形状: {ff_output.shape}")
    print(f"AddNorm 输出形状: {output.shape}")
    print()


def test_residual_connection():
    """测试残差连接的效果"""
    print("=== 测试残差连接效果 ===")

    batch_size, seq_len, d_model = 2, 10, 512
    dropout = 0.0  # 关闭 dropout 以便观察残差连接效果

    # 创建 AddNorm 模块（无 dropout）
    add_norm = AddNorm(d_model, dropout)

    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)

    # 创建一个零输出的子层（模拟恒等映射）
    zero_output = torch.zeros_like(x)

    # AddNorm 应该近似等于 LayerNorm(x)
    output = add_norm(x, zero_output)

    # 直接使用 LayerNorm
    layer_norm = LayerNorm(d_model)
    expected_output = layer_norm(x)

    # 比较结果
    diff = torch.abs(output - expected_output).mean()
    print(f"AddNorm 与 LayerNorm 的差异: {diff.item():.8f}")
    print(f"差异应该很小，表明残差连接正常工作")
    print()


if __name__ == "__main__":
    print("开始测试 AddNorm 模块...")
    print()

    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)

    # 运行所有测试
    test_addnorm_basic()
    test_addnorm_with_attention()
    test_addnorm_with_feedforward()
    test_residual_connection()

    print("所有测试完成！")
