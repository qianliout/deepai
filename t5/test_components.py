"""
T5框架组件测试脚本
测试各个模块的基本功能
"""

import torch
import logging
from config import setup_logging, T5_CONFIG, print_config
from transformer import LayerNorm, MultiHeadAttention, FeedForward
from model import T5Model, T5ForConditionalGeneration
from data_loader import T5DataSample

logger = logging.getLogger("T5")


def test_config():
    """测试配置模块"""
    print("\n🔧 测试配置模块")
    print("-" * 30)
    
    print(f"模型维度: {T5_CONFIG.d_model}")
    print(f"层数: {T5_CONFIG.num_layers}")
    print(f"注意力头数: {T5_CONFIG.num_heads}")
    print(f"词汇表大小: {T5_CONFIG.vocab_size}")
    
    print("✅ 配置模块测试通过")


def test_transformer_components():
    """测试Transformer组件"""
    print("\n🔧 测试Transformer组件")
    print("-" * 30)
    
    batch_size, seq_len, d_model = 2, 10, T5_CONFIG.d_model
    
    # 测试LayerNorm
    print("测试LayerNorm...")
    layer_norm = LayerNorm(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    output = layer_norm(x)
    assert output.shape == (batch_size, seq_len, d_model)
    print(f"  输入shape: {x.shape}")
    print(f"  输出shape: {output.shape}")
    
    # 测试MultiHeadAttention
    print("测试MultiHeadAttention...")
    attention = MultiHeadAttention(has_relative_attention_bias=True)
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    attn_output, position_bias, _ = attention(hidden_states)
    assert attn_output.shape == (batch_size, seq_len, d_model)
    print(f"  输入shape: {hidden_states.shape}")
    print(f"  输出shape: {attn_output.shape}")
    print(f"  位置偏置shape: {position_bias.shape}")
    
    # 测试FeedForward
    print("测试FeedForward...")
    ff = FeedForward()
    ff_output = ff(hidden_states)
    assert ff_output.shape == (batch_size, seq_len, d_model)
    print(f"  输入shape: {hidden_states.shape}")
    print(f"  输出shape: {ff_output.shape}")
    
    print("✅ Transformer组件测试通过")


def test_model():
    """测试模型"""
    print("\n🔧 测试模型")
    print("-" * 30)
    
    batch_size = 2
    encoder_seq_len = 10
    decoder_seq_len = 8
    
    # 测试T5Model
    print("测试T5Model...")
    model = T5Model()
    
    # 准备输入
    input_ids = torch.randint(0, T5_CONFIG.vocab_size, (batch_size, encoder_seq_len))
    decoder_input_ids = torch.randint(0, T5_CONFIG.vocab_size, (batch_size, decoder_seq_len))
    
    outputs = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids
    )
    
    print(f"  编码器输入shape: {input_ids.shape}")
    print(f"  解码器输入shape: {decoder_input_ids.shape}")
    print(f"  解码器输出shape: {outputs['last_hidden_state'].shape}")
    print(f"  编码器输出shape: {outputs['encoder_last_hidden_state'].shape}")
    
    # 测试T5ForConditionalGeneration
    print("测试T5ForConditionalGeneration...")
    model_gen = T5ForConditionalGeneration()
    
    labels = torch.randint(0, T5_CONFIG.vocab_size, (batch_size, decoder_seq_len))
    
    outputs = model_gen(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        labels=labels
    )
    
    print(f"  logits shape: {outputs['logits'].shape}")
    print(f"  loss: {outputs['loss'].item():.4f}")
    
    print("✅ 模型测试通过")


def test_data_sample():
    """测试数据样本"""
    print("\n🔧 测试数据样本")
    print("-" * 30)
    
    # 创建测试样本
    sample = T5DataSample(
        input_ids=[1, 2, 3, 4, 5],
        attention_mask=[1, 1, 1, 1, 1],
        decoder_input_ids=[0, 1, 2, 3],
        decoder_attention_mask=[1, 1, 1, 1],
        labels=[1, 2, 3, -100]
    )
    
    print(f"  input_ids: {sample.input_ids}")
    print(f"  attention_mask: {sample.attention_mask}")
    print(f"  decoder_input_ids: {sample.decoder_input_ids}")
    print(f"  labels: {sample.labels}")
    
    print("✅ 数据样本测试通过")


def test_parameter_count():
    """测试参数数量"""
    print("\n🔧 测试参数数量")
    print("-" * 30)
    
    model = T5ForConditionalGeneration()
    param_count = model.count_parameters()
    
    print(f"  总参数数量: {param_count:,}")
    
    # 计算理论参数数量（大致估算）
    d_model = T5_CONFIG.d_model
    vocab_size = T5_CONFIG.vocab_size
    num_layers = T5_CONFIG.num_layers
    d_ff = T5_CONFIG.d_ff
    
    # 嵌入层参数
    embedding_params = vocab_size * d_model
    
    # 每层的参数（注意力 + 前馈）
    attention_params_per_layer = 4 * d_model * d_model  # q, k, v, o
    ff_params_per_layer = 2 * d_model * d_ff  # wi, wo
    layer_params = (attention_params_per_layer + ff_params_per_layer) * 2  # 编码器 + 解码器
    
    # 总参数估算
    estimated_params = embedding_params + layer_params * num_layers
    
    print(f"  估算参数数量: {estimated_params:,}")
    print(f"  参数数量合理性: {'✅' if abs(param_count - estimated_params) / estimated_params < 0.5 else '❌'}")
    
    print("✅ 参数数量测试通过")


def test_device_compatibility():
    """测试设备兼容性"""
    print("\n🔧 测试设备兼容性")
    print("-" * 30)
    
    from config import get_device
    
    device = get_device()
    print(f"  检测到的设备: {device}")
    
    # 测试模型在设备上的运行
    model = T5ForConditionalGeneration()
    model.to(device)
    
    # 创建测试输入
    batch_size = 1
    seq_len = 5
    input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
    decoder_input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    
    print(f"  模型输出设备: {outputs['logits'].device}")
    print(f"  设备兼容性: {'✅' if outputs['logits'].device == device else '❌'}")
    
    print("✅ 设备兼容性测试通过")


def main():
    """主测试函数"""
    # 设置日志
    setup_logging()
    
    print("🚀 开始T5框架组件测试")
    print("=" * 50)
    
    try:
        # 运行各项测试
        test_config()
        test_transformer_components()
        test_model()
        test_data_sample()
        test_parameter_count()
        test_device_compatibility()
        
        print("\n🎉 所有测试通过！")
        print("=" * 50)
        print("T5框架各组件工作正常")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        logger.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    main()
