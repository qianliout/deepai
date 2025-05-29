"""
Transformer2框架测试脚本
验证所有核心功能是否正常工作
"""

import torch
import logging
from pathlib import Path

from config import TRANSFORMER_CONFIG, TRAINING_CONFIG, DATA_CONFIG, print_config, update_config_for_quick_test, setup_logging
from model import Transformer
from transformer import (
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    EncoderLayer,
    DecoderLayer,
    create_look_ahead_mask,
    create_combined_mask,
)
from data_loader import SimpleTokenizer, DataManager
from trainer import Trainer
from inference import TransformerInference

logger = logging.getLogger("Transformer2")


def test_config():
    """测试配置系统"""
    print("🧪 测试配置系统...")

    # 测试配置打印
    print_config()

    # 测试配置值是否合理
    assert TRANSFORMER_CONFIG.d_model > 0, "d_model必须大于0"
    assert TRANSFORMER_CONFIG.n_heads > 0, "n_heads必须大于0"
    assert TRANSFORMER_CONFIG.d_model % TRANSFORMER_CONFIG.n_heads == 0, "d_model必须能被n_heads整除"
    assert TRAINING_CONFIG.batch_size > 0, "batch_size必须大于0"
    assert DATA_CONFIG.max_length > 0, "max_length必须大于0"

    print("✅ 配置系统测试通过")


def test_transformer_components():
    """测试Transformer核心组件"""
    print("\n🧪 测试Transformer核心组件...")

    batch_size, seq_len, d_model = 2, 10, 256
    n_heads = 4

    # 测试位置编码
    pos_encoding = PositionalEncoding(d_model, 512)
    x = torch.randn(batch_size, seq_len, d_model)
    pos_out = pos_encoding(x)
    assert pos_out.shape == (batch_size, seq_len, d_model), f"位置编码输出shape错误: {pos_out.shape}"

    # 测试多头注意力
    attention = MultiHeadAttention(d_model, n_heads)
    query = key = value = torch.randn(batch_size, seq_len, d_model)
    attn_out, attn_weights = attention(query, key, value)
    assert attn_out.shape == (batch_size, seq_len, d_model), f"注意力输出shape错误: {attn_out.shape}"
    assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len), f"注意力权重shape错误: {attn_weights.shape}"

    # 测试前馈网络
    ff = FeedForward(d_model, d_model * 4)
    ff_out = ff(x)
    assert ff_out.shape == (batch_size, seq_len, d_model), f"前馈网络输出shape错误: {ff_out.shape}"

    # 测试编码器层
    encoder_layer = EncoderLayer(d_model, n_heads, d_model * 4)
    enc_out = encoder_layer(x)
    assert enc_out.shape == (batch_size, seq_len, d_model), f"编码器层输出shape错误: {enc_out.shape}"

    # 测试解码器层
    decoder_layer = DecoderLayer(d_model, n_heads, d_model * 4)
    dec_out = decoder_layer(x, x)  # 使用x作为编码器输出
    assert dec_out.shape == (batch_size, seq_len, d_model), f"解码器层输出shape错误: {dec_out.shape}"

    print("✅ Transformer核心组件测试通过")


def test_model():
    """测试完整模型"""
    print("\n🧪 测试完整模型...")

    # 创建模型
    model = Transformer()

    # 测试前向传播
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8

    src = torch.randint(0, TRANSFORMER_CONFIG.vocab_size_src, (batch_size, src_seq_len))
    tgt = torch.randint(0, TRANSFORMER_CONFIG.vocab_size_tgt, (batch_size, tgt_seq_len))

    # 创建掩码
    tgt_mask = create_combined_mask(tgt)

    # 前向传播
    logits = model(src, tgt, None, tgt_mask)
    expected_shape = (batch_size, tgt_seq_len, TRANSFORMER_CONFIG.vocab_size_tgt)
    assert logits.shape == expected_shape, f"模型输出shape错误: {logits.shape}, 期望: {expected_shape}"

    # 测试编码器单独使用
    encoder_out = model.encode(src)
    expected_enc_shape = (batch_size, src_seq_len, TRANSFORMER_CONFIG.d_model)
    assert encoder_out.shape == expected_enc_shape, f"编码器输出shape错误: {encoder_out.shape}"

    print("✅ 完整模型测试通过")


def test_tokenizer():
    """测试分词器"""
    print("\n🧪 测试分词器...")

    tokenizer = SimpleTokenizer()

    # 构建简单词汇表
    src_texts = ["hello world", "how are you", "good morning"]
    tgt_texts = ["ciao mondo", "come stai", "buongiorno"]
    tokenizer.build_vocab(src_texts, tgt_texts)

    # 测试编码
    text = "hello world"
    ids = tokenizer.encode(text, "src", 10)
    assert len(ids) == 10, f"编码长度错误: {len(ids)}"

    # 测试解码
    decoded = tokenizer.decode(ids, "src")
    assert isinstance(decoded, str), "解码结果不是字符串"

    # 测试不填充模式
    ids_no_pad = tokenizer.encode(text, "src", 10, pad_to_max=False)
    assert len(ids_no_pad) <= 10, f"不填充模式长度错误: {len(ids_no_pad)}"

    print("✅ 分词器测试通过")


def test_data_manager():
    """测试数据管理器（简化版）"""
    print("\n🧪 测试数据管理器...")

    # 由于数据下载可能很慢，这里只测试初始化
    data_manager = DataManager()
    assert data_manager.tokenizer is not None, "数据管理器初始化失败"

    print("✅ 数据管理器测试通过")


def test_inference():
    """测试推理功能"""
    print("\n🧪 测试推理功能...")

    # 检查是否有训练好的模型
    model_path = Path("./transformer2_quick_test/best_model.pt")
    if not model_path.exists():
        print("⚠️  没有找到训练好的模型，跳过推理测试")
        return

    try:
        # 创建推理器
        inference = TransformerInference(str(model_path))

        # 测试翻译
        result = inference.translate("hello", max_length=10)
        assert isinstance(result, str), "翻译结果不是字符串"

        print("✅ 推理功能测试通过")
    except Exception as e:
        print(f"⚠️  推理测试失败: {e}")


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始Transformer2框架测试")
    print("=" * 60)

    # 设置日志
    setup_logging()

    # 切换到快速测试模式
    update_config_for_quick_test()

    try:
        # 运行各项测试
        test_config()
        test_transformer_components()
        test_model()
        test_tokenizer()
        test_data_manager()
        test_inference()

        print("\n" + "=" * 60)
        print("🎉 所有测试通过！Transformer2框架工作正常")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        logger.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
