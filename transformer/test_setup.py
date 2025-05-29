#!/usr/bin/env python3
"""
测试脚本 - 验证代码设置是否正确
"""
import sys
import traceback


def test_imports():
    """测试所有模块导入"""
    print("测试模块导入...")

    try:
        import torch

        print("✓ PyTorch")

        # 测试设备
        if torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon GPU) 可用")
        elif torch.cuda.is_available():
            print("✓ CUDA 可用")
        else:
            print("✓ CPU 模式")

    except ImportError as e:
        print(f"✗ PyTorch 导入失败: {e}")
        return False

    try:
        from pydantic import BaseModel

        print("✓ Pydantic")
    except ImportError as e:
        print(f"✗ Pydantic 导入失败: {e}")
        return False

    try:
        import datasets

        print("✓ Datasets")
    except ImportError as e:
        print(f"✗ Datasets 导入失败: {e}")
        return False

    try:
        import numpy

        print("✓ NumPy")
    except ImportError as e:
        print(f"✗ NumPy 导入失败: {e}")
        return False

    return True


def test_local_imports():
    """测试本地模块导入"""
    print("\n测试本地模块导入...")

    modules = ["config", "utils", "tokenizer", "model", "data_loader", "trainer"]

    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name} 导入失败: {e}")
            traceback.print_exc()
            return False

    return True


def test_config():
    """测试配置"""
    print("\n测试配置...")

    try:
        from config import Config, default_config

        # 测试默认配置
        config = default_config
        print(f"✓ 默认配置加载成功")
        print(f"  - 模型维度: {config.model.d_model}")
        print(f"  - 注意力头数: {config.model.n_heads}")
        print(f"  - 设备: {config.training.device}")

        return True

    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")

    try:
        from config import default_config
        from model import Transformer

        # 创建小型模型用于测试
        config = default_config
        config.model.d_model = 128
        config.model.n_heads = 4
        config.model.n_layers = 2
        config.model.vocab_size_en = 1000
        config.model.vocab_size_it = 1000

        model = Transformer(config.model)
        print(f"✓ 模型创建成功")

        # 计算参数数量
        from utils import count_parameters

        param_count = count_parameters(model)
        print(f"  - 参数数量: {param_count:,}")

        return True

    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        traceback.print_exc()
        return False


def test_tokenizer():
    """测试分词器"""
    print("\n测试分词器...")

    try:
        from config import default_config
        from tokenizer import SimpleTokenizer

        tokenizer = SimpleTokenizer(default_config.model)

        # 测试分词
        text = "Hello, how are you?"
        tokens = tokenizer.tokenize(text)
        print(f"✓ 分词测试: '{text}' -> {tokens}")

        # 测试词典构建
        texts = ["Hello world", "How are you", "I am fine", "Thank you very much"]

        vocab = tokenizer.build_vocab(texts, "en", min_freq=1, max_vocab_size=100)
        print(f"✓ 词典构建成功，大小: {len(vocab)}")

        return True

    except Exception as e:
        print(f"✗ 分词器测试失败: {e}")
        traceback.print_exc()
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\n测试前向传播...")

    try:
        import torch
        from config import default_config
        from model import Transformer

        # 创建小型模型
        config = default_config
        config.model.d_model = 64
        config.model.n_heads = 4
        config.model.n_layers = 2
        config.model.vocab_size_en = 100
        config.model.vocab_size_it = 100
        config.model.max_seq_len = 10

        model = Transformer(config.model)
        model.eval()

        # 创建测试数据
        batch_size = 2
        src_len = 8
        tgt_len = 6

        src = torch.randint(1, 100, (batch_size, src_len))
        tgt = torch.randint(1, 100, (batch_size, tgt_len))

        # 前向传播
        with torch.no_grad():
            output = model(src, tgt)

        print(f"✓ 前向传播成功")
        print(f"  - 输入形状: src={src.shape}, tgt={tgt.shape}")
        print(f"  - 输出形状: {output.shape}")

        return True

    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("Transformer实现 - 设置验证")
    print("=" * 60)

    tests = [
        test_imports,
        test_local_imports,
        test_config,
        test_model_creation,
        test_tokenizer,
        test_forward_pass,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("测试失败，停止后续测试")
                break
        except Exception as e:
            print(f"测试异常: {e}")
            traceback.print_exc()
            break

    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！代码设置正确。")
        print("\n可以开始训练:")
        print("  python main.py --mode train")
        print("\n或使用一键脚本:")
        print("  python run.py")
    else:
        print("❌ 部分测试失败，请检查依赖和代码。")
        print("\n安装依赖:")
        print("  pip install -r requirements.txt")

    print("=" * 60)


if __name__ == "__main__":
    main()
