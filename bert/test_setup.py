"""
测试脚本 - 验证BERT2框架的安装和基本功能
"""

import torch
import sys
from pathlib import Path


def test_imports():
    """测试所有模块导入"""
    print("🔍 测试模块导入...")

    try:
        from config import (
            BERT_CONFIG,
            TRAINING_CONFIG,
            DATA_CONFIG,
            get_device,
            print_config,
        )

        print("✅ config模块导入成功")

        from transformer import MultiHeadSelfAttention, TransformerEncoderLayer

        print("✅ transformer模块导入成功")

        from model import BertModel, BertForPreTraining, BertForSequenceClassification

        print("✅ model模块导入成功")

        from data_loader import BertDataCollator, create_pretraining_dataloader

        print("✅ data_loader模块导入成功")

        from trainer import BertTrainer

        print("✅ trainer模块导入成功")

        from fine_tuning import BertFineTuner

        print("✅ fine_tuning模块导入成功")

        from inference import BertInference

        print("✅ inference模块导入成功")

        return True

    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False


def test_device():
    """测试设备检测"""
    print("\n🖥️ 测试设备检测...")

    try:
        from config import get_device

        device = get_device()
        print(f"✅ 检测到设备: {device}")

        # 测试tensor创建
        x = torch.randn(2, 3).to(device)
        print(f"✅ 在{device}上创建tensor成功: {x.shape}")

        return True

    except Exception as e:
        print(f"❌ 设备测试失败: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n🏗️ 测试模型创建...")

    try:
        from model import BertForPreTraining, BertForSequenceClassification
        from config import get_device

        device = get_device()

        # 测试预训练模型
        print("创建预训练模型...")
        pretrain_model = BertForPreTraining()
        pretrain_model.to(device)
        print(f"✅ 预训练模型创建成功，参数数量: {pretrain_model.count_parameters():,}")

        # 测试分类模型
        print("创建分类模型...")
        classification_model = BertForSequenceClassification(num_labels=2)
        classification_model.to(device)
        print(f"✅ 分类模型创建成功，参数数量: {classification_model.count_parameters():,}")

        return True

    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n📊 测试数据加载...")

    try:
        from data_loader import BertDataCollator, BertPretrainingDataset
        from transformers import AutoTokenizer
        from config import DATA_CONFIG

        # 加载tokenizer
        print("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(DATA_CONFIG.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        print("✅ tokenizer加载成功")

        # 创建测试数据
        print("创建测试数据集...")
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
        ]

        dataset = BertPretrainingDataset(test_texts, tokenizer)
        print(f"✅ 数据集创建成功，样本数量: {len(dataset)}")

        # 测试数据整理器
        print("测试数据整理器...")
        collator = BertDataCollator(tokenizer)

        # 获取一个批次
        batch_samples = [dataset[i] for i in range(min(2, len(dataset)))]
        batch = collator(batch_samples)

        print(f"✅ 批次数据创建成功:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

        return True

    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\n⚡ 测试前向传播...")

    try:
        from model import BertForPreTraining
        from data_loader import BertDataCollator, BertPretrainingDataset
        from transformers import AutoTokenizer
        from config import DATA_CONFIG, get_device

        device = get_device()

        # 创建模型
        model = BertForPreTraining()
        model.to(device)
        model.eval()

        # 创建测试数据
        tokenizer = AutoTokenizer.from_pretrained(DATA_CONFIG.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "[MASK]"})

        test_texts = ["Hello world. This is a test."]
        dataset = BertPretrainingDataset(test_texts, tokenizer)
        collator = BertDataCollator(tokenizer)

        # 创建批次
        batch_samples = [dataset[0]]
        batch = collator(batch_samples)
        batch = {k: v.to(device) for k, v in batch.items()}

        # 前向传播
        print("执行前向传播...")
        with torch.no_grad():
            outputs = model(**batch)

        print(f"✅ 前向传播成功:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif value is not None:
                print(f"  {key}: {type(value)}")

        return True

    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        return False


def test_config():
    """测试配置"""
    print("\n⚙️ 测试配置...")

    try:
        from config import print_config, BERT_CONFIG, TRAINING_CONFIG, DATA_CONFIG

        print("配置信息:")
        print_config()

        # 验证配置的合理性
        assert BERT_CONFIG.hidden_size % BERT_CONFIG.num_attention_heads == 0, "hidden_size必须能被num_attention_heads整除"
        assert TRAINING_CONFIG.batch_size > 0, "batch_size必须大于0"
        assert DATA_CONFIG.max_length > 0, "max_length必须大于0"

        print("✅ 配置验证通过")

        return True

    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 BERT2框架测试")
    print("=" * 50)

    tests = [
        ("模块导入", test_imports),
        ("设备检测", test_device),
        ("配置验证", test_config),
        ("模型创建", test_model_creation),
        ("数据加载", test_data_loading),
        ("前向传播", test_forward_pass),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")

    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！框架安装正确。")
        print("\n📚 接下来你可以:")
        print("  1. 运行快速测试: python main.py quick")
        print("  2. 查看配置: python -c 'from config import print_config; print_config()'")
        print("  3. 开始预训练: python main.py pretrain")
        return True
    else:
        print("❌ 部分测试失败，请检查安装和配置。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
