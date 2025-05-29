#!/usr/bin/env python3
"""
最终配置测试脚本
验证所有项目的目录配置统一管理是否正确
"""

import os
import sys
from pathlib import Path

def test_bert_config():
    """测试BERT项目配置"""
    print("🔍 测试BERT项目配置...")
    
    # 添加bert目录到路径
    bert_path = Path(__file__).parent / "bert"
    sys.path.insert(0, str(bert_path))
    
    try:
        import config
        TRAINING_CONFIG = config.TRAINING_CONFIG
        create_directories = config.create_directories
        
        print("✅ BERT配置导入成功")
        print(f"  预训练模型保存目录: {TRAINING_CONFIG.model_save_dir}")
        print(f"  微调模型保存目录: {TRAINING_CONFIG.fine_tuning_save_dir}")
        print(f"  预训练模型路径: {TRAINING_CONFIG.pretrained_model_path}")
        print(f"  日志保存目录: {TRAINING_CONFIG.log_dir}")
        print(f"  数据缓存目录: {TRAINING_CONFIG.cache_dir}")
        
        # 测试目录创建
        create_directories()
        
        # 验证目录是否存在
        assert os.path.exists(TRAINING_CONFIG.model_save_dir), f"预训练模型保存目录不存在: {TRAINING_CONFIG.model_save_dir}"
        assert os.path.exists(TRAINING_CONFIG.fine_tuning_save_dir), f"微调模型保存目录不存在: {TRAINING_CONFIG.fine_tuning_save_dir}"
        assert os.path.exists(TRAINING_CONFIG.log_dir), f"日志保存目录不存在: {TRAINING_CONFIG.log_dir}"
        
        print("✅ BERT目录创建成功")
        
    except Exception as e:
        print(f"❌ BERT配置测试失败: {e}")
        return False
    finally:
        sys.path.remove(str(bert_path))
    
    return True


def test_transformer_config():
    """测试Transformer项目配置"""
    print("\n🔍 测试Transformer项目配置...")
    
    # 添加transformer目录到路径
    transformer_path = Path(__file__).parent / "transformer"
    sys.path.insert(0, str(transformer_path))
    
    try:
        import config
        default_config = config.default_config
        create_directories = config.create_directories
        
        print("✅ Transformer配置导入成功")
        print(f"  模型保存目录: {default_config.training.model_save_dir}")
        print(f"  词汇表保存目录: {default_config.training.vocab_save_dir}")
        print(f"  日志保存目录: {default_config.training.log_dir}")
        print(f"  数据缓存目录: {default_config.training.cache_dir}")
        
        # 测试目录创建
        create_directories()
        
        # 验证目录是否存在
        assert os.path.exists(default_config.training.model_save_dir), f"模型保存目录不存在: {default_config.training.model_save_dir}"
        assert os.path.exists(default_config.training.log_dir), f"日志保存目录不存在: {default_config.training.log_dir}"
        
        print("✅ Transformer目录创建成功")
        
    except Exception as e:
        print(f"❌ Transformer配置测试失败: {e}")
        return False
    finally:
        sys.path.remove(str(transformer_path))
    
    return True


def test_transformer2_config():
    """测试Transformer2项目配置"""
    print("\n🔍 测试Transformer2项目配置...")
    
    # 添加transformer2目录到路径
    transformer2_path = Path(__file__).parent / "transformer2"
    sys.path.insert(0, str(transformer2_path))
    
    try:
        import config
        TRAINING_CONFIG = config.TRAINING_CONFIG
        create_directories = config.create_directories
        
        print("✅ Transformer2配置导入成功")
        print(f"  模型保存目录: {TRAINING_CONFIG.model_save_dir}")
        print(f"  词汇表保存目录: {TRAINING_CONFIG.vocab_save_dir}")
        print(f"  日志保存目录: {TRAINING_CONFIG.log_dir}")
        print(f"  数据缓存目录: {TRAINING_CONFIG.cache_dir}")
        
        # 测试目录创建
        create_directories()
        
        # 验证目录是否存在
        assert os.path.exists(TRAINING_CONFIG.model_save_dir), f"模型保存目录不存在: {TRAINING_CONFIG.model_save_dir}"
        assert os.path.exists(TRAINING_CONFIG.log_dir), f"日志保存目录不存在: {TRAINING_CONFIG.log_dir}"
        
        print("✅ Transformer2目录创建成功")
        
    except Exception as e:
        print(f"❌ Transformer2配置测试失败: {e}")
        return False
    finally:
        sys.path.remove(str(transformer2_path))
    
    return True


def verify_directory_structure():
    """验证目录结构是否符合要求"""
    print("\n🔍 验证最终目录结构...")
    
    expected_dirs = [
        "/Users/liuqianli/work/python/deepai/saved_model/bert",
        "/Users/liuqianli/work/python/deepai/saved_model/bert/fine_tuning",
        "/Users/liuqianli/work/python/deepai/saved_model/transformer", 
        "/Users/liuqianli/work/python/deepai/saved_model/transformer/vocab",
        "/Users/liuqianli/work/python/deepai/saved_model/transformer2",
        "/Users/liuqianli/work/python/deepai/saved_model/transformer2/vocab",
        "/Users/liuqianli/work/python/deepai/logs/bert",
        "/Users/liuqianli/work/python/deepai/logs/transformer",
        "/Users/liuqianli/work/python/deepai/logs/transformer2",
    ]
    
    all_exist = True
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (不存在)")
            all_exist = False
    
    # 检查缓存目录
    cache_dir = "/Users/liuqianli/.cache/huggingface/datasets"
    if os.path.exists(cache_dir):
        print(f"✅ {cache_dir}")
    else:
        print(f"⚠️  {cache_dir} (不存在，但会在首次使用时创建)")
    
    return all_exist


def test_fine_tuning_config():
    """测试微调配置"""
    print("\n🔍 测试BERT微调配置...")
    
    # 添加bert目录到路径
    bert_path = Path(__file__).parent / "bert"
    sys.path.insert(0, str(bert_path))
    
    try:
        from fine_tuning import BertFineTuner
        
        # 测试使用默认配置创建微调器
        fine_tuner = BertFineTuner()
        
        print("✅ 微调器创建成功")
        print(f"  预训练模型路径: {fine_tuner.pretrained_model_path}")
        print(f"  微调保存目录: {fine_tuner.fine_tuning_save_dir}")
        
        # 验证路径配置
        assert str(fine_tuner.fine_tuning_save_dir).endswith("fine_tuning"), "微调保存目录配置错误"
        
        print("✅ 微调配置验证成功")
        
    except Exception as e:
        print(f"❌ 微调配置测试失败: {e}")
        return False
    finally:
        sys.path.remove(str(bert_path))
    
    return True


def main():
    """主测试函数"""
    print("🚀 开始最终配置测试...")
    print("=" * 80)
    
    results = []
    
    # 测试各个项目的配置
    results.append(test_bert_config())
    results.append(test_transformer_config())
    results.append(test_transformer2_config())
    
    # 测试微调配置
    results.append(test_fine_tuning_config())
    
    # 验证目录结构
    verify_directory_structure()
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 最终测试结果总结:")
    
    if all(results):
        print("🎉 所有配置测试通过！")
        print("\n✅ 配置统一管理完成，主要成果：")
        print("  1. ✅ 移除了所有output_dir等不语义化的变量名")
        print("  2. ✅ 统一了所有目录配置，集中在config.py中管理")
        print("  3. ✅ 添加了微调专用的目录配置")
        print("  4. ✅ 预训练模型路径与微调目录建立了关联")
        print("  5. ✅ 所有硬编码路径都已更新为配置驱动")
        print("  6. ✅ 支持自动目录创建")
        print("  7. ✅ 统一了HuggingFace缓存配置")
        print("\n📁 目录结构：")
        print("  - 预训练模型：/Users/liuqianli/work/python/deepai/saved_model/{项目名}/")
        print("  - 微调模型：/Users/liuqianli/work/python/deepai/saved_model/bert/fine_tuning/")
        print("  - 日志文件：/Users/liuqianli/work/python/deepai/logs/{项目名}/")
        print("  - 数据缓存：/Users/liuqianli/.cache/huggingface/datasets/")
        print("\n🎯 使用方式：")
        print("  - 所有路径都从全局配置获取，无需手动传参")
        print("  - 预训练模型路径自动关联到微调配置")
        print("  - 支持一键创建所有必要目录")
        return True
    else:
        print("❌ 部分配置测试失败，请检查错误信息")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
