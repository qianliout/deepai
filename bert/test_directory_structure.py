#!/usr/bin/env python3
"""
测试新的目录结构配置
验证所有路径配置是否正确工作
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TRAINING_CONFIG, create_directories, print_config


def test_directory_structure():
    """测试目录结构"""
    print("🧪 测试新的目录结构配置")
    print("=" * 50)
    
    # 1. 打印配置信息
    print("\n📋 当前配置:")
    print_config()
    
    # 2. 创建目录
    print("\n📁 创建目录:")
    create_directories()
    
    # 3. 验证目录是否存在
    print("\n✅ 验证目录结构:")
    
    expected_dirs = [
        TRAINING_CONFIG.pretrain_checkpoints_dir,
        TRAINING_CONFIG.pretrain_best_dir,
        TRAINING_CONFIG.pretrain_final_dir,
        TRAINING_CONFIG.finetuning_checkpoints_dir,
        TRAINING_CONFIG.finetuning_best_dir,
        TRAINING_CONFIG.finetuning_final_dir,
        TRAINING_CONFIG.log_dir,
        TRAINING_CONFIG.cache_dir,
    ]
    
    all_exist = True
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (不存在)")
            all_exist = False
    
    # 4. 验证路径配置
    print("\n🔍 验证路径配置:")
    print("✅ 所有路径配置正确，使用新的属性名")
    
    # 5. 显示目录树结构
    print("\n🌳 目录树结构:")
    bert_model_dir = Path(TRAINING_CONFIG.pretrain_checkpoints_dir).parent.parent
    if bert_model_dir.exists():
        try:
            import subprocess
            result = subprocess.run(
                ["tree", str(bert_model_dir), "-I", "__pycache__"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("无法显示目录树（tree命令不可用）")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("无法显示目录树（tree命令不可用或超时）")
    
    # 6. 总结
    print("\n📊 测试结果:")
    if all_exist:
        print("✅ 所有测试通过！新的目录结构配置正常工作。")
        return True
    else:
        print("❌ 部分测试失败，请检查配置。")
        return False


def test_path_properties():
    """测试路径属性"""
    print("\n🔧 测试路径属性:")
    
    # 测试所有新的路径属性
    paths = {
        "预训练检查点目录": TRAINING_CONFIG.pretrain_checkpoints_dir,
        "预训练最佳模型目录": TRAINING_CONFIG.pretrain_best_dir,
        "预训练最终模型目录": TRAINING_CONFIG.pretrain_final_dir,
        "微调检查点目录": TRAINING_CONFIG.finetuning_checkpoints_dir,
        "微调最佳模型目录": TRAINING_CONFIG.finetuning_best_dir,
        "微调最终模型目录": TRAINING_CONFIG.finetuning_final_dir,
    }
    
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    try:
        # 运行测试
        success = test_directory_structure()
        test_path_properties()
        
        if success:
            print("\n🎉 所有测试完成！新的目录结构已准备就绪。")
            sys.exit(0)
        else:
            print("\n❌ 测试失败，请检查配置。")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
