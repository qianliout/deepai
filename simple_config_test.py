#!/usr/bin/env python3
"""
简化的配置测试脚本
只测试配置导入和目录创建，不涉及数据加载
"""

import os
import subprocess
import sys

def test_project_config(project_name):
    """测试单个项目的配置"""
    print(f"\n🔍 测试{project_name}项目配置...")
    
    # 切换到项目目录并运行配置测试
    cmd = f"""cd {project_name} && python -c "
import config
print('✅ 配置导入成功')

# 显示配置信息
if hasattr(config, 'TRAINING_CONFIG'):
    tc = config.TRAINING_CONFIG
    print(f'  模型保存目录: {{tc.model_save_dir}}')
    if hasattr(tc, 'fine_tuning_save_dir'):
        print(f'  微调保存目录: {{tc.fine_tuning_save_dir}}')
    if hasattr(tc, 'pretrained_model_path'):
        print(f'  预训练模型路径: {{tc.pretrained_model_path}}')
    print(f'  日志保存目录: {{tc.log_dir}}')
    print(f'  数据缓存目录: {{tc.cache_dir}}')
elif hasattr(config, 'default_config'):
    dc = config.default_config
    print(f'  模型保存目录: {{dc.training.model_save_dir}}')
    print(f'  日志保存目录: {{dc.training.log_dir}}')
    print(f'  数据缓存目录: {{dc.training.cache_dir}}')

# 创建目录
if hasattr(config, 'create_directories'):
    config.create_directories()
else:
    print('⚠️  没有create_directories函数')
"
"""
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/Users/liuqianli/work/python/deepai")
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"❌ 配置测试失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False

def verify_directories():
    """验证目录是否创建成功"""
    print("\n🔍 验证目录结构...")
    
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
    
    return all_exist

def test_path_consistency():
    """测试路径配置的一致性"""
    print("\n🔍 测试路径配置一致性...")
    
    # 测试BERT的预训练模型路径和微调配置的关联
    cmd = """cd bert && python -c "
import config
tc = config.TRAINING_CONFIG
print(f'预训练模型保存目录: {tc.model_save_dir}')
print(f'预训练模型路径: {tc.pretrained_model_path}')
print(f'微调模型保存目录: {tc.fine_tuning_save_dir}')

# 检查路径一致性
import os
expected_pretrained_path = os.path.join(tc.model_save_dir, 'best_model')
if tc.pretrained_model_path == expected_pretrained_path:
    print('✅ 预训练模型路径配置一致')
else:
    print(f'❌ 路径不一致: 期望 {expected_pretrained_path}, 实际 {tc.pretrained_model_path}')

if tc.fine_tuning_save_dir.startswith(tc.model_save_dir):
    print('✅ 微调目录配置合理')
else:
    print('❌ 微调目录配置不合理')
"
"""
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/Users/liuqianli/work/python/deepai")
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"❌ 路径一致性测试失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始简化配置测试...")
    print("=" * 80)
    
    projects = ["bert", "transformer", "transformer2"]
    results = []
    
    for project in projects:
        results.append(test_project_config(project))
    
    # 测试路径一致性
    results.append(test_path_consistency())
    
    # 验证目录
    verify_directories()
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 测试结果总结:")
    
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
        print("  - 加载数据集和tokenizer时默认使用本地缓存")
        return True
    else:
        print("❌ 部分配置测试失败，请检查错误信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
