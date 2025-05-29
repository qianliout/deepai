#!/usr/bin/env python3
"""
简单的配置测试脚本
"""

import os
import subprocess
import sys

def test_project_config(project_name):
    """测试单个项目的配置"""
    print(f"\n🔍 测试{project_name}项目配置...")
    
    # 切换到项目目录并运行配置测试
    cmd = f"cd {project_name} && python -c \"import config; print('✅ 配置导入成功'); config.create_directories() if hasattr(config, 'create_directories') else print('⚠️  没有create_directories函数')\""
    
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
        "/Users/liuqianli/work/python/deepai/saved_model/transformer", 
        "/Users/liuqianli/work/python/deepai/saved_model/transformer2",
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

def main():
    """主函数"""
    print("🚀 开始简单配置测试...")
    print("=" * 60)
    
    projects = ["bert", "transformer", "transformer2"]
    results = []
    
    for project in projects:
        results.append(test_project_config(project))
    
    # 验证目录
    verify_directories()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    
    if all(results):
        print("🎉 所有配置测试通过！")
        return True
    else:
        print("❌ 部分配置测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
