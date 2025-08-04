#!/usr/bin/env python3
"""
测试所有推荐的命令
验证它们是否能正确执行
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def run_command(command, description, timeout=30):
    """运行命令并返回结果"""
    print(f"\n🧪 测试: {description}")
    print(f"命令: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT
        )
        
        if result.returncode == 0:
            print("✅ 命令执行成功")
            if result.stdout:
                print("输出:")
                print(result.stdout[-500:])  # 显示最后500字符
            return True
        else:
            print("❌ 命令执行失败")
            print(f"返回码: {result.returncode}")
            if result.stderr:
                print("错误:")
                print(result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 命令超时 ({timeout}秒)")
        return False
    except Exception as e:
        print(f"❌ 命令异常: {e}")
        return False

def test_basic_commands():
    """测试基础命令"""
    print("🚀 测试基础命令")
    print("=" * 60)
    
    commands = [
        ("python test_simple.py", "基础环境测试", 30),
        ("python start.py test", "启动脚本测试", 30),
        ("python check_project_status.py", "项目状态检查", 30),
        ("python test_api.py", "API功能测试", 60),
    ]
    
    results = []
    
    for command, description, timeout in commands:
        success = run_command(command, description, timeout)
        results.append((description, success))
        
        # 短暂休息
        time.sleep(1)
    
    return results

def test_import_commands():
    """测试导入相关命令"""
    print("\n🔍 测试导入功能")
    print("=" * 60)
    
    import_tests = [
        ("配置系统", "from config.config import get_config; print('配置导入成功')"),
        ("日志系统", "from utils.logger import get_logger; print('日志导入成功')"),
        ("存储模块", "from storage.redis_manager import RedisManager; print('存储导入成功')"),
        ("API模块", "from run_simple_api import create_simple_api; print('API导入成功')"),
    ]
    
    results = []
    
    for description, code in import_tests:
        print(f"\n🧪 测试: {description}")
        print(f"代码: {code}")
        print("-" * 50)
        
        try:
            result = subprocess.run([
                sys.executable, "-c", code
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ 导入成功")
                print(f"输出: {result.stdout.strip()}")
                results.append((description, True))
            else:
                print("❌ 导入失败")
                print(f"错误: {result.stderr.strip()}")
                results.append((description, False))
                
        except Exception as e:
            print(f"❌ 导入异常: {e}")
            results.append((description, False))
    
    return results

def test_file_existence():
    """测试关键文件是否存在"""
    print("\n📁 测试文件完整性")
    print("=" * 60)
    
    key_files = [
        "test_simple.py",
        "start.py", 
        "run_simple_api.py",
        "test_api.py",
        "check_project_status.py",
        "config/config.py",
        "utils/logger.py",
        "README.md",
        "TROUBLESHOOTING.md",
        "FINAL_STATUS.md"
    ]
    
    results = []
    
    for file_path in key_files:
        path = PROJECT_ROOT / file_path
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {file_path}")
        results.append((file_path, exists))
    
    return results

def main():
    """主函数"""
    print("🧪 RAG2项目命令验证测试")
    print("=" * 80)
    
    # 测试文件完整性
    file_results = test_file_existence()
    
    # 测试导入功能
    import_results = test_import_commands()
    
    # 测试基础命令
    command_results = test_basic_commands()
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)
    
    print("\n📁 文件完整性:")
    file_success = sum(1 for _, success in file_results if success)
    print(f"  {file_success}/{len(file_results)} 文件存在")
    
    print("\n🔍 导入功能:")
    import_success = sum(1 for _, success in import_results if success)
    for description, success in import_results:
        status = "✅" if success else "❌"
        print(f"  {status} {description}")
    print(f"  总计: {import_success}/{len(import_results)} 成功")
    
    print("\n🧪 命令执行:")
    command_success = sum(1 for _, success in command_results if success)
    for description, success in command_results:
        status = "✅" if success else "❌"
        print(f"  {status} {description}")
    print(f"  总计: {command_success}/{len(command_results)} 成功")
    
    # 总体评估
    total_tests = len(file_results) + len(import_results) + len(command_results)
    total_success = file_success + import_success + command_success
    
    print(f"\n📈 总体成功率: {total_success}/{total_tests} ({total_success/total_tests*100:.1f}%)")
    
    if total_success == total_tests:
        print("\n🎉 所有测试通过！项目完全可用。")
        print("\n📋 推荐使用命令:")
        print("  python test_simple.py      # 基础测试")
        print("  python start.py test       # 完整测试")
        print("  python test_api.py         # API测试")
        print("  python run_simple_api.py   # 启动简化API")
        return 0
    elif total_success >= total_tests * 0.8:
        print("\n✅ 大部分测试通过，项目基本可用。")
        print("  建议先使用基础功能，逐步完善。")
        return 0
    else:
        print("\n⚠️  多项测试失败，需要进一步修复。")
        print("  请查看上述错误信息并修复问题。")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        sys.exit(1)
