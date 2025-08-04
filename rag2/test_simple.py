#!/usr/bin/env python3
"""
RAG2项目简单测试
只测试基础功能，避免复杂依赖
"""

import sys
import os
from pathlib import Path

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def test_python_environment():
    """测试Python环境"""
    print("🐍 Python环境检查:")
    print(f"  版本: {sys.version}")
    print(f"  路径: {sys.executable}")
    print(f"  项目路径: {PROJECT_ROOT}")
    
    # 检查Python版本
    if sys.version_info < (3, 9):
        print("❌ Python版本过低，需要3.9+")
        return False
    else:
        print("✅ Python版本符合要求")
        return True

def test_basic_imports():
    """测试基础导入"""
    print("\n📦 基础模块导入测试:")
    
    # 测试标准库
    try:
        import json, os, sys, pathlib
        print("✅ 标准库导入正常")
    except Exception as e:
        print(f"❌ 标准库导入失败: {e}")
        return False
    
    # 测试第三方库
    missing_packages = []
    
    try:
        import torch
        print("✅ PyTorch导入成功")
    except ImportError:
        missing_packages.append("torch")
        print("❌ PyTorch未安装")
    
    try:
        import numpy
        print("✅ NumPy导入成功")
    except ImportError:
        missing_packages.append("numpy")
        print("❌ NumPy未安装")
    
    try:
        import fastapi
        print("✅ FastAPI导入成功")
    except ImportError:
        missing_packages.append("fastapi")
        print("❌ FastAPI未安装")
    
    try:
        import loguru
        print("✅ Loguru导入成功")
    except ImportError:
        missing_packages.append("loguru")
        print("❌ Loguru未安装")
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements_basic.txt")
        return False
    
    return True

def test_project_structure():
    """测试项目结构"""
    print("\n📁 项目结构检查:")
    
    required_dirs = [
        "config",
        "utils", 
        "api",
        "core",
        "models",
        "storage",
        "retrieval",
        "data"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ 目录存在")
        else:
            print(f"❌ {dir_name}/ 目录缺失")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"⚠️  缺少目录: {', '.join(missing_dirs)}")
        return False
    
    return True

def test_config_loading():
    """测试配置加载"""
    print("\n⚙️  配置系统测试:")
    
    try:
        # 创建基础的环境变量
        os.environ.setdefault('RAG_ENV', 'development')
        os.environ.setdefault('MODEL_DEVICE', 'cpu')
        
        from config.config import get_config
        config = get_config()
        
        print(f"✅ 配置加载成功")
        print(f"  环境: {config.environment}")
        print(f"  调试模式: {config.debug}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_logging():
    """测试日志系统"""
    print("\n📝 日志系统测试:")
    
    try:
        from utils.logger import get_logger
        
        logger = get_logger("test")
        logger.info("这是一条测试日志消息")
        
        print("✅ 日志系统正常")
        return True
        
    except Exception as e:
        print(f"❌ 日志系统失败: {e}")
        return False

def test_device_support():
    """测试设备支持"""
    print("\n🖥️  设备支持检查:")
    
    try:
        import torch
        
        print(f"  CPU支持: ✅")
        
        if torch.backends.mps.is_available():
            print(f"  MPS支持: ✅ (Mac M1优化可用)")
        else:
            print(f"  MPS支持: ❌ (不可用)")
        
        if torch.cuda.is_available():
            print(f"  CUDA支持: ✅")
        else:
            print(f"  CUDA支持: ❌ (不可用)")
        
        return True
        
    except Exception as e:
        print(f"❌ 设备检查失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 RAG2项目简单测试")
    print("=" * 50)
    
    tests = [
        ("Python环境", test_python_environment),
        ("基础导入", test_basic_imports),
        ("项目结构", test_project_structure),
        ("配置加载", test_config_loading),
        ("日志系统", test_logging),
        ("设备支持", test_device_support)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
    
    # 总结
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
        print("\n📋 下一步:")
        print("1. 安装完整依赖: pip install -r requirements.txt")
        print("2. 启动Docker服务: docker-compose up -d")
        print("3. 运行完整测试: python start.py test")
        print("4. 启动API服务: python start.py api")
        return 0
    else:
        print("⚠️  部分测试失败")
        print("\n🔧 建议:")
        if passed < total // 2:
            print("1. 检查Python环境和依赖安装")
            print("2. 安装基础依赖: pip install -r requirements_basic.txt")
        else:
            print("1. 基础环境正常，可以尝试安装完整依赖")
            print("2. pip install -r requirements.txt")
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
