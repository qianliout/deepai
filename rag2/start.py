#!/usr/bin/env python3
"""
RAG2项目统一启动脚本
解决所有导入问题的统一入口
"""

import sys
import os
import asyncio
from pathlib import Path

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def setup_environment():
    """设置环境"""
    # 设置环境变量
    os.environ.setdefault('PYTHONPATH', str(PROJECT_ROOT))
    
    # 创建必要的目录
    (PROJECT_ROOT / "data" / "logs").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "documents").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "temp").mkdir(parents=True, exist_ok=True)

def test_imports():
    """测试导入"""
    print("🔍 测试模块导入...")
    
    try:
        from config.config import get_config
        print("✅ 配置模块导入成功")
        
        from utils.logger import get_logger
        print("✅ 日志模块导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

async def test_basic_setup():
    """运行基础设置测试"""
    print("\n🧪 运行基础设置测试...")
    
    try:
        # 导入测试模块
        from config.config import get_config
        from utils.logger import get_logger
        
        # 测试配置
        config = get_config()
        print(f"✅ 配置加载成功: 环境={config.environment}")
        
        # 测试日志
        logger = get_logger("test")
        logger.info("测试日志消息")
        print("✅ 日志系统正常")
        
        # 测试数据库连接（如果可用）
        try:
            from storage.postgresql_manager import PostgreSQLManager
            from storage.redis_manager import RedisManager
            from storage.mysql_manager import MySQLManager
            
            print("✅ 存储模块导入成功")
            
            # 简单的连接测试
            pg_manager = PostgreSQLManager()
            redis_manager = RedisManager()
            mysql_manager = MySQLManager()
            
            print("✅ 存储管理器创建成功")
            
        except Exception as e:
            print(f"⚠️  存储模块测试跳过: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础测试失败: {e}")
        return False

def start_api():
    """启动API服务"""
    print("\n🚀 启动API服务...")
    
    try:
        import uvicorn
        
        # 直接使用模块路径
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError:
        print("❌ uvicorn未安装，请运行: pip install uvicorn")
    except Exception as e:
        print(f"❌ API启动失败: {e}")

def show_help():
    """显示帮助信息"""
    print("""
🚀 RAG2项目启动脚本

用法:
    python start.py [命令]

命令:
    test        - 运行导入和基础功能测试
    api         - 启动API服务
    help        - 显示此帮助信息

示例:
    python start.py test     # 测试系统
    python start.py api      # 启动API服务
    python start.py          # 默认运行测试
""")

async def main():
    """主函数"""
    print("🚀 RAG2项目启动器")
    print("=" * 50)
    
    # 设置环境
    setup_environment()
    
    # 获取命令行参数
    command = sys.argv[1] if len(sys.argv) > 1 else "test"
    
    if command == "help":
        show_help()
        return 0
    
    elif command == "test":
        print("📋 运行测试模式...")
        
        # 测试导入
        if not test_imports():
            return 1
        
        # 测试基础功能
        if not await test_basic_setup():
            return 1
        
        print("\n🎉 所有测试通过！")
        print("\n📋 下一步:")
        print("1. 启动API服务: python start.py api")
        print("2. 访问API文档: http://localhost:8000/docs")
        
        return 0
    
    elif command == "api":
        print("📋 启动API服务...")
        
        # 先测试导入
        if not test_imports():
            print("❌ 导入测试失败，无法启动API服务")
            return 1
        
        # 启动API
        start_api()
        return 0
    
    else:
        print(f"❌ 未知命令: {command}")
        show_help()
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n程序运行错误: {e}")
        sys.exit(1)
