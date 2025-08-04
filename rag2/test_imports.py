#!/usr/bin/env python3
"""
测试所有模块导入是否正常
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有关键模块的导入"""
    print("🔍 测试模块导入...")
    
    tests = []
    
    # 测试配置模块
    try:
        from config.config import get_config
        from config.environment_config import DEV_CONFIG, PROD_CONFIG
        tests.append(("✅", "配置模块"))
    except Exception as e:
        tests.append(("❌", f"配置模块: {e}"))
    
    # 测试日志模块
    try:
        from utils.logger import get_logger
        tests.append(("✅", "日志模块"))
    except Exception as e:
        tests.append(("❌", f"日志模块: {e}"))
    
    # 测试存储模块
    try:
        from storage.postgresql_manager import PostgreSQLManager
        from storage.mysql_manager import MySQLManager
        from storage.redis_manager import RedisManager
        tests.append(("✅", "存储模块"))
    except Exception as e:
        tests.append(("❌", f"存储模块: {e}"))
    
    # 测试模型模块
    try:
        from models.llm_client import LLMClient
        from models.embeddings import EmbeddingManager
        from models.rerank_models import RerankManager
        tests.append(("✅", "模型模块"))
    except Exception as e:
        tests.append(("❌", f"模型模块: {e}"))
    
    # 测试核心模块
    try:
        from core.document_processor import DocumentProcessor
        from core.rag_pipeline import RAGPipeline
        tests.append(("✅", "核心模块"))
    except Exception as e:
        tests.append(("❌", f"核心模块: {e}"))
    
    # 测试检索模块
    try:
        from retrieval.base_retriever import BaseRetriever
        from retrieval.semantic_retriever import SemanticRetriever
        tests.append(("✅", "检索模块"))
    except Exception as e:
        tests.append(("❌", f"检索模块: {e}"))
    
    # 显示结果
    print("\n📊 导入测试结果:")
    print("-" * 40)
    
    success_count = 0
    for status, description in tests:
        print(f"{status} {description}")
        if status == "✅":
            success_count += 1
    
    print("-" * 40)
    print(f"总计: {success_count}/{len(tests)} 成功")
    
    if success_count == len(tests):
        print("🎉 所有模块导入成功！")
        return True
    else:
        print("⚠️  部分模块导入失败")
        return False

def test_basic_functionality():
    """测试基础功能"""
    print("\n🧪 测试基础功能...")
    
    try:
        # 测试配置加载
        from config.config import get_config
        config = get_config()
        print(f"✅ 配置加载成功: 环境={config.environment}")
        
        # 测试日志系统
        from utils.logger import get_logger
        logger = get_logger("test")
        logger.info("测试日志消息")
        print("✅ 日志系统正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 RAG2模块导入测试")
    print("=" * 50)
    
    # 显示Python环境信息
    print(f"Python版本: {sys.version}")
    print(f"项目路径: {project_root}")
    print(f"Python路径: {sys.path[0]}")
    
    # 测试导入
    import_success = test_imports()
    
    # 测试基础功能
    if import_success:
        func_success = test_basic_functionality()
    else:
        func_success = False
    
    # 总结
    print("\n" + "=" * 50)
    if import_success and func_success:
        print("🎉 所有测试通过！可以继续运行其他脚本。")
        return 0
    else:
        print("❌ 测试失败，请检查依赖安装和环境配置。")
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
