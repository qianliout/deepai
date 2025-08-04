#!/usr/bin/env python3
"""
RAG2项目基础设置测试脚本
测试配置加载、日志系统和基本组件
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 现在可以直接导入
from config.config import get_config, get_model_config
from utils.logger import get_logger, log_query, log_retrieval, LogContext

logger = get_logger("test_basic_setup")

async def test_configuration():
    """测试配置系统"""
    logger.info("=== 测试配置系统 ===")
    
    try:
        # 测试基本配置
        config = get_config()
        logger.info(f"当前环境: {config.environment}")
        logger.info(f"调试模式: {config.debug}")
        
        # 测试数据库配置
        logger.info(f"PostgreSQL URL: {config.get_postgres_url()}")
        logger.info(f"MySQL URL: {config.get_mysql_url()}")
        logger.info(f"Redis URL: {config.get_redis_url()}")
        logger.info(f"Elasticsearch URL: {config.get_es_url()}")
        logger.info(f"Neo4j URI: {config.database.neo4j_uri}")
        
        # 测试模型配置
        model_config = get_model_config()
        logger.info(f"LLM提供商: {model_config['llm']['provider']}")
        logger.info(f"LLM模型: {model_config['llm']['model_name']}")
        logger.info(f"嵌入模型: {model_config['embedding']['model_name']}")
        logger.info(f"重排序模型: {model_config['reranker']['model_name']}")
        
        logger.info("✅ 配置系统测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 配置系统测试失败: {str(e)}")
        return False

async def test_logging_system():
    """测试日志系统"""
    logger.info("=== 测试日志系统 ===")
    
    try:
        # 测试不同级别的日志
        logger.debug("这是一条调试日志")
        logger.info("这是一条信息日志")
        logger.warning("这是一条警告日志")
        logger.error("这是一条错误日志")
        
        # 测试专用日志记录器
        log_query("测试查询", user_id="test_user", session_id="test_session")
        log_retrieval("测试查询", retrieved_count=5, method="semantic", duration=0.1)
        
        # 测试日志上下文管理器
        with LogContext("测试操作", operation_type="test"):
            await asyncio.sleep(0.1)  # 模拟操作
        
        logger.info("✅ 日志系统测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 日志系统测试失败: {str(e)}")
        return False

async def test_database_connections():
    """测试数据库连接"""
    logger.info("=== 测试数据库连接 ===")
    
    results = {}
    
    # 测试PostgreSQL连接
    try:
        from storage.postgresql_manager import PostgreSQLManager
        pg_manager = PostgreSQLManager()
        await pg_manager.initialize()
        
        health = await pg_manager.health_check()
        results["PostgreSQL"] = health
        logger.info(f"PostgreSQL连接: {'✅ 成功' if health else '❌ 失败'}")
        
        await pg_manager.close()
        
    except Exception as e:
        logger.error(f"PostgreSQL连接测试失败: {str(e)}")
        results["PostgreSQL"] = False
    
    # 测试Redis连接
    try:
        from storage.redis_manager import RedisManager
        redis_manager = RedisManager()
        await redis_manager.initialize()
        
        health = await redis_manager.health_check()
        results["Redis"] = health
        logger.info(f"Redis连接: {'✅ 成功' if health else '❌ 失败'}")
        
        await redis_manager.close()
        
    except Exception as e:
        logger.error(f"Redis连接测试失败: {str(e)}")
        results["Redis"] = False
    
    # 测试MySQL连接
    try:
        from storage.mysql_manager import MySQLManager
        mysql_manager = MySQLManager()
        await mysql_manager.initialize()
        
        health = await mysql_manager.health_check()
        results["MySQL"] = health
        logger.info(f"MySQL连接: {'✅ 成功' if health else '❌ 失败'}")
        
        await mysql_manager.close()
        
    except Exception as e:
        logger.error(f"MySQL连接测试失败: {str(e)}")
        results["MySQL"] = False
    
    # 汇总结果
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    logger.info(f"数据库连接测试完成: {success_count}/{total_count} 成功")
    return success_count == total_count

async def test_model_loading():
    """测试模型加载"""
    logger.info("=== 测试模型加载 ===")
    
    results = {}
    
    # 测试嵌入模型
    try:
        from models.embeddings import get_embedding_manager
        embedding_manager = get_embedding_manager()
        
        # 测试编码
        test_text = "这是一个测试文本"
        embedding = embedding_manager.encode_text(test_text)
        
        health = embedding_manager.health_check()
        results["嵌入模型"] = health and len(embedding) > 0
        logger.info(f"嵌入模型: {'✅ 成功' if results['嵌入模型'] else '❌ 失败'}")
        
    except Exception as e:
        logger.error(f"嵌入模型测试失败: {str(e)}")
        results["嵌入模型"] = False
    
    # 测试重排序模型
    try:
        from models.rerank_models import get_rerank_manager
        rerank_manager = get_rerank_manager()
        
        # 测试重排序
        test_query = "测试查询"
        test_docs = ["相关文档", "不相关文档"]
        rerank_results = rerank_manager.rerank_documents(test_query, test_docs)
        
        health = rerank_manager.health_check()
        results["重排序模型"] = health and len(rerank_results) > 0
        logger.info(f"重排序模型: {'✅ 成功' if results['重排序模型'] else '❌ 失败'}")
        
    except Exception as e:
        logger.error(f"重排序模型测试失败: {str(e)}")
        results["重排序模型"] = False
    
    # 测试LLM客户端
    try:
        from models.llm_client import get_llm_client
        llm_client = await get_llm_client()
        
        health = await llm_client.health_check()
        results["LLM客户端"] = health
        logger.info(f"LLM客户端: {'✅ 成功' if health else '❌ 失败'}")
        
    except Exception as e:
        logger.error(f"LLM客户端测试失败: {str(e)}")
        results["LLM客户端"] = False
    
    # 汇总结果
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    logger.info(f"模型加载测试完成: {success_count}/{total_count} 成功")
    return success_count > 0  # 至少有一个模型成功加载

async def main():
    """主测试函数"""
    logger.info("🚀 开始RAG2项目基础设置测试")
    
    test_results = []
    
    # 运行各项测试
    test_results.append(await test_configuration())
    test_results.append(await test_logging_system())
    test_results.append(await test_database_connections())
    test_results.append(await test_model_loading())
    
    # 汇总测试结果
    success_count = sum(1 for result in test_results if result)
    total_count = len(test_results)
    
    logger.info("=" * 50)
    logger.info(f"📊 测试总结: {success_count}/{total_count} 项测试通过")
    
    if success_count == total_count:
        logger.info("🎉 所有测试通过！RAG2项目基础设置完成")
        return 0
    else:
        logger.warning(f"⚠️  有 {total_count - success_count} 项测试失败，请检查相关配置")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试过程中发生未预期的错误: {str(e)}")
        sys.exit(1)
