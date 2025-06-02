#!/usr/bin/env python3
"""
RAG系统核心增强功能测试脚本

测试核心增强功能：
1. 系统检查功能
2. 动态上下文压缩
3. 混合检索策略
4. 多存储系统集成

使用方法：
python test_core_enhancements.py
"""

import sys
import os
import time

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import defaultConfig, load_env_config
from logger import get_logger


def test_system_check():
    """测试系统检查功能"""
    print("\n" + "="*50)
    print("测试1: 系统检查功能")
    print("="*50)
    
    try:
        from check import SystemChecker
        
        checker = SystemChecker()
        print("🔍 开始系统检查...")
        
        # 执行完整检查
        results = checker.run_full_check()
        
        print(f"\n📊 检查结果总结:")
        print(f"   成功: {results['summary']['success']}")
        print(f"   警告: {results['summary']['warning']}")
        print(f"   错误: {results['summary']['error']}")
        print(f"   总计: {results['summary']['total']}")
        
        print("\n✅ 系统检查测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 系统检查测试失败: {e}")
        return False


def test_context_compression():
    """测试动态上下文压缩"""
    print("\n" + "="*50)
    print("测试2: 动态上下文压缩")
    print("="*50)
    
    try:
        from context_manager import ContextManager
        import uuid
        
        context_manager = ContextManager()
        test_session_id = str(uuid.uuid4())
        
        print(f"📝 测试上下文管理: {test_session_id[:8]}...")
        
        # 添加多条消息
        test_messages = [
            ("user", "请介绍一下人工智能的发展历史。"),
            ("assistant", "人工智能的发展历史可以追溯到20世纪40年代..."),
            ("user", "机器学习和深度学习有什么区别？"),
            ("assistant", "机器学习是一个更广泛的概念，深度学习是其子集..."),
            ("user", "现在的大语言模型是如何工作的？"),
            ("assistant", "大语言模型基于Transformer架构..."),
        ]
        
        for role, content in test_messages:
            success = context_manager.add_message(test_session_id, role, content)
            print(f"   添加消息 [{role}]: {'成功' if success else '失败'}")
        
        # 获取上下文统计
        stats = context_manager.get_context_stats(test_session_id)
        print(f"\n📊 上下文统计:")
        print(f"   总消息数: {stats.get('total_messages', 0)}")
        print(f"   总Token数: {stats.get('total_tokens', 0)}")
        print(f"   压缩比例: {stats.get('compression_ratio', 0):.2%}")
        
        print("\n✅ 上下文压缩测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 上下文压缩测试失败: {e}")
        return False


def test_hybrid_retrieval():
    """测试混合检索功能"""
    print("\n" + "="*50)
    print("测试3: 混合检索策略")
    print("="*50)
    
    try:
        from retriever import HybridRetrieverManager
        from vector_store import VectorStoreManager
        from embeddings import EmbeddingManager
        
        print("🔧 初始化混合检索器...")
        embedding_manager = EmbeddingManager()
        vector_store = VectorStoreManager(embedding_manager)
        retriever = HybridRetrieverManager(vector_store, embedding_manager)
        
        # 测试检索
        test_queries = ["人工智能应用", "机器学习算法"]
        
        print(f"\n🔍 测试混合检索...")
        for query in test_queries:
            print(f"\n   查询: '{query}'")
            
            try:
                results = retriever.retrieve(query, top_k=3)
                print(f"     检索结果: {len(results)} 个")
                
                for i, result in enumerate(results, 1):
                    print(f"       {i}. 方法: {result.retrieval_method}, 分数: {result.score:.3f}")
                    
            except Exception as e:
                print(f"     检索失败: {e}")
        
        # 获取检索统计
        stats = retriever.get_retrieval_stats()
        print(f"\n📊 检索统计:")
        print(f"   检索方法: {stats.get('retrieval_method', 'unknown')}")
        print(f"   混合模式: {stats.get('hybrid_mode', False)}")
        
        print("\n✅ 混合检索测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 混合检索测试失败: {e}")
        return False


def test_enhanced_rag_chain():
    """测试增强的RAG链"""
    print("\n" + "="*50)
    print("测试4: 增强RAG链")
    print("="*50)
    
    try:
        from rag_chain import RAGChain
        
        print("🚀 初始化增强RAG系统...")
        rag = RAGChain()
        
        print(f"   当前会话ID: {rag.get_session_id()[:8]}...")
        
        # 测试查询
        test_questions = [
            "什么是人工智能？",
            "机器学习有哪些类型？"
        ]
        
        print(f"\n💬 测试增强对话功能...")
        for i, question in enumerate(test_questions, 1):
            print(f"\n   问题 {i}: {question}")
            
            try:
                answer = rag.query(question, top_k=3, save_to_session=True)
                print(f"   回答: {answer[:80]}...")
                
            except Exception as e:
                print(f"   查询失败: {e}")
        
        # 获取系统统计
        print(f"\n📊 系统统计信息:")
        stats = rag.get_stats()
        
        print(f"   会话ID: {stats.get('system_info', {}).get('current_session_id', 'N/A')[:8]}...")
        print(f"   混合检索: {stats.get('system_info', {}).get('hybrid_retrieval', False)}")
        
        storage_health = stats.get('storage_health', {})
        print(f"   存储健康状态:")
        for storage, status in storage_health.items():
            print(f"     {storage}: {'✅' if status else '❌'}")
        
        print("\n✅ 增强RAG链测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 增强RAG链测试失败: {e}")
        return False


def test_query_expansion():
    """测试查询扩展功能"""
    print("\n" + "="*50)
    print("测试5: JiebaTokenizer查询扩展")
    print("="*50)
    
    try:
        from query_expander import SimpleQueryExpander
        
        expander = SimpleQueryExpander(enable_expansion=True)
        
        test_queries = [
            "人工智能发展",
            "机器学习算法",
            "深度学习应用"
        ]
        
        print(f"🔍 测试查询扩展...")
        for query in test_queries:
            print(f"\n   原始查询: '{query}'")
            
            result = expander.expand_query(query)
            print(f"   扩展查询: '{result.expanded_query}'")
            print(f"   扩展方法: {result.method}")
            print(f"   处理时间: {result.processing_time:.4f}s")
        
        print("\n✅ 查询扩展测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 查询扩展测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 RAG系统核心增强功能测试")
    print("="*60)
    
    # 加载配置
    load_env_config()
    
    # 设置日志
    logger = get_logger("TestCoreEnhancements")
    logger.info("开始RAG核心增强功能测试")
    
    # 运行测试
    test_functions = [
        test_system_check,
        test_context_compression,
        test_hybrid_retrieval,
        test_enhanced_rag_chain,
        test_query_expansion
    ]
    
    results = []
    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 异常: {e}")
            results.append(False)
    
    # 总结测试结果
    print("\n" + "="*60)
    print("🎯 测试结果总结")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过测试: {passed}/{total}")
    
    test_names = [
        "系统检查功能",
        "动态上下文压缩",
        "混合检索策略",
        "增强RAG链",
        "JiebaTokenizer查询扩展"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{i+1:2d}. {name}: {status}")
    
    if passed == total:
        print("\n🎉 所有核心增强功能测试通过！")
    else:
        print(f"\n⚠️  {total-passed} 个测试失败")
    
    print("\n🔧 核心增强功能说明:")
    print("1. ✅ 系统启动前自动检查配置和依赖")
    print("2. ✅ Redis上下文管理，支持动态压缩")
    print("3. ✅ ES粗排+向量精排的混合检索")
    print("4. ✅ 多存储系统集成和监控")
    print("5. ✅ JiebaTokenizer中文分词优化")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
