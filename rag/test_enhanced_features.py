#!/usr/bin/env python3
"""
RAG系统增强功能测试脚本

测试以下增强功能：
1. 系统检查功能（check.py）
2. Elasticsearch文档存储
3. MySQL对话存储
4. Redis动态上下文压缩
5. ES粗排+向量精排混合检索

使用方法：
python test_enhanced_features.py
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
    print("\n" + "="*60)
    print("测试1: 系统检查功能")
    print("="*60)
    
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
        
        # 显示详细结果
        print(f"\n📋 详细检查结果:")
        for result in results['results']:
            status_icon = {"success": "✅", "warning": "⚠️", "error": "❌"}.get(result['status'], "❓")
            print(f"   {status_icon} {result['check_name']}: {result['message']}")
        
        print("\n✅ 系统检查测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 系统检查测试失败: {e}")
        return False


def test_elasticsearch_storage():
    """测试Elasticsearch文档存储"""
    print("\n" + "="*60)
    print("测试2: Elasticsearch文档存储")
    print("="*60)
    
    try:
        from elasticsearch_manager import ElasticsearchManager, DocumentRecord
        from datetime import datetime
        
        es_manager = ElasticsearchManager()
        
        # 测试文档索引
        test_docs = [
            {
                "title": "人工智能基础",
                "content": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                "keywords": ["人工智能", "机器学习", "深度学习"]
            },
            {
                "title": "机器学习算法",
                "content": "机器学习包括监督学习、无监督学习和强化学习三大类算法。",
                "keywords": ["机器学习", "算法", "监督学习"]
            }
        ]
        
        print("📝 索引测试文档...")
        for i, doc_data in enumerate(test_docs):
            doc_record = DocumentRecord(
                doc_id=f"test_doc_{i+1}",
                title=doc_data["title"],
                content=doc_data["content"],
                source="test",
                doc_type="article",
                metadata={"test": True},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                content_length=len(doc_data["content"]),
                keywords=doc_data["keywords"]
            )
            
            success = es_manager.index_document(doc_record)
            print(f"   文档 {i+1}: {'成功' if success else '失败'}")
        
        # 测试搜索
        print(f"\n🔍 测试搜索功能...")
        search_queries = ["人工智能", "机器学习算法", "深度学习"]
        
        for query in search_queries:
            results = es_manager.search_documents(query, size=5)
            print(f"   查询 '{query}': 找到 {len(results)} 个结果")
            for result in results:
                print(f"     - {result.title} (分数: {result.score:.3f})")
        
        # 测试连接信息
        connection_info = es_manager.get_connection_info()
        print(f"\n📊 ES连接信息: {connection_info}")
        
        print("\n✅ Elasticsearch存储测试完成")
        return True
        
    except Exception as e:
        print(f"❌ Elasticsearch存储测试失败: {e}")
        return False


def test_mysql_storage():
    """测试MySQL对话存储"""
    print("\n" + "="*60)
    print("测试3: MySQL对话存储")
    print("="*60)
    
    try:
        from mysql_manager import MySQLManager, ConversationData
        import uuid
        
        mysql_manager = MySQLManager()
        
        # 创建测试会话
        test_session_id = str(uuid.uuid4())
        print(f"📝 创建测试会话: {test_session_id[:8]}...")
        
        success = mysql_manager.create_session(
            test_session_id, 
            user_id="test_user",
            title="测试会话"
        )
        print(f"   会话创建: {'成功' if success else '失败'}")
        
        # 保存测试对话
        test_conversations = [
            ("user", "什么是人工智能？"),
            ("assistant", "人工智能是计算机科学的一个分支..."),
            ("user", "机器学习有哪些类型？"),
            ("assistant", "机器学习主要分为监督学习、无监督学习和强化学习...")
        ]
        
        print(f"\n💬 保存测试对话...")
        for role, content in test_conversations:
            conversation_data = ConversationData(
                session_id=test_session_id,
                role=role,
                content=content,
                user_id="test_user",
                processing_time=0.5 if role == "assistant" else None
            )
            
            success = mysql_manager.save_conversation(conversation_data)
            print(f"   {role}: {'成功' if success else '失败'}")
        
        # 获取会话对话记录
        print(f"\n📚 获取会话对话记录...")
        conversations = mysql_manager.get_session_conversations(test_session_id)
        print(f"   获取到 {len(conversations)} 条对话记录")
        
        for conv in conversations:
            print(f"     [{conv['role']}]: {conv['content'][:50]}...")
        
        # 获取统计信息
        stats = mysql_manager.get_conversation_stats(days=1)
        print(f"\n📊 对话统计: {stats}")
        
        # 测试连接信息
        connection_info = mysql_manager.get_connection_info()
        print(f"\n📊 MySQL连接信息: {connection_info}")
        
        print("\n✅ MySQL存储测试完成")
        return True
        
    except Exception as e:
        print(f"❌ MySQL存储测试失败: {e}")
        return False


def test_context_compression():
    """测试Redis动态上下文压缩"""
    print("\n" + "="*60)
    print("测试4: Redis动态上下文压缩")
    print("="*60)
    
    try:
        from context_manager import ContextManager
        import uuid
        
        context_manager = ContextManager()
        test_session_id = str(uuid.uuid4())
        
        print(f"📝 测试上下文管理: {test_session_id[:8]}...")
        
        # 添加多条消息以触发压缩
        test_messages = [
            ("user", "请介绍一下人工智能的发展历史，包括早期的研究和重要的里程碑事件。"),
            ("assistant", "人工智能的发展历史可以追溯到20世纪40年代。早期的重要里程碑包括：1950年图灵提出图灵测试，1956年达特茅斯会议正式确立了人工智能这一学科..."),
            ("user", "机器学习和深度学习有什么区别？请详细解释一下。"),
            ("assistant", "机器学习和深度学习的主要区别在于：机器学习是一个更广泛的概念，包括各种算法和技术；而深度学习是机器学习的一个子集，专门使用深度神经网络..."),
            ("user", "现在的大语言模型是如何工作的？"),
            ("assistant", "大语言模型基于Transformer架构，通过自注意力机制处理文本序列。它们在大量文本数据上进行预训练，学习语言的统计规律和语义关系..."),
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
        print(f"   压缩消息数: {stats.get('compressed_messages', 0)}")
        print(f"   利用率: {stats.get('utilization', 0):.2%}")
        
        # 获取上下文消息
        context_messages = context_manager.get_context_messages(test_session_id, max_tokens=1000)
        print(f"\n📚 获取上下文消息 (限制1000 tokens): {len(context_messages)} 条")
        
        for msg in context_messages:
            compressed_flag = "[压缩]" if msg.is_compressed else ""
            print(f"   {compressed_flag}[{msg.role}]: {msg.content[:50]}... (tokens: {msg.token_count})")
        
        print("\n✅ 上下文压缩测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 上下文压缩测试失败: {e}")
        return False


def test_hybrid_retrieval():
    """测试混合检索功能"""
    print("\n" + "="*60)
    print("测试5: ES粗排+向量精排混合检索")
    print("="*60)
    
    try:
        from retriever import HybridRetrieverManager
        from vector_store import VectorStoreManager
        from embeddings import EmbeddingManager
        
        print("🔧 初始化混合检索器...")
        embedding_manager = EmbeddingManager()
        vector_store = VectorStoreManager(embedding_manager)
        retriever = HybridRetrieverManager(vector_store, embedding_manager)
        
        # 测试检索
        test_queries = [
            "人工智能的应用领域",
            "机器学习算法分类",
            "深度学习神经网络"
        ]
        
        print(f"\n🔍 测试混合检索...")
        for query in test_queries:
            print(f"\n   查询: '{query}'")
            
            try:
                results = retriever.retrieve(
                    query, 
                    top_k=5, 
                    es_candidates=20,
                    use_query_expansion=True
                )
                
                print(f"     检索结果: {len(results)} 个")
                for i, result in enumerate(results, 1):
                    print(f"       {i}. 方法: {result.retrieval_method}, 分数: {result.score:.3f}")
                    if result.es_score:
                        print(f"          ES分数: {result.es_score:.3f}, 向量分数: {result.vector_score:.3f}")
                    
            except Exception as e:
                print(f"     检索失败: {e}")
        
        # 获取检索统计
        stats = retriever.get_retrieval_stats()
        print(f"\n📊 检索统计:")
        print(f"   检索方法: {stats.get('retrieval_method', 'unknown')}")
        print(f"   混合模式: {stats.get('hybrid_mode', False)}")
        print(f"   查询扩展: {stats.get('query_expansion_enabled', False)}")
        
        if 'elasticsearch_info' in stats:
            es_info = stats['elasticsearch_info']
            print(f"   ES连接: {es_info.get('connected', False)}")
        
        print("\n✅ 混合检索测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 混合检索测试失败: {e}")
        return False


def test_integrated_rag_system():
    """测试集成的RAG系统"""
    print("\n" + "="*60)
    print("测试6: 集成RAG系统")
    print("="*60)
    
    try:
        from rag_chain import RAGChain
        
        print("🚀 初始化增强RAG系统...")
        rag = RAGChain()
        
        print(f"   当前会话ID: {rag.get_session_id()[:8]}...")
        
        # 测试查询
        test_questions = [
            "什么是人工智能？",
            "请解释机器学习的基本概念",
            "深度学习有哪些应用？"
        ]
        
        print(f"\n💬 测试增强对话功能...")
        for i, question in enumerate(test_questions, 1):
            print(f"\n   问题 {i}: {question}")
            
            try:
                answer = rag.query(question, top_k=3, save_to_session=True)
                print(f"   回答: {answer[:100]}...")
                
            except Exception as e:
                print(f"   查询失败: {e}")
        
        # 获取系统统计
        print(f"\n📊 系统统计信息:")
        stats = rag.get_stats()
        
        print(f"   会话ID: {stats.get('system_info', {}).get('current_session_id', 'N/A')[:8]}...")
        print(f"   混合检索: {stats.get('system_info', {}).get('hybrid_retrieval', False)}")
        
        storage_health = stats.get('storage_health', {})
        print(f"   存储健康状态:")
        print(f"     Redis: {'✅' if storage_health.get('redis') else '❌'}")
        print(f"     MySQL: {'✅' if storage_health.get('mysql') else '❌'}")
        print(f"     向量库: {'✅' if storage_health.get('vector_store') else '❌'}")
        print(f"     ES: {'✅' if storage_health.get('elasticsearch') else '❌'}")
        
        context_stats = stats.get('context_management', {})
        if context_stats.get('exists'):
            print(f"   上下文管理:")
            print(f"     消息数: {context_stats.get('total_messages', 0)}")
            print(f"     Token数: {context_stats.get('total_tokens', 0)}")
            print(f"     压缩比: {context_stats.get('compression_ratio', 0):.2%}")
        
        conversation_stats = stats.get('conversation_stats', {})
        if conversation_stats:
            print(f"   对话统计:")
            print(f"     总对话数: {conversation_stats.get('total_conversations', 0)}")
            print(f"     总会话数: {conversation_stats.get('total_sessions', 0)}")
        
        print("\n✅ 集成RAG系统测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 集成RAG系统测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 RAG系统增强功能测试")
    print("="*80)
    
    # 加载配置
    load_env_config()
    
    # 设置日志
    logger = get_logger("TestEnhancedFeatures")
    logger.info("开始RAG增强功能测试")
    
    # 运行测试
    test_functions = [
        test_system_check,
        test_elasticsearch_storage,
        test_mysql_storage,
        test_context_compression,
        test_hybrid_retrieval,
        test_integrated_rag_system
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
    print("\n" + "="*80)
    print("🎯 测试结果总结")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过测试: {passed}/{total}")
    
    test_names = [
        "系统检查功能",
        "Elasticsearch文档存储",
        "MySQL对话存储", 
        "Redis动态上下文压缩",
        "ES粗排+向量精排混合检索",
        "集成RAG系统"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{i+1:2d}. {name}: {status}")
    
    if passed == total:
        print("\n🎉 所有增强功能测试通过！RAG系统升级成功")
    else:
        print(f"\n⚠️  {total-passed} 个测试失败，请检查相关配置和服务")
    
    print("\n🔧 增强功能说明:")
    print("1. ✅ 系统启动前自动检查配置和依赖")
    print("2. ✅ 文档数据存储到Elasticsearch，支持关键词检索")
    print("3. ✅ 对话数据持久化存储到MySQL")
    print("4. ✅ Redis上下文管理，支持动态压缩减少token占用")
    print("5. ✅ ES粗排+向量精排的混合检索策略")
    print("6. ✅ 完整的多存储系统集成和监控")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
