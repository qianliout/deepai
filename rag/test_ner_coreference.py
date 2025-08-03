"""
NER和指代消解功能测试

该测试文件演示了完整的NER和指代消解功能，包括：
1. 实体识别测试
2. 指代消解测试
3. 对话状态管理测试
4. 查询重写测试
5. 完整的多轮对话测试

运行方式：
python test_ner_coreference.py
"""

import sys
import time
from typing import List, Dict, Any

# 添加项目路径
sys.path.append('.')

from logger import get_logger
from llm import LLMManager
from ner_manager import NERManager, Entity, EntityType
from coreference_resolver import CoreferenceManager
from dialogue_state_manager import DialogueStateManager
from query_rewriter import QueryRewriteManager
from rag_chain import RAGChain


def test_ner_functionality():
    """测试NER功能"""
    print("\n" + "="*60)
    print("🔍 测试实体识别功能")
    print("="*60)
    
    try:
        # 初始化LLM和NER管理器
        llm = LLMManager()
        ner_manager = NERManager(llm)
        
        # 测试文本
        test_texts = [
            "请查看服务器192.168.1.100的CPU使用率",
            "web-server-01的内存占用情况如何？",
            "检查mysql服务在主机db-server-02上的运行状态",
            "用户ID为user123的订单order456有什么问题？",
            "联系邮箱admin@example.com，电话号码是13800138000"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📝 测试文本 {i}: {text}")
            
            # 提取实体
            entities = ner_manager.extract_entities(text)
            
            print(f"✅ 识别到 {len(entities)} 个实体:")
            for entity in entities:
                print(f"   - {entity.entity_type.value}: {entity.text} (置信度: {entity.confidence:.2f})")
        
        print("\n✅ NER功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ NER功能测试失败: {e}")
        return False


def test_coreference_functionality():
    """测试指代消解功能"""
    print("\n" + "="*60)
    print("🔗 测试指代消解功能")
    print("="*60)
    
    try:
        # 初始化组件
        llm = LLMManager()
        ner_manager = NERManager(llm)
        coreference_manager = CoreferenceManager(llm)
        
        session_id = "test_session_001"
        
        # 模拟多轮对话
        dialogue_turns = [
            "请查看服务器web-server-01的状态",
            "它的CPU使用率如何？",
            "那内存占用呢？",
            "这个服务器上运行的mysql服务正常吗？"
        ]
        
        for i, text in enumerate(dialogue_turns, 1):
            print(f"\n📝 对话轮次 {i}: {text}")
            
            # 提取实体
            entities = ner_manager.extract_entities(text)
            print(f"   实体: {[f'{e.entity_type.value}:{e.text}' for e in entities]}")
            
            # 指代消解
            result = coreference_manager.process_text(text, session_id, entities)
            
            print(f"   原始文本: {result['original_text']}")
            print(f"   消解后文本: {result['resolved_text']}")
            
            if result['has_references']:
                print(f"   指代关系: {len(result['references'])} 个")
                for ref in result['references']:
                    if isinstance(ref, dict) and ref.get('resolved_entity'):
                        print(f"     - '{ref['text']}' -> {ref['resolved_entity']['text']}")
        
        print("\n✅ 指代消解功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 指代消解功能测试失败: {e}")
        return False


def test_dialogue_state_management():
    """测试对话状态管理功能"""
    print("\n" + "="*60)
    print("💬 测试对话状态管理功能")
    print("="*60)
    
    try:
        # 初始化组件
        llm = LLMManager()
        ner_manager = NERManager(llm)
        dialogue_manager = DialogueStateManager()
        
        session_id = "test_session_002"
        
        # 模拟对话序列
        dialogue_sequence = [
            "查看服务器web-server-01的状态",
            "检查它的CPU和内存使用情况",
            "mysql服务运行正常吗？",
            "那redis服务呢？"
        ]
        
        for i, user_input in enumerate(dialogue_sequence, 1):
            print(f"\n📝 对话轮次 {i}: {user_input}")
            
            # 提取实体
            entities = ner_manager.extract_entities(user_input)
            
            # 更新对话状态
            state = dialogue_manager.update_state(session_id, user_input, entities)
            
            print(f"   对话轮次: {state.turn_count}")
            print(f"   活跃实体: {len(state.active_entities)}")
            print(f"   已填充槽位: {len([s for s in state.slots.values() if s.is_filled()])}")
            
            if state.last_intent:
                print(f"   当前意图: {state.last_intent.intent_type.value}")
            
            if state.context_focus:
                print(f"   上下文焦点: {state.context_focus}")
        
        # 获取状态摘要
        summary = dialogue_manager.get_state_summary(session_id)
        print(f"\n📊 对话状态摘要:")
        print(f"   总轮次: {summary['turn_count']}")
        print(f"   活跃实体数: {summary['active_entities_count']}")
        print(f"   槽位填充率: {summary['slot_summary']['fill_rate']:.2f}")
        
        print("\n✅ 对话状态管理功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 对话状态管理功能测试失败: {e}")
        return False


def test_query_rewriting():
    """测试查询重写功能"""
    print("\n" + "="*60)
    print("✏️ 测试查询重写功能")
    print("="*60)
    
    try:
        # 初始化组件
        llm = LLMManager()
        ner_manager = NERManager(llm)
        dialogue_manager = DialogueStateManager()
        query_rewriter = QueryRewriteManager(llm)
        
        session_id = "test_session_003"
        
        # 建立上下文
        context_queries = [
            "查看服务器web-server-01的状态",
            "它的CPU使用率如何？"
        ]
        
        for query in context_queries:
            entities = ner_manager.extract_entities(query)
            dialogue_manager.update_state(session_id, query, entities)
        
        # 测试需要重写的查询
        test_queries = [
            "那内存占用呢？",
            "这个服务器正常吗？",
            "它的磁盘空间还有多少？",
            "mysql服务状态如何？"
        ]
        
        for query in test_queries:
            print(f"\n📝 原始查询: {query}")
            
            # 获取对话状态
            state = dialogue_manager.get_state(session_id)
            
            # 查询重写
            result = query_rewriter.process_query(query, state)
            
            print(f"   重写后查询: {result['rewritten_query']}")
            print(f"   是否需要重写: {result['rewrite_needed']}")
            print(f"   重写类型: {result['rewrite_type']}")
            print(f"   置信度: {result['confidence']:.2f}")
            
            if result.get('reasoning'):
                print(f"   重写推理: {result['reasoning']}")
        
        print("\n✅ 查询重写功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 查询重写功能测试失败: {e}")
        return False


def test_integrated_rag_system():
    """测试完整的RAG系统集成"""
    print("\n" + "="*60)
    print("🚀 测试完整RAG系统集成")
    print("="*60)
    
    try:
        # 初始化RAG系统
        print("📚 初始化RAG系统...")
        rag = RAGChain()
        
        # 模拟多轮对话
        dialogue_sequence = [
            "什么是机器学习？",
            "它有哪些应用场景？",
            "深度学习和机器学习有什么区别？",
            "那神经网络呢？"
        ]
        
        for i, question in enumerate(dialogue_sequence, 1):
            print(f"\n📝 问题 {i}: {question}")
            
            # 使用增强查询功能
            response = rag.enhanced_query(
                question, 
                top_k=3,
                enable_ner=True,
                enable_coreference=True,
                enable_query_rewrite=True
            )
            
            print(f"   识别实体: {len(response.entities)} 个")
            print(f"   指代消解: {len(response.resolved_references)} 个")
            print(f"   重写查询: {response.rewritten_query}")
            print(f"   检索时间: {response.retrieval_time:.3f}s")
            print(f"   生成时间: {response.generation_time:.3f}s")
            print(f"   回答: {response.answer[:100]}...")
        
        # 获取对话状态
        dialogue_state = rag.get_dialogue_state()
        print(f"\n📊 最终对话状态:")
        print(f"   对话轮次: {dialogue_state.get('turn_count', 0)}")
        print(f"   活跃实体: {dialogue_state.get('active_entities_count', 0)}")
        
        print("\n✅ 完整RAG系统集成测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 完整RAG系统集成测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 NER和指代消解功能综合测试")
    print("="*60)
    
    # 测试结果统计
    test_results = []
    
    # 运行各项测试
    test_functions = [
        ("实体识别功能", test_ner_functionality),
        ("指代消解功能", test_coreference_functionality),
        ("对话状态管理", test_dialogue_state_management),
        ("查询重写功能", test_query_rewriting),
        ("完整RAG系统集成", test_integrated_rag_system)
    ]
    
    for test_name, test_func in test_functions:
        print(f"\n🔄 开始测试: {test_name}")
        start_time = time.time()
        
        try:
            result = test_func()
            test_results.append((test_name, result, time.time() - start_time))
        except Exception as e:
            print(f"❌ 测试 {test_name} 出现异常: {e}")
            test_results.append((test_name, False, time.time() - start_time))
    
    # 输出测试总结
    print("\n" + "="*60)
    print("📋 测试总结")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result, duration in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} {test_name} (耗时: {duration:.2f}s)")
        if result:
            passed += 1
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！NER和指代消解功能运行正常。")
    else:
        print("⚠️ 部分测试失败，请检查相关功能。")


if __name__ == "__main__":
    main()
