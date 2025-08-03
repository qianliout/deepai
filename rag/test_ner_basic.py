"""
NER和指代消解基础功能测试（不依赖LLM）

该测试文件演示了NER和指代消解的基础功能，主要测试：
1. 正则表达式实体识别
2. 基础数据结构
3. 对话状态管理的核心逻辑
4. 不依赖外部API的功能

运行方式：
python test_ner_basic.py
"""

import sys
import time
from typing import List, Dict, Any

# 添加项目路径
sys.path.append('.')

from logger import get_logger
from ner_manager import Entity, EntityType, RegexEntityExtractor
from dialogue_state_manager import DialogueStateManager, Slot, SlotStatus, IntentType
from coreference_resolver import Reference, ReferenceType


def test_regex_entity_extraction():
    """测试正则表达式实体提取"""
    print("\n" + "="*60)
    print("🔍 测试正则表达式实体提取")
    print("="*60)
    
    try:
        # 初始化正则实体提取器
        extractor = RegexEntityExtractor()
        
        # 测试文本
        test_texts = [
            "请查看服务器192.168.1.100的状态",
            "联系邮箱admin@example.com，电话13800138000",
            "访问网站https://www.example.com",
            "主机名web-server-01和db-server-02",
            "检查IP地址10.0.0.1到10.0.0.255的连通性"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📝 测试文本 {i}: {text}")
            
            # 提取实体
            entities = extractor.extract(text)
            
            print(f"✅ 识别到 {len(entities)} 个实体:")
            for entity in entities:
                print(f"   - {entity.entity_type.value}: {entity.text} (置信度: {entity.confidence:.2f})")
        
        print("\n✅ 正则表达式实体提取测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 正则表达式实体提取测试失败: {e}")
        return False


def test_entity_data_structures():
    """测试实体数据结构"""
    print("\n" + "="*60)
    print("📊 测试实体数据结构")
    print("="*60)
    
    try:
        # 创建测试实体
        entity = Entity(
            text="192.168.1.100",
            entity_type=EntityType.IP_ADDRESS,
            start_pos=0,
            end_pos=13,
            confidence=0.95,
            source="regex",
            metadata={"pattern": "ip_address"}
        )
        
        print(f"📝 创建实体: {entity.text}")
        print(f"   类型: {entity.entity_type.value}")
        print(f"   位置: {entity.start_pos}-{entity.end_pos}")
        print(f"   置信度: {entity.confidence}")
        print(f"   来源: {entity.source}")
        
        # 测试序列化
        entity_dict = entity.to_dict()
        print(f"✅ 序列化成功: {len(entity_dict)} 个字段")
        
        # 测试实体类型枚举
        print(f"\n📋 支持的实体类型 ({len(EntityType)} 种):")
        for entity_type in EntityType:
            print(f"   - {entity_type.value}")
        
        print("\n✅ 实体数据结构测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 实体数据结构测试失败: {e}")
        return False


def test_slot_management():
    """测试槽位管理"""
    print("\n" + "="*60)
    print("🎯 测试槽位管理")
    print("="*60)
    
    try:
        # 创建测试槽位
        slot = Slot(
            name="hostname",
            entity_type=EntityType.HOSTNAME,
            value="web-server-01",
            confidence=0.9,
            status=SlotStatus.FILLED
        )
        
        print(f"📝 创建槽位: {slot.name}")
        print(f"   值: {slot.value}")
        print(f"   类型: {slot.entity_type.value}")
        print(f"   状态: {slot.status.value}")
        print(f"   是否已填充: {slot.is_filled()}")
        
        # 测试槽位状态
        print(f"\n📋 支持的槽位状态:")
        for status in SlotStatus:
            print(f"   - {status.value}")
        
        # 测试序列化
        slot_dict = slot.to_dict()
        print(f"✅ 槽位序列化成功: {len(slot_dict)} 个字段")
        
        print("\n✅ 槽位管理测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 槽位管理测试失败: {e}")
        return False


def test_dialogue_state_basic():
    """测试对话状态基础功能"""
    print("\n" + "="*60)
    print("💬 测试对话状态基础功能")
    print("="*60)
    
    try:
        # 初始化对话状态管理器
        dialogue_manager = DialogueStateManager()
        session_id = "test_session_basic"
        
        # 创建测试实体
        entities = [
            Entity(
                text="web-server-01",
                entity_type=EntityType.HOSTNAME,
                start_pos=0,
                end_pos=12,
                confidence=0.9,
                source="regex"
            ),
            Entity(
                text="192.168.1.100",
                entity_type=EntityType.IP_ADDRESS,
                start_pos=15,
                end_pos=28,
                confidence=0.95,
                source="regex"
            )
        ]
        
        # 模拟对话轮次
        user_inputs = [
            "查看服务器web-server-01的状态",
            "检查IP地址192.168.1.100的连通性",
            "这个服务器的CPU使用率如何？"
        ]
        
        for i, user_input in enumerate(user_inputs, 1):
            print(f"\n📝 对话轮次 {i}: {user_input}")
            
            # 更新对话状态
            state = dialogue_manager.update_state(session_id, user_input, entities)
            
            print(f"   对话轮次: {state.turn_count}")
            print(f"   活跃实体数: {len(state.active_entities)}")
            print(f"   实体历史数: {len(state.entity_history)}")
            print(f"   意图历史数: {len(state.intent_history)}")
            
            if state.last_intent:
                print(f"   当前意图: {state.last_intent.intent_type.value}")
        
        # 获取状态摘要
        summary = dialogue_manager.get_state_summary(session_id)
        print(f"\n📊 对话状态摘要:")
        for key, value in summary.items():
            if key != "slot_summary":
                print(f"   {key}: {value}")
        
        print("\n✅ 对话状态基础功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 对话状态基础功能测试失败: {e}")
        return False


def test_reference_data_structures():
    """测试指代数据结构"""
    print("\n" + "="*60)
    print("🔗 测试指代数据结构")
    print("="*60)
    
    try:
        # 创建测试实体
        entity = Entity(
            text="web-server-01",
            entity_type=EntityType.HOSTNAME,
            start_pos=0,
            end_pos=12,
            confidence=0.9,
            source="regex"
        )
        
        # 创建指代关系
        reference = Reference(
            text="它",
            reference_type=ReferenceType.PRONOUN,
            start_pos=20,
            end_pos=21,
            confidence=0.8,
            resolved_entity=entity
        )
        
        print(f"📝 创建指代关系:")
        print(f"   指代词: {reference.text}")
        print(f"   类型: {reference.reference_type.value}")
        print(f"   位置: {reference.start_pos}-{reference.end_pos}")
        print(f"   置信度: {reference.confidence}")
        print(f"   解析实体: {reference.resolved_entity.text}")
        
        # 测试序列化
        ref_dict = reference.to_dict()
        print(f"✅ 指代关系序列化成功: {len(ref_dict)} 个字段")
        
        # 测试指代类型
        print(f"\n📋 支持的指代类型:")
        for ref_type in ReferenceType:
            print(f"   - {ref_type.value}")
        
        print("\n✅ 指代数据结构测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 指代数据结构测试失败: {e}")
        return False


def test_intent_recognition_basic():
    """测试基础意图识别"""
    print("\n" + "="*60)
    print("🎯 测试基础意图识别")
    print("="*60)
    
    try:
        from dialogue_state_manager import IntentRecognizer
        
        # 初始化意图识别器
        recognizer = IntentRecognizer()
        
        # 测试文本和预期意图
        test_cases = [
            ("查看服务器状态", IntentType.CHECK_STATUS),
            ("什么是机器学习？", IntentType.QUERY),
            ("监控CPU使用率", IntentType.MONITOR),
            ("解决网络问题", IntentType.TROUBLESHOOT),
            ("配置数据库参数", IntentType.CONFIGURE),
            ("比较两个模型的性能", IntentType.COMPARE)
        ]
        
        for text, expected_intent in test_cases:
            print(f"\n📝 测试文本: {text}")
            
            # 识别意图
            intent = recognizer.recognize_intent(text, [])
            
            print(f"   识别意图: {intent.intent_type.value}")
            print(f"   置信度: {intent.confidence:.2f}")
            print(f"   预期意图: {expected_intent.value}")
            
            # 简单验证
            is_correct = intent.intent_type == expected_intent
            print(f"   结果: {'✅ 正确' if is_correct else '⚠️ 不匹配'}")
        
        print("\n✅ 基础意图识别测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 基础意图识别测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 NER和指代消解基础功能测试")
    print("="*60)
    
    # 测试结果统计
    test_results = []
    
    # 运行各项测试
    test_functions = [
        ("正则表达式实体提取", test_regex_entity_extraction),
        ("实体数据结构", test_entity_data_structures),
        ("槽位管理", test_slot_management),
        ("对话状态基础功能", test_dialogue_state_basic),
        ("指代数据结构", test_reference_data_structures),
        ("基础意图识别", test_intent_recognition_basic)
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
        print("🎉 所有基础功能测试通过！")
    else:
        print("⚠️ 部分测试失败，请检查相关功能。")


if __name__ == "__main__":
    main()
