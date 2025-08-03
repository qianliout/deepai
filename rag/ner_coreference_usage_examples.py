"""
NER和指代消解功能使用示例

该文件展示了如何在RAG系统中使用NER和指代消解功能，包括：
1. 基础实体识别示例
2. 多轮对话中的指代消解示例
3. 对话状态管理示例
4. 查询重写示例
5. 完整的RAG增强查询示例

使用方式：
python ner_coreference_usage_examples.py
"""

import sys
import json
from typing import List, Dict, Any

# 添加项目路径
sys.path.append('.')

from logger import get_logger
from ner_manager import NERManager, Entity, EntityType, RegexEntityExtractor
from coreference_resolver import CoreferenceManager, Reference, ReferenceType
from dialogue_state_manager import DialogueStateManager, Slot, SlotStatus
from query_rewriter import QueryRewriteManager


def example_1_basic_ner():
    """示例1: 基础实体识别"""
    print("\n" + "="*60)
    print("📝 示例1: 基础实体识别")
    print("="*60)
    
    # 初始化正则实体提取器
    extractor = RegexEntityExtractor()
    
    # 示例文本
    texts = [
        "请检查服务器192.168.1.100的状态",
        "联系管理员admin@company.com，电话13800138000",
        "访问网站https://www.example.com查看文档",
        "主机web-server-01和db-server-02需要维护"
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"\n📄 文本{i}: {text}")
        entities = extractor.extract(text)
        
        if entities:
            print("🔍 识别的实体:")
            for entity in entities:
                print(f"   • {entity.entity_type.value}: '{entity.text}' (置信度: {entity.confidence:.2f})")
        else:
            print("   未识别到实体")


def example_2_dialogue_state_management():
    """示例2: 对话状态管理"""
    print("\n" + "="*60)
    print("💬 示例2: 对话状态管理")
    print("="*60)
    
    # 初始化组件
    extractor = RegexEntityExtractor()
    dialogue_manager = DialogueStateManager()
    session_id = "demo_session_001"
    
    # 模拟多轮对话
    conversation = [
        "查看服务器web-server-01的状态",
        "检查它的CPU使用率",
        "内存占用情况如何？",
        "这个服务器上的mysql服务正常吗？"
    ]
    
    print("🗣️ 模拟对话:")
    for turn, user_input in enumerate(conversation, 1):
        print(f"\n👤 用户 (轮次{turn}): {user_input}")
        
        # 提取实体
        entities = extractor.extract(user_input)
        
        # 更新对话状态
        state = dialogue_manager.update_state(session_id, user_input, entities)
        
        print(f"🤖 系统分析:")
        print(f"   • 识别实体: {len(entities)} 个")
        print(f"   • 对话轮次: {state.turn_count}")
        print(f"   • 活跃实体: {len(state.active_entities)} 个")
        
        if state.last_intent:
            print(f"   • 当前意图: {state.last_intent.intent_type.value}")
        
        if state.context_focus:
            print(f"   • 上下文焦点: {state.context_focus}")
    
    # 显示最终状态摘要
    summary = dialogue_manager.get_state_summary(session_id)
    print(f"\n📊 对话状态摘要:")
    print(f"   • 总轮次: {summary['turn_count']}")
    print(f"   • 活跃实体数: {summary['active_entities_count']}")
    print(f"   • 槽位填充率: {summary['slot_summary']['fill_rate']:.2%}")


def example_3_coreference_resolution():
    """示例3: 指代消解演示"""
    print("\n" + "="*60)
    print("🔗 示例3: 指代消解演示")
    print("="*60)
    
    # 创建示例指代关系
    print("📝 创建指代关系示例:")
    
    # 原始实体
    server_entity = Entity(
        text="web-server-01",
        entity_type=EntityType.HOSTNAME,
        start_pos=0,
        end_pos=12,
        confidence=0.9,
        source="regex"
    )
    
    # 指代关系
    pronoun_ref = Reference(
        text="它",
        reference_type=ReferenceType.PRONOUN,
        start_pos=15,
        end_pos=16,
        confidence=0.8,
        resolved_entity=server_entity
    )
    
    definite_ref = Reference(
        text="这个服务器",
        reference_type=ReferenceType.DEFINITE,
        start_pos=20,
        end_pos=25,
        confidence=0.85,
        resolved_entity=server_entity
    )
    
    print(f"🎯 原始实体: {server_entity.text} ({server_entity.entity_type.value})")
    print(f"🔗 代词指代: '{pronoun_ref.text}' -> {pronoun_ref.resolved_entity.text}")
    print(f"🔗 定指指代: '{definite_ref.text}' -> {definite_ref.resolved_entity.text}")
    
    # 演示指代消解过程
    print(f"\n📖 指代消解示例:")
    original_texts = [
        "web-server-01需要重启",
        "它的CPU使用率过高",
        "这个服务器的内存也不足"
    ]
    
    resolved_texts = [
        "web-server-01需要重启",
        "web-server-01的CPU使用率过高",
        "web-server-01的内存也不足"
    ]
    
    for i, (original, resolved) in enumerate(zip(original_texts, resolved_texts), 1):
        print(f"   {i}. 原文: {original}")
        print(f"      消解: {resolved}")


def example_4_query_rewriting():
    """示例4: 查询重写演示"""
    print("\n" + "="*60)
    print("✏️ 示例4: 查询重写演示")
    print("="*60)
    
    # 模拟查询重写场景
    print("📝 查询重写场景:")
    
    scenarios = [
        {
            "context": "用户之前询问了服务器web-server-01的状态",
            "original_query": "它的CPU使用率如何？",
            "rewritten_query": "web-server-01的CPU使用率如何？",
            "rewrite_type": "指代消解"
        },
        {
            "context": "用户正在讨论机器学习模型",
            "original_query": "那深度学习呢？",
            "rewritten_query": "深度学习和机器学习有什么区别？",
            "rewrite_type": "上下文补全"
        },
        {
            "context": "用户询问了数据库性能",
            "original_query": "优化方法有哪些？",
            "rewritten_query": "数据库性能优化方法有哪些？",
            "rewrite_type": "主题补全"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🔄 场景{i}: {scenario['rewrite_type']}")
        print(f"   📋 上下文: {scenario['context']}")
        print(f"   ❓ 原始查询: {scenario['original_query']}")
        print(f"   ✅ 重写查询: {scenario['rewritten_query']}")


def example_5_entity_types_showcase():
    """示例5: 实体类型展示"""
    print("\n" + "="*60)
    print("🏷️ 示例5: 支持的实体类型展示")
    print("="*60)
    
    # 展示各种实体类型的示例
    entity_examples = {
        EntityType.IP_ADDRESS: ["192.168.1.100", "10.0.0.1"],
        EntityType.EMAIL: ["admin@company.com", "user@example.org"],
        EntityType.PHONE: ["13800138000", "021-12345678"],
        EntityType.URL: ["https://www.example.com", "http://api.service.com"],
        EntityType.HOSTNAME: ["web-server-01", "db-cluster-master"],
        EntityType.SERVICE: ["mysql", "nginx", "redis"],
        EntityType.ERROR_CODE: ["404", "500", "ERR_001"],
        EntityType.USER_ID: ["user123", "admin001"],
        EntityType.ORDER_ID: ["order456", "ORD-2024-001"],
        EntityType.PRODUCT: ["iPhone", "MacBook Pro"]
    }
    
    print("📋 实体类型和示例:")
    for entity_type, examples in entity_examples.items():
        print(f"   • {entity_type.value}: {', '.join(examples)}")


def example_6_slot_filling():
    """示例6: 槽位填充演示"""
    print("\n" + "="*60)
    print("🎯 示例6: 槽位填充演示")
    print("="*60)
    
    # 创建槽位示例
    slots = [
        Slot(
            name="hostname",
            entity_type=EntityType.HOSTNAME,
            value="web-server-01",
            confidence=0.9,
            status=SlotStatus.FILLED
        ),
        Slot(
            name="service",
            entity_type=EntityType.SERVICE,
            value="mysql",
            confidence=0.85,
            status=SlotStatus.CONFIRMED
        ),
        Slot(
            name="metric",
            entity_type=EntityType.METRIC,
            value="CPU使用率",
            confidence=0.8,
            status=SlotStatus.UPDATED
        )
    ]
    
    print("📊 槽位填充状态:")
    for slot in slots:
        status_emoji = {
            SlotStatus.EMPTY: "⭕",
            SlotStatus.FILLED: "✅",
            SlotStatus.CONFIRMED: "🔒",
            SlotStatus.UPDATED: "🔄",
            SlotStatus.EXPIRED: "⏰"
        }
        
        emoji = status_emoji.get(slot.status, "❓")
        print(f"   {emoji} {slot.name}: {slot.value} ({slot.status.value})")


def example_7_integration_workflow():
    """示例7: 完整集成工作流程"""
    print("\n" + "="*60)
    print("🚀 示例7: 完整集成工作流程")
    print("="*60)
    
    print("📋 NER和指代消解在RAG系统中的完整工作流程:")
    
    workflow_steps = [
        "1. 用户输入问题",
        "2. 实体识别 (NER)",
        "   • 正则表达式提取结构化实体",
        "   • LLM提取自然语言实体",
        "   • 实体去重和合并",
        "3. 对话状态更新",
        "   • 更新活跃实体",
        "   • 填充相关槽位",
        "   • 识别用户意图",
        "4. 指代消解",
        "   • 检测指代词",
        "   • 匹配历史实体",
        "   • 生成消解文本",
        "5. 查询重写",
        "   • 分析查询完整性",
        "   • 补充上下文信息",
        "   • 生成增强查询",
        "6. 文档检索",
        "   • 使用增强查询检索",
        "   • 混合检索策略",
        "7. 回答生成",
        "   • 构建增强提示词",
        "   • 包含实体和状态信息",
        "   • 生成上下文相关回答"
    ]
    
    for step in workflow_steps:
        if step.startswith("   "):
            print(f"     {step[3:]}")
        else:
            print(f"  {step}")


def main():
    """主函数 - 运行所有示例"""
    print("🎯 NER和指代消解功能使用示例")
    print("="*60)
    print("本示例展示了在RAG系统中如何使用NER和指代消解功能")
    
    # 运行所有示例
    examples = [
        example_1_basic_ner,
        example_2_dialogue_state_management,
        example_3_coreference_resolution,
        example_4_query_rewriting,
        example_5_entity_types_showcase,
        example_6_slot_filling,
        example_7_integration_workflow
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ 示例执行失败: {e}")
    
    print("\n" + "="*60)
    print("✅ 所有示例演示完成！")
    print("💡 提示: 这些功能已集成到RAG系统中，可以通过enhanced_query方法使用")
    print("="*60)


if __name__ == "__main__":
    main()
