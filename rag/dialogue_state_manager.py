"""
对话状态管理模块

该模块负责管理多轮对话中的状态信息，包括：
1. 实体状态跟踪和槽位填充
2. 对话意图历史管理
3. 上下文窗口维护
4. 实体生命周期管理

数据流：
1. 用户输入 -> 实体提取 -> 状态更新 -> 槽位填充
2. 对话历史 -> 意图识别 -> 状态维护 -> 上下文管理
3. 实体过期 -> 状态清理 -> 内存优化

学习要点：
1. 对话状态的数据结构设计
2. 槽位填充的策略和算法
3. 实体生命周期管理
4. 多轮对话的状态一致性
"""

import json
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta

from logger import get_logger
from ner_manager import Entity, EntityType


class SlotStatus(Enum):
    """槽位状态枚举"""

    EMPTY = "EMPTY"  # 空槽位
    FILLED = "FILLED"  # 已填充
    CONFIRMED = "CONFIRMED"  # 已确认
    UPDATED = "UPDATED"  # 已更新
    EXPIRED = "EXPIRED"  # 已过期


class IntentType(Enum):
    """意图类型枚举"""

    QUERY = "QUERY"  # 查询意图
    CHECK_STATUS = "CHECK_STATUS"  # 状态检查
    MONITOR = "MONITOR"  # 监控意图
    TROUBLESHOOT = "TROUBLESHOOT"  # 故障排查
    CONFIGURE = "CONFIGURE"  # 配置意图
    COMPARE = "COMPARE"  # 比较意图
    FOLLOW_UP = "FOLLOW_UP"  # 后续询问
    CLARIFICATION = "CLARIFICATION"  # 澄清意图
    OTHER = "OTHER"  # 其他意图


@dataclass
class Slot:
    """槽位数据结构"""

    name: str  # 槽位名称
    entity_type: EntityType  # 实体类型
    value: Optional[str] = None  # 槽位值
    entity: Optional[Entity] = None  # 关联实体
    status: SlotStatus = SlotStatus.EMPTY  # 槽位状态
    confidence: float = 0.0  # 置信度
    last_updated: float = 0.0  # 最后更新时间
    update_count: int = 0  # 更新次数
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result["entity_type"] = self.entity_type.value
        result["status"] = self.status.value
        if self.entity:
            result["entity"] = self.entity.to_dict()
        return result

    def is_filled(self) -> bool:
        """检查槽位是否已填充"""
        return self.status in [
            SlotStatus.FILLED,
            SlotStatus.CONFIRMED,
            SlotStatus.UPDATED,
        ]

    def is_expired(self, expire_time: float = 3600) -> bool:
        """检查槽位是否已过期

        Args:
            expire_time: 过期时间（秒）

        Returns:
            是否已过期
        """
        return time.time() - self.last_updated > expire_time


@dataclass
class DialogueIntent:
    """对话意图数据结构"""

    intent_type: IntentType  # 意图类型
    confidence: float  # 置信度
    timestamp: float  # 时间戳
    turn_id: int  # 对话轮次ID
    entities: List[Entity] = field(default_factory=list)  # 相关实体
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result["intent_type"] = self.intent_type.value
        result["entities"] = [entity.to_dict() for entity in self.entities]
        return result


@dataclass
class DialogueState:
    """对话状态数据结构"""

    session_id: str  # 会话ID
    turn_count: int = 0  # 对话轮次
    slots: Dict[str, Slot] = field(default_factory=dict)  # 槽位字典
    intent_history: List[DialogueIntent] = field(default_factory=list)  # 意图历史
    active_entities: Dict[str, Entity] = field(default_factory=dict)  # 活跃实体
    entity_history: List[Entity] = field(default_factory=list)  # 实体历史
    last_intent: Optional[DialogueIntent] = None  # 最后意图
    context_focus: Optional[str] = None  # 上下文焦点
    created_at: float = field(default_factory=time.time)  # 创建时间
    last_updated: float = field(default_factory=time.time)  # 最后更新时间
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "slots": {k: v.to_dict() for k, v in self.slots.items()},
            "intent_history": [intent.to_dict() for intent in self.intent_history],
            "active_entities": {
                k: v.to_dict() for k, v in self.active_entities.items()
            },
            "entity_history": [entity.to_dict() for entity in self.entity_history],
            "last_intent": self.last_intent.to_dict() if self.last_intent else None,
            "context_focus": self.context_focus,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "metadata": self.metadata,
        }
        return result


class SlotManager:
    """槽位管理器

    负责槽位的创建、更新、验证和生命周期管理
    """

    def __init__(self):
        """初始化槽位管理器"""
        self.logger = get_logger("SlotManager")

        # 预定义槽位模板
        self.slot_templates = {
            "hostname": Slot("hostname", EntityType.HOSTNAME),
            "ip_address": Slot("ip_address", EntityType.IP_ADDRESS),
            "service": Slot("service", EntityType.SERVICE),
            "metric": Slot("metric", EntityType.METRIC),
            "time_range": Slot("time_range", EntityType.TIME_RANGE),
            "user_id": Slot("user_id", EntityType.USER_ID),
            "error_code": Slot("error_code", EntityType.ERROR_CODE),
            "product": Slot("product", EntityType.PRODUCT),
        }

    def fill_slots(
        self, state: DialogueState, entities: List[Entity]
    ) -> Dict[str, Slot]:
        """填充槽位

        Args:
            state: 对话状态
            entities: 实体列表

        Returns:
            更新后的槽位字典
        """
        updated_slots = {}

        for entity in entities:
            slot_name = self._get_slot_name_for_entity(entity)
            if slot_name:
                # 获取或创建槽位
                if slot_name in state.slots:
                    slot = state.slots[slot_name]
                else:
                    slot = self._create_slot_from_template(slot_name, entity)

                # 更新槽位
                old_value = slot.value
                slot.value = entity.text
                slot.entity = entity
                slot.confidence = entity.confidence
                slot.last_updated = time.time()
                slot.update_count += 1

                # 确定槽位状态
                if old_value != entity.text:
                    slot.status = SlotStatus.UPDATED if old_value else SlotStatus.FILLED
                else:
                    slot.status = SlotStatus.CONFIRMED

                updated_slots[slot_name] = slot
                self.logger.debug(f"槽位更新: {slot_name} = {entity.text}")

        return updated_slots

    def _get_slot_name_for_entity(self, entity: Entity) -> Optional[str]:
        """根据实体类型获取槽位名称

        Args:
            entity: 实体对象

        Returns:
            槽位名称
        """
        entity_to_slot = {
            EntityType.HOSTNAME: "hostname",
            EntityType.IP_ADDRESS: "ip_address",
            EntityType.SERVICE: "service",
            EntityType.METRIC: "metric",
            EntityType.TIME_RANGE: "time_range",
            EntityType.USER_ID: "user_id",
            EntityType.ERROR_CODE: "error_code",
            EntityType.PRODUCT: "product",
        }

        return entity_to_slot.get(entity.entity_type)

    def _create_slot_from_template(self, slot_name: str, entity: Entity) -> Slot:
        """从模板创建槽位

        Args:
            slot_name: 槽位名称
            entity: 实体对象

        Returns:
            新创建的槽位
        """
        if slot_name in self.slot_templates:
            template = self.slot_templates[slot_name]
            return Slot(
                name=template.name,
                entity_type=template.entity_type,
                value=entity.text,
                entity=entity,
                status=SlotStatus.FILLED,
                confidence=entity.confidence,
                last_updated=time.time(),
                update_count=1,
            )
        else:
            # 创建默认槽位
            return Slot(
                name=slot_name,
                entity_type=entity.entity_type,
                value=entity.text,
                entity=entity,
                status=SlotStatus.FILLED,
                confidence=entity.confidence,
                last_updated=time.time(),
                update_count=1,
            )

    def clean_expired_slots(
        self, slots: Dict[str, Slot], expire_time: float = 3600
    ) -> Dict[str, Slot]:
        """清理过期槽位

        Args:
            slots: 槽位字典
            expire_time: 过期时间（秒）

        Returns:
            清理后的槽位字典
        """
        cleaned_slots = {}
        expired_count = 0

        for name, slot in slots.items():
            if slot.is_expired(expire_time):
                slot.status = SlotStatus.EXPIRED
                expired_count += 1
                self.logger.debug(f"槽位过期: {name}")
            else:
                cleaned_slots[name] = slot

        if expired_count > 0:
            self.logger.info(f"清理了 {expired_count} 个过期槽位")

        return cleaned_slots

    def get_filled_slots(self, slots: Dict[str, Slot]) -> Dict[str, Slot]:
        """获取已填充的槽位

        Args:
            slots: 槽位字典

        Returns:
            已填充的槽位字典
        """
        return {name: slot for name, slot in slots.items() if slot.is_filled()}

    def get_slot_summary(self, slots: Dict[str, Slot]) -> Dict[str, Any]:
        """获取槽位摘要信息

        Args:
            slots: 槽位字典

        Returns:
            槽位摘要
        """
        total_slots = len(slots)
        filled_slots = len(self.get_filled_slots(slots))

        status_count = {}
        for slot in slots.values():
            status = slot.status.value
            status_count[status] = status_count.get(status, 0) + 1

        return {
            "total_slots": total_slots,
            "filled_slots": filled_slots,
            "fill_rate": filled_slots / total_slots if total_slots > 0 else 0.0,
            "status_distribution": status_count,
        }


class IntentRecognizer:
    """意图识别器

    负责识别用户的对话意图
    """

    def __init__(self):
        """初始化意图识别器"""
        self.logger = get_logger("IntentRecognizer")

        # 意图关键词映射
        self.intent_keywords = {
            IntentType.QUERY: ["查询", "查看", "显示", "获取", "什么", "如何", "怎么"],
            IntentType.CHECK_STATUS: ["状态", "情况", "运行", "正常", "异常", "健康"],
            IntentType.MONITOR: ["监控", "观察", "跟踪", "实时", "持续"],
            IntentType.TROUBLESHOOT: ["问题", "故障", "错误", "异常", "排查", "解决"],
            IntentType.CONFIGURE: ["配置", "设置", "修改", "调整", "更改"],
            IntentType.COMPARE: ["比较", "对比", "差异", "区别", "相比"],
            IntentType.FOLLOW_UP: ["还有", "另外", "继续", "然后", "接下来"],
            IntentType.CLARIFICATION: ["确认", "澄清", "明确", "具体", "详细"],
        }

    def recognize_intent(
        self,
        text: str,
        entities: List[Entity],
        previous_intent: Optional[DialogueIntent] = None,
    ) -> DialogueIntent:
        """识别对话意图

        Args:
            text: 用户输入文本
            entities: 提取的实体
            previous_intent: 前一个意图

        Returns:
            识别的意图
        """
        # 基于关键词的简单意图识别
        intent_scores = {}

        for intent_type, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1

            if score > 0:
                intent_scores[intent_type] = score / len(keywords)

        # 选择得分最高的意图
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            intent_type, confidence = best_intent
        else:
            # 默认为查询意图
            intent_type = IntentType.QUERY
            confidence = 0.5

        # 考虑上下文连续性
        if previous_intent and self._is_follow_up_pattern(text):
            intent_type = IntentType.FOLLOW_UP
            confidence = 0.8

        return DialogueIntent(
            intent_type=intent_type,
            confidence=confidence,
            timestamp=time.time(),
            turn_id=0,  # 将在状态管理器中设置
            entities=entities,
        )

    def _is_follow_up_pattern(self, text: str) -> bool:
        """检查是否为后续询问模式

        Args:
            text: 用户输入文本

        Returns:
            是否为后续询问
        """
        follow_up_patterns = ["那", "它", "这个", "还有", "另外", "继续", "然后"]
        return any(pattern in text for pattern in follow_up_patterns)


class DialogueStateManager:
    """对话状态管理器

    负责管理完整的对话状态，包括槽位、意图、实体等
    """

    def __init__(self):
        """初始化对话状态管理器"""
        self.logger = get_logger("DialogueStateManager")

        # 初始化组件
        self.slot_manager = SlotManager()
        self.intent_recognizer = IntentRecognizer()

        # 对话状态存储
        self.dialogue_states: Dict[str, DialogueState] = {}

        self.logger.info("对话状态管理器初始化完成")

    def update_state(
        self, session_id: str, user_input: str, entities: List[Entity]
    ) -> DialogueState:
        """更新对话状态

        Args:
            session_id: 会话ID
            user_input: 用户输入
            entities: 提取的实体

        Returns:
            更新后的对话状态
        """
        try:
            # 获取或创建对话状态
            state = self._get_or_create_state(session_id)

            # 增加对话轮次
            state.turn_count += 1

            # 识别意图
            intent = self.intent_recognizer.recognize_intent(
                user_input, entities, state.last_intent
            )
            intent.turn_id = state.turn_count

            # 更新意图历史
            state.intent_history.append(intent)
            state.last_intent = intent

            # 填充槽位
            updated_slots = self.slot_manager.fill_slots(state, entities)
            state.slots.update(updated_slots)

            # 更新实体信息
            self._update_entities(state, entities)

            # 更新上下文焦点
            self._update_context_focus(state, entities)

            # 清理过期数据
            self._cleanup_expired_data(state)

            # 更新时间戳
            state.last_updated = time.time()

            self.logger.debug(
                f"对话状态更新完成: {session_id}, 轮次: {state.turn_count}"
            )
            return state

        except Exception as e:
            self.logger.error(f"对话状态更新失败: {e}")
            return self._get_or_create_state(session_id)

    def _get_or_create_state(self, session_id: str) -> DialogueState:
        """获取或创建对话状态

        Args:
            session_id: 会话ID

        Returns:
            对话状态
        """
        if session_id not in self.dialogue_states:
            self.dialogue_states[session_id] = DialogueState(session_id=session_id)
            self.logger.info(f"创建新的对话状态: {session_id}")

        return self.dialogue_states[session_id]

    def _update_entities(self, state: DialogueState, entities: List[Entity]):
        """更新实体信息

        Args:
            state: 对话状态
            entities: 新实体列表
        """
        # 添加到实体历史
        state.entity_history.extend(entities)

        # 更新活跃实体
        for entity in entities:
            key = f"{entity.entity_type.value}_{entity.text.lower()}"
            state.active_entities[key] = entity

        # 保持活跃实体数量在合理范围内
        if len(state.active_entities) > 50:
            # 移除最旧的实体
            sorted_entities = sorted(
                state.active_entities.items(),
                key=lambda x: getattr(x[1], "timestamp", 0),
            )
            for key, _ in sorted_entities[:10]:  # 移除最旧的10个
                del state.active_entities[key]

    def _update_context_focus(self, state: DialogueState, entities: List[Entity]):
        """更新上下文焦点

        Args:
            state: 对话状态
            entities: 实体列表
        """
        if entities:
            # 选择置信度最高的实体作为焦点
            focus_entity = max(entities, key=lambda x: x.confidence)
            state.context_focus = (
                f"{focus_entity.entity_type.value}:{focus_entity.text}"
            )

    def _cleanup_expired_data(self, state: DialogueState):
        """清理过期数据

        Args:
            state: 对话状态
        """
        # 清理过期槽位
        state.slots = self.slot_manager.clean_expired_slots(state.slots)

        # 清理过期意图历史（保留最近20个）
        if len(state.intent_history) > 20:
            state.intent_history = state.intent_history[-20:]

        # 清理过期实体历史（保留最近100个）
        if len(state.entity_history) > 100:
            state.entity_history = state.entity_history[-100:]

    def get_state(self, session_id: str) -> Optional[DialogueState]:
        """获取对话状态

        Args:
            session_id: 会话ID

        Returns:
            对话状态
        """
        return self.dialogue_states.get(session_id)

    def get_active_entities(self, session_id: str) -> Dict[str, Entity]:
        """获取活跃实体

        Args:
            session_id: 会话ID

        Returns:
            活跃实体字典
        """
        state = self.get_state(session_id)
        return state.active_entities if state else {}

    def get_filled_slots(self, session_id: str) -> Dict[str, Slot]:
        """获取已填充的槽位

        Args:
            session_id: 会话ID

        Returns:
            已填充的槽位字典
        """
        state = self.get_state(session_id)
        if state:
            return self.slot_manager.get_filled_slots(state.slots)
        return {}

    def clear_state(self, session_id: str):
        """清除对话状态

        Args:
            session_id: 会话ID
        """
        if session_id in self.dialogue_states:
            del self.dialogue_states[session_id]
            self.logger.info(f"已清除对话状态: {session_id}")

    def get_state_summary(self, session_id: str) -> Dict[str, Any]:
        """获取对话状态摘要

        Args:
            session_id: 会话ID

        Returns:
            状态摘要信息
        """
        state = self.get_state(session_id)
        if not state:
            return {"error": "会话不存在"}

        slot_summary = self.slot_manager.get_slot_summary(state.slots)

        return {
            "session_id": session_id,
            "turn_count": state.turn_count,
            "active_entities_count": len(state.active_entities),
            "entity_history_count": len(state.entity_history),
            "intent_history_count": len(state.intent_history),
            "last_intent": (
                state.last_intent.intent_type.value if state.last_intent else None
            ),
            "context_focus": state.context_focus,
            "slot_summary": slot_summary,
            "created_at": state.created_at,
            "last_updated": state.last_updated,
        }
