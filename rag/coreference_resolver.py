"""
指代消解模块

该模块负责在多轮对话中解决指代关系，包括：
1. 使用LLM识别指代词（它、这个、那个等）
2. 将指代词与历史实体进行匹配
3. 在对话上下文中维护实体的指代关系
4. 支持复杂的指代消解场景

数据流：
1. 当前用户输入 -> 指代词识别 -> 候选实体匹配
2. 对话历史 -> 实体提取 -> 上下文分析 -> 指代关系建立
3. 指代消解 -> 实体替换 -> 完整查询生成

学习要点：
1. LLM在指代消解中的应用
2. 对话上下文的实体跟踪
3. 指代关系的建立和维护
4. 多轮对话中的实体状态管理
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from logger import get_logger
from llm import LLMManager
from ner_manager import Entity, EntityType


class ReferenceType(Enum):
    """指代类型枚举"""

    PRONOUN = "PRONOUN"  # 代词指代（它、这个、那个）
    DEFINITE = "DEFINITE"  # 定指代（该服务器、这台机器）
    IMPLICIT = "IMPLICIT"  # 隐式指代（继续、还有）
    NONE = "NONE"  # 无指代


@dataclass
class Reference:
    """指代关系数据结构"""

    text: str  # 指代词文本
    reference_type: ReferenceType  # 指代类型
    start_pos: int  # 开始位置
    end_pos: int  # 结束位置
    confidence: float  # 置信度
    resolved_entity: Optional[Entity] = None  # 解析到的实体
    candidates: List[Entity] = None  # 候选实体列表

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result["reference_type"] = self.reference_type.value
        if self.resolved_entity:
            result["resolved_entity"] = self.resolved_entity.to_dict()
        if self.candidates:
            result["candidates"] = [entity.to_dict() for entity in self.candidates]
        return result


@dataclass
class DialogueContext:
    """对话上下文数据结构"""

    session_id: str  # 会话ID
    turn_count: int  # 对话轮次
    entities_history: List[Entity]  # 历史实体列表
    active_entities: Dict[str, Entity]  # 当前活跃实体
    last_mentioned_entities: List[Entity]  # 最近提及的实体
    last_updated: float  # 最后更新时间

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "entities_history": [entity.to_dict() for entity in self.entities_history],
            "active_entities": {
                k: v.to_dict() for k, v in self.active_entities.items()
            },
            "last_mentioned_entities": [
                entity.to_dict() for entity in self.last_mentioned_entities
            ],
            "last_updated": self.last_updated,
        }


class ReferenceDetector:
    """指代词检测器

    使用LLM和规则方法检测文本中的指代词
    """

    def __init__(self, llm_manager: LLMManager):
        """初始化指代词检测器

        Args:
            llm_manager: LLM管理器实例
        """
        self.logger = get_logger("ReferenceDetector")
        self.llm = llm_manager

        # 常见指代词模式
        self.pronoun_patterns = {
            "它",
            "他",
            "她",
            "这个",
            "那个",
            "这些",
            "那些",
            "此",
            "该",
            "这台",
            "那台",
            "这种",
            "那种",
            "这样",
            "那样",
            "上述",
            "前面",
            "刚才",
            "刚刚",
            "刚才的",
            "上面的",
            "前面的",
            "之前的",
        }

        # LLM指代检测提示模板
        self.reference_detection_prompt = """你是一个专业的指代消解助手。请从以下文本中识别出所有的指代词和指代表达。

指代类型包括：
- PRONOUN: 代词指代（它、这个、那个、这些、那些等）
- DEFINITE: 定指代（该服务器、这台机器、上述问题等）
- IMPLICIT: 隐式指代（继续、还有、另外等）

文本: "{text}"

请以JSON格式返回结果：
{{
    "references": [
        {{
            "text": "指代词文本",
            "type": "指代类型",
            "start": 开始位置,
            "end": 结束位置,
            "confidence": 置信度(0.0-1.0)
        }}
    ]
}}

只返回JSON，不要其他解释。"""

    def detect_references(self, text: str) -> List[Reference]:
        """检测文本中的指代词

        Args:
            text: 输入文本

        Returns:
            检测到的指代词列表
        """
        references = []

        try:
            # 1. 规则方法检测常见指代词
            rule_references = self._detect_by_rules(text)
            references.extend(rule_references)

            # 2. LLM方法检测复杂指代
            llm_references = self._detect_by_llm(text)
            references.extend(llm_references)

            # 3. 去重和合并
            merged_references = self._merge_references(references)

            self.logger.debug(
                f"指代词检测完成，共检测到 {len(merged_references)} 个指代"
            )
            return merged_references

        except Exception as e:
            self.logger.error(f"指代词检测失败: {e}")
            return []

    def _detect_by_rules(self, text: str) -> List[Reference]:
        """使用规则方法检测指代词

        Args:
            text: 输入文本

        Returns:
            检测到的指代词列表
        """
        references = []

        # 检测常见指代词
        for pronoun in self.pronoun_patterns:
            start = 0
            while True:
                pos = text.find(pronoun, start)
                if pos == -1:
                    break

                # 检查是否为完整词汇
                if self._is_complete_word(text, pos, pos + len(pronoun)):
                    reference = Reference(
                        text=pronoun,
                        reference_type=ReferenceType.PRONOUN,
                        start_pos=pos,
                        end_pos=pos + len(pronoun),
                        confidence=0.8,
                    )
                    references.append(reference)

                start = pos + 1

        return references

    def _detect_by_llm(self, text: str) -> List[Reference]:
        """使用LLM检测指代词

        Args:
            text: 输入文本

        Returns:
            检测到的指代词列表
        """
        references = []

        try:
            prompt = self.reference_detection_prompt.format(text=text)
            response = self.llm.generate(prompt)

            # 解析LLM响应
            response_data = json.loads(response.strip())

            if "references" in response_data:
                for ref_data in response_data["references"]:
                    try:
                        ref_type_str = ref_data.get("type", "PRONOUN")
                        ref_type = (
                            ReferenceType(ref_type_str)
                            if ref_type_str in [t.value for t in ReferenceType]
                            else ReferenceType.PRONOUN
                        )

                        reference = Reference(
                            text=ref_data["text"],
                            reference_type=ref_type,
                            start_pos=ref_data.get("start", 0),
                            end_pos=ref_data.get("end", 0),
                            confidence=ref_data.get("confidence", 0.7),
                        )
                        references.append(reference)

                    except Exception as e:
                        self.logger.warning(f"解析单个指代失败: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"LLM指代检测失败: {e}")

        return references

    def _is_complete_word(self, text: str, start: int, end: int) -> bool:
        """检查是否为完整词汇

        Args:
            text: 文本
            start: 开始位置
            end: 结束位置

        Returns:
            是否为完整词汇
        """
        # 检查前后字符是否为分隔符
        if start > 0 and text[start - 1].isalnum():
            return False
        if end < len(text) and text[end].isalnum():
            return False
        return True

    def _merge_references(self, references: List[Reference]) -> List[Reference]:
        """合并重复的指代词

        Args:
            references: 原始指代词列表

        Returns:
            合并后的指代词列表
        """
        if not references:
            return []

        # 按位置排序
        references.sort(key=lambda x: x.start_pos)

        merged = []
        for ref in references:
            # 检查是否与已有指代重叠
            overlapped = False
            for existing in merged:
                if (
                    ref.start_pos < existing.end_pos
                    and ref.end_pos > existing.start_pos
                ):
                    # 有重叠，选择置信度更高的
                    if ref.confidence > existing.confidence:
                        merged.remove(existing)
                        merged.append(ref)
                    overlapped = True
                    break

            if not overlapped:
                merged.append(ref)

        return merged


class CoreferenceResolver:
    """指代消解器

    负责将指代词与历史实体进行匹配和消解
    """

    def __init__(self, llm_manager: LLMManager):
        """初始化指代消解器

        Args:
            llm_manager: LLM管理器实例
        """
        self.logger = get_logger("CoreferenceResolver")
        self.llm = llm_manager
        self.reference_detector = ReferenceDetector(llm_manager)

        # 对话上下文存储
        self.dialogue_contexts: Dict[str, DialogueContext] = {}

        # LLM指代消解提示模板
        self.resolution_prompt = """你是一个专业的指代消解助手。请根据对话历史和当前文本，将指代词与具体的实体进行匹配。

对话历史中的实体：
{entities_context}

当前文本: "{current_text}"

检测到的指代词: {references}

请为每个指代词选择最合适的实体进行匹配，考虑以下因素：
1. 实体类型的匹配度
2. 时间距离（最近提及的优先）
3. 语义相关性
4. 上下文连贯性

请以JSON格式返回结果：
{{
    "resolutions": [
        {{
            "reference_text": "指代词文本",
            "resolved_entity": "匹配的实体文本",
            "entity_type": "实体类型",
            "confidence": 置信度(0.0-1.0),
            "reason": "匹配原因"
        }}
    ]
}}

只返回JSON，不要其他解释。"""

    def resolve_references(
        self, text: str, session_id: str, current_entities: List[Entity] = None
    ) -> Tuple[List[Reference], str]:
        """解析文本中的指代关系

        Args:
            text: 当前文本
            session_id: 会话ID
            current_entities: 当前文本中的实体

        Returns:
            (解析后的指代列表, 消解后的文本)
        """
        try:
            self.logger.info(f"开始指代消解: {text[:50]}...")

            # 1. 检测指代词
            references = self.reference_detector.detect_references(text)
            if not references:
                self.logger.debug("未检测到指代词")
                return [], text

            # 2. 获取对话上下文
            context = self._get_dialogue_context(session_id)

            # 3. 更新上下文（添加当前实体）
            if current_entities:
                self._update_context_entities(context, current_entities)

            # 4. 执行指代消解
            resolved_references = self._resolve_with_llm(text, references, context)

            # 5. 生成消解后的文本
            resolved_text = self._generate_resolved_text(text, resolved_references)

            # 6. 更新对话上下文
            self._update_dialogue_context(session_id, context)

            self.logger.info(f"指代消解完成，解析了 {len(resolved_references)} 个指代")
            return resolved_references, resolved_text

        except Exception as e:
            self.logger.error(f"指代消解失败: {e}")
            return [], text

    def _get_dialogue_context(self, session_id: str) -> DialogueContext:
        """获取对话上下文

        Args:
            session_id: 会话ID

        Returns:
            对话上下文
        """
        if session_id not in self.dialogue_contexts:
            self.dialogue_contexts[session_id] = DialogueContext(
                session_id=session_id,
                turn_count=0,
                entities_history=[],
                active_entities={},
                last_mentioned_entities=[],
                last_updated=time.time(),
            )

        return self.dialogue_contexts[session_id]

    def _update_context_entities(
        self, context: DialogueContext, entities: List[Entity]
    ):
        """更新上下文中的实体信息

        Args:
            context: 对话上下文
            entities: 新实体列表
        """
        # 添加到历史实体
        context.entities_history.extend(entities)

        # 更新活跃实体（按类型分组）
        for entity in entities:
            key = f"{entity.entity_type.value}_{entity.text.lower()}"
            context.active_entities[key] = entity

        # 更新最近提及的实体（保留最近5个）
        context.last_mentioned_entities = (
            entities[-5:] if len(entities) > 5 else entities
        )

        # 增加对话轮次
        context.turn_count += 1
        context.last_updated = time.time()

    def _resolve_with_llm(
        self, text: str, references: List[Reference], context: DialogueContext
    ) -> List[Reference]:
        """使用LLM进行指代消解

        Args:
            text: 当前文本
            references: 指代词列表
            context: 对话上下文

        Returns:
            解析后的指代列表
        """
        try:
            # 构建实体上下文信息
            entities_context = self._build_entities_context(context)

            # 构建指代词信息
            references_info = [
                {
                    "text": ref.text,
                    "type": ref.reference_type.value,
                    "position": f"{ref.start_pos}-{ref.end_pos}",
                }
                for ref in references
            ]

            # 调用LLM进行消解
            prompt = self.resolution_prompt.format(
                entities_context=entities_context,
                current_text=text,
                references=json.dumps(references_info, ensure_ascii=False),
            )

            response = self.llm.generate(prompt)

            # 解析LLM响应
            resolved_references = self._parse_resolution_response(
                response, references, context
            )

            return resolved_references

        except Exception as e:
            self.logger.error(f"LLM指代消解失败: {e}")
            return references  # 返回原始指代列表

    def _build_entities_context(self, context: DialogueContext) -> str:
        """构建实体上下文信息

        Args:
            context: 对话上下文

        Returns:
            格式化的实体上下文字符串
        """
        if not context.entities_history:
            return "暂无历史实体"

        # 按类型分组实体
        entities_by_type = {}
        for entity in context.entities_history[-20:]:  # 只考虑最近20个实体
            entity_type = entity.entity_type.value
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity.text)

        # 构建上下文字符串
        context_lines = []
        for entity_type, entity_texts in entities_by_type.items():
            unique_texts = list(set(entity_texts))  # 去重
            context_lines.append(f"{entity_type}: {', '.join(unique_texts)}")

        return "\n".join(context_lines)

    def _parse_resolution_response(
        self,
        response: str,
        original_references: List[Reference],
        context: DialogueContext,
    ) -> List[Reference]:
        """解析LLM的指代消解响应

        Args:
            response: LLM响应
            original_references: 原始指代列表
            context: 对话上下文

        Returns:
            解析后的指代列表
        """
        resolved_references = []

        try:
            response_data = json.loads(response.strip())

            if "resolutions" in response_data:
                # 创建指代词到原始Reference的映射
                ref_map = {ref.text: ref for ref in original_references}

                for resolution in response_data["resolutions"]:
                    ref_text = resolution.get("reference_text", "")
                    if ref_text in ref_map:
                        original_ref = ref_map[ref_text]

                        # 查找匹配的实体
                        resolved_entity = self._find_entity_by_text(
                            resolution.get("resolved_entity", ""), context
                        )

                        # 创建解析后的指代
                        resolved_ref = Reference(
                            text=original_ref.text,
                            reference_type=original_ref.reference_type,
                            start_pos=original_ref.start_pos,
                            end_pos=original_ref.end_pos,
                            confidence=resolution.get("confidence", 0.7),
                            resolved_entity=resolved_entity,
                        )

                        resolved_references.append(resolved_ref)

        except Exception as e:
            self.logger.error(f"解析指代消解响应失败: {e}")
            # 返回原始指代列表
            resolved_references = original_references

        return resolved_references

    def _find_entity_by_text(
        self, entity_text: str, context: DialogueContext
    ) -> Optional[Entity]:
        """根据文本查找实体

        Args:
            entity_text: 实体文本
            context: 对话上下文

        Returns:
            匹配的实体对象
        """
        if not entity_text:
            return None

        # 首先在活跃实体中查找
        for entity in context.active_entities.values():
            if entity.text.lower() == entity_text.lower():
                return entity

        # 然后在历史实体中查找
        for entity in reversed(context.entities_history):  # 从最近的开始查找
            if entity.text.lower() == entity_text.lower():
                return entity

        return None

    def _generate_resolved_text(
        self, original_text: str, resolved_references: List[Reference]
    ) -> str:
        """生成消解后的文本

        Args:
            original_text: 原始文本
            resolved_references: 解析后的指代列表

        Returns:
            消解后的文本
        """
        if not resolved_references:
            return original_text

        # 按位置倒序排序，从后往前替换避免位置偏移
        sorted_refs = sorted(
            resolved_references, key=lambda x: x.start_pos, reverse=True
        )

        resolved_text = original_text
        for ref in sorted_refs:
            if ref.resolved_entity:
                # 替换指代词为具体实体
                replacement = ref.resolved_entity.text
                resolved_text = (
                    resolved_text[: ref.start_pos]
                    + replacement
                    + resolved_text[ref.end_pos :]
                )

        return resolved_text

    def _update_dialogue_context(self, session_id: str, context: DialogueContext):
        """更新对话上下文

        Args:
            session_id: 会话ID
            context: 对话上下文
        """
        self.dialogue_contexts[session_id] = context

    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """获取对话上下文摘要

        Args:
            session_id: 会话ID

        Returns:
            上下文摘要信息
        """
        if session_id not in self.dialogue_contexts:
            return {"error": "会话不存在"}

        context = self.dialogue_contexts[session_id]

        # 统计实体类型
        entity_types = {}
        for entity in context.entities_history:
            entity_type = entity.entity_type.value
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        return {
            "session_id": session_id,
            "turn_count": context.turn_count,
            "total_entities": len(context.entities_history),
            "active_entities": len(context.active_entities),
            "entity_types": entity_types,
            "last_updated": context.last_updated,
        }

    def clear_context(self, session_id: str):
        """清除对话上下文

        Args:
            session_id: 会话ID
        """
        if session_id in self.dialogue_contexts:
            del self.dialogue_contexts[session_id]
            self.logger.info(f"已清除会话上下文: {session_id}")


class CoreferenceManager:
    """指代消解管理器

    提供完整的指代消解功能，整合实体识别和指代消解
    """

    def __init__(self, llm_manager: LLMManager):
        """初始化指代消解管理器

        Args:
            llm_manager: LLM管理器实例
        """
        self.logger = get_logger("CoreferenceManager")
        self.resolver = CoreferenceResolver(llm_manager)

        self.logger.info("指代消解管理器初始化完成")

    def process_text(
        self, text: str, session_id: str, current_entities: List[Entity] = None
    ) -> Dict[str, Any]:
        """处理文本的指代消解

        Args:
            text: 输入文本
            session_id: 会话ID
            current_entities: 当前文本中的实体

        Returns:
            处理结果，包含原始文本、消解后文本、指代信息等
        """
        try:
            # 执行指代消解
            resolved_references, resolved_text = self.resolver.resolve_references(
                text, session_id, current_entities
            )

            # 构建返回结果
            result = {
                "original_text": text,
                "resolved_text": resolved_text,
                "has_references": len(resolved_references) > 0,
                "references": [ref.to_dict() for ref in resolved_references],
                "session_id": session_id,
                "processing_time": time.time(),
            }

            self.logger.debug(
                f"指代消解处理完成: {text[:30]}... -> {resolved_text[:30]}..."
            )
            return result

        except Exception as e:
            self.logger.error(f"指代消解处理失败: {e}")
            return {
                "original_text": text,
                "resolved_text": text,
                "has_references": False,
                "references": [],
                "session_id": session_id,
                "error": str(e),
            }

    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """获取会话上下文信息

        Args:
            session_id: 会话ID

        Returns:
            会话上下文信息
        """
        return self.resolver.get_context_summary(session_id)

    def clear_session(self, session_id: str):
        """清除会话上下文

        Args:
            session_id: 会话ID
        """
        self.resolver.clear_context(session_id)
