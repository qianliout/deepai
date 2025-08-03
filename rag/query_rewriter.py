"""
查询重写模块

该模块负责基于对话状态和实体信息重写用户查询，使其更完整和准确，包括：
1. 使用LLM进行智能查询重写
2. 基于对话状态补全省略的上下文
3. 整合实体信息和指代消解结果
4. 生成适合检索的完整查询

数据流：
1. 用户查询 + 对话状态 -> 上下文分析 -> 查询重写
2. 实体信息 + 指代消解 -> 实体替换 -> 查询补全
3. 历史意图 + 当前意图 -> 意图连贯性 -> 查询优化

学习要点：
1. LLM在查询重写中的应用
2. 多轮对话的上下文理解
3. 实体信息的有效整合
4. 查询质量的评估和优化
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from logger import get_logger
from llm import LLMManager
from ner_manager import Entity, EntityType
from dialogue_state_manager import DialogueState, Slot
from coreference_resolver import Reference


@dataclass
class RewriteResult:
    """查询重写结果数据结构"""
    original_query: str          # 原始查询
    rewritten_query: str         # 重写后查询
    confidence: float            # 重写置信度
    rewrite_type: str           # 重写类型
    used_entities: List[Entity] = None  # 使用的实体
    used_slots: List[Slot] = None       # 使用的槽位
    reasoning: str = ""         # 重写推理过程
    metadata: Dict[str, Any] = None     # 元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        if self.used_entities:
            result['used_entities'] = [entity.to_dict() for entity in self.used_entities]
        if self.used_slots:
            result['used_slots'] = [slot.to_dict() for slot in self.used_slots]
        return result


class ContextAnalyzer:
    """上下文分析器
    
    分析对话状态，提取重写所需的上下文信息
    """
    
    def __init__(self):
        """初始化上下文分析器"""
        self.logger = get_logger("ContextAnalyzer")
    
    def analyze_context(self, query: str, dialogue_state: DialogueState) -> Dict[str, Any]:
        """分析查询上下文
        
        Args:
            query: 用户查询
            dialogue_state: 对话状态
            
        Returns:
            上下文分析结果
        """
        try:
            # 分析查询完整性
            completeness = self._analyze_query_completeness(query)
            
            # 提取相关实体
            relevant_entities = self._extract_relevant_entities(query, dialogue_state)
            
            # 提取相关槽位
            relevant_slots = self._extract_relevant_slots(query, dialogue_state)
            
            # 分析意图连续性
            intent_continuity = self._analyze_intent_continuity(query, dialogue_state)
            
            # 识别缺失信息
            missing_info = self._identify_missing_info(query, dialogue_state)
            
            context_analysis = {
                "query_completeness": completeness,
                "relevant_entities": relevant_entities,
                "relevant_slots": relevant_slots,
                "intent_continuity": intent_continuity,
                "missing_info": missing_info,
                "needs_rewrite": completeness < 0.7 or len(missing_info) > 0
            }
            
            self.logger.debug(f"上下文分析完成: 完整性={completeness:.2f}, 需要重写={context_analysis['needs_rewrite']}")
            return context_analysis
            
        except Exception as e:
            self.logger.error(f"上下文分析失败: {e}")
            return {"needs_rewrite": False, "error": str(e)}
    
    def _analyze_query_completeness(self, query: str) -> float:
        """分析查询完整性
        
        Args:
            query: 用户查询
            
        Returns:
            完整性评分 (0.0-1.0)
        """
        # 简单的完整性评估
        incomplete_indicators = [
            "它", "这个", "那个", "这些", "那些", "该", "此",
            "继续", "还有", "另外", "然后", "接下来"
        ]
        
        # 检查是否包含不完整指示词
        has_incomplete = any(indicator in query for indicator in incomplete_indicators)
        
        # 检查查询长度
        length_score = min(len(query) / 20, 1.0)  # 20字符以上认为较完整
        
        # 检查是否包含实体
        has_entities = any(char.isdigit() or char.isupper() for char in query)
        entity_score = 0.3 if has_entities else 0.0
        
        # 综合评分
        if has_incomplete:
            completeness = 0.3  # 有指代词，完整性较低
        else:
            completeness = 0.5 + length_score * 0.3 + entity_score
        
        return min(completeness, 1.0)
    
    def _extract_relevant_entities(self, query: str, dialogue_state: DialogueState) -> List[Entity]:
        """提取相关实体
        
        Args:
            query: 用户查询
            dialogue_state: 对话状态
            
        Returns:
            相关实体列表
        """
        relevant_entities = []
        
        # 从活跃实体中查找相关的
        for entity in dialogue_state.active_entities.values():
            # 检查实体是否在查询中被提及
            if entity.text.lower() in query.lower():
                relevant_entities.append(entity)
            # 检查实体类型是否与查询相关
            elif self._is_entity_type_relevant(entity.entity_type, query):
                relevant_entities.append(entity)
        
        # 按置信度排序
        relevant_entities.sort(key=lambda x: x.confidence, reverse=True)
        
        return relevant_entities[:5]  # 最多返回5个相关实体
    
    def _extract_relevant_slots(self, query: str, dialogue_state: DialogueState) -> List[Slot]:
        """提取相关槽位
        
        Args:
            query: 用户查询
            dialogue_state: 对话状态
            
        Returns:
            相关槽位列表
        """
        relevant_slots = []
        
        for slot in dialogue_state.slots.values():
            if slot.is_filled():
                # 检查槽位值是否在查询中
                if slot.value and slot.value.lower() in query.lower():
                    relevant_slots.append(slot)
                # 检查槽位类型是否相关
                elif self._is_slot_type_relevant(slot.entity_type, query):
                    relevant_slots.append(slot)
        
        return relevant_slots
    
    def _analyze_intent_continuity(self, query: str, dialogue_state: DialogueState) -> Dict[str, Any]:
        """分析意图连续性
        
        Args:
            query: 用户查询
            dialogue_state: 对话状态
            
        Returns:
            意图连续性分析结果
        """
        if not dialogue_state.intent_history:
            return {"is_continuous": False, "last_intent": None}
        
        last_intent = dialogue_state.intent_history[-1]
        
        # 检查是否为后续询问
        follow_up_patterns = ["那", "它", "这个", "还有", "另外", "继续", "然后"]
        is_follow_up = any(pattern in query for pattern in follow_up_patterns)
        
        return {
            "is_continuous": is_follow_up,
            "last_intent": last_intent.intent_type.value,
            "last_intent_confidence": last_intent.confidence
        }
    
    def _identify_missing_info(self, query: str, dialogue_state: DialogueState) -> List[str]:
        """识别缺失信息
        
        Args:
            query: 用户查询
            dialogue_state: 对话状态
            
        Returns:
            缺失信息列表
        """
        missing_info = []
        
        # 检查常见的缺失信息类型
        if "它" in query or "这个" in query or "那个" in query:
            missing_info.append("具体实体指代不明")
        
        if any(word in query for word in ["状态", "情况", "如何"]) and not any(slot.name == "hostname" for slot in dialogue_state.slots.values() if slot.is_filled()):
            missing_info.append("缺少目标主机信息")
        
        if any(word in query for word in ["CPU", "内存", "磁盘"]) and not any(slot.name == "metric" for slot in dialogue_state.slots.values() if slot.is_filled()):
            missing_info.append("缺少具体指标信息")
        
        return missing_info
    
    def _is_entity_type_relevant(self, entity_type: EntityType, query: str) -> bool:
        """检查实体类型是否与查询相关
        
        Args:
            entity_type: 实体类型
            query: 查询文本
            
        Returns:
            是否相关
        """
        relevance_map = {
            EntityType.HOSTNAME: ["服务器", "主机", "机器", "节点"],
            EntityType.IP_ADDRESS: ["IP", "地址", "网络"],
            EntityType.SERVICE: ["服务", "应用", "进程"],
            EntityType.METRIC: ["CPU", "内存", "磁盘", "性能", "指标"],
            EntityType.TIME_RANGE: ["时间", "小时", "天", "周", "月"]
        }
        
        keywords = relevance_map.get(entity_type, [])
        return any(keyword in query for keyword in keywords)
    
    def _is_slot_type_relevant(self, entity_type: EntityType, query: str) -> bool:
        """检查槽位类型是否与查询相关"""
        return self._is_entity_type_relevant(entity_type, query)


class QueryRewriter:
    """查询重写器
    
    使用LLM和规则方法重写用户查询
    """
    
    def __init__(self, llm_manager: LLMManager):
        """初始化查询重写器
        
        Args:
            llm_manager: LLM管理器实例
        """
        self.logger = get_logger("QueryRewriter")
        self.llm = llm_manager
        self.context_analyzer = ContextAnalyzer()
        
        # LLM查询重写提示模板
        self.rewrite_prompt = """你是一个专业的查询重写助手。请根据对话历史和上下文信息，将用户的省略式查询重写为完整、清晰、适合信息检索的查询。

对话上下文信息：
{context_info}

当前用户查询: "{current_query}"

重写要求：
1. 补全省略的实体和上下文信息
2. 解决指代关系（将"它"、"这个"等替换为具体实体）
3. 保持查询的原始意图
4. 生成适合信息检索的完整查询
5. 如果查询已经完整，可以保持不变

请以JSON格式返回结果：
{{
    "rewritten_query": "重写后的完整查询",
    "confidence": 置信度(0.0-1.0),
    "rewrite_type": "重写类型(entity_completion/reference_resolution/context_expansion/no_change)",
    "reasoning": "重写推理过程",
    "used_context": ["使用的上下文信息"]
}}

只返回JSON，不要其他解释。"""

        self.logger.info("查询重写器初始化完成")

    def rewrite_query(self, query: str, dialogue_state: DialogueState,
                     resolved_references: List[Reference] = None) -> RewriteResult:
        """重写查询

        Args:
            query: 原始查询
            dialogue_state: 对话状态
            resolved_references: 已解析的指代关系

        Returns:
            查询重写结果
        """
        try:
            # 分析上下文
            context_analysis = self.context_analyzer.analyze_context(query, dialogue_state)

            # 如果不需要重写，直接返回
            if not context_analysis.get("needs_rewrite", False):
                return RewriteResult(
                    original_query=query,
                    rewritten_query=query,
                    confidence=1.0,
                    rewrite_type="no_change",
                    reasoning="查询已经完整，无需重写"
                )

            # 首先应用指代消解结果
            query_with_resolved_refs = self._apply_reference_resolution(query, resolved_references)

            # 使用LLM进行查询重写
            llm_result = self._rewrite_with_llm(query_with_resolved_refs, context_analysis, dialogue_state)

            # 如果LLM重写失败，使用规则方法
            if not llm_result:
                rule_result = self._rewrite_with_rules(query_with_resolved_refs, context_analysis, dialogue_state)
                return rule_result

            return llm_result

        except Exception as e:
            self.logger.error(f"查询重写失败: {e}")
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                confidence=0.5,
                rewrite_type="error",
                reasoning=f"重写过程出错: {str(e)}"
            )

    def _apply_reference_resolution(self, query: str, resolved_references: List[Reference]) -> str:
        """应用指代消解结果

        Args:
            query: 原始查询
            resolved_references: 已解析的指代关系

        Returns:
            应用指代消解后的查询
        """
        if not resolved_references:
            return query

        # 按位置倒序排序，从后往前替换避免位置偏移
        sorted_refs = sorted(resolved_references, key=lambda x: x.start_pos, reverse=True)

        resolved_query = query
        for ref in sorted_refs:
            if ref.resolved_entity:
                # 替换指代词为具体实体
                replacement = ref.resolved_entity.text
                resolved_query = (
                    resolved_query[:ref.start_pos] +
                    replacement +
                    resolved_query[ref.end_pos:]
                )

        return resolved_query

    def _rewrite_with_llm(self, query: str, context_analysis: Dict[str, Any],
                         dialogue_state: DialogueState) -> Optional[RewriteResult]:
        """使用LLM重写查询

        Args:
            query: 查询文本
            context_analysis: 上下文分析结果
            dialogue_state: 对话状态

        Returns:
            LLM重写结果
        """
        try:
            # 构建上下文信息
            context_info = self._build_context_info(context_analysis, dialogue_state)

            # 构建提示
            prompt = self.rewrite_prompt.format(
                context_info=context_info,
                current_query=query
            )

            # 调用LLM
            response = self.llm.generate_response(prompt)

            # 解析LLM响应
            try:
                result_data = json.loads(response)

                return RewriteResult(
                    original_query=query,
                    rewritten_query=result_data.get("rewritten_query", query),
                    confidence=result_data.get("confidence", 0.8),
                    rewrite_type=result_data.get("rewrite_type", "llm_rewrite"),
                    reasoning=result_data.get("reasoning", "LLM重写"),
                    used_entities=context_analysis.get("relevant_entities", []),
                    used_slots=context_analysis.get("relevant_slots", []),
                    metadata={"llm_response": result_data}
                )

            except json.JSONDecodeError:
                self.logger.warning("LLM返回的JSON格式无效")
                return None

        except Exception as e:
            self.logger.error(f"LLM查询重写失败: {e}")
            return None

    def _rewrite_with_rules(self, query: str, context_analysis: Dict[str, Any],
                           dialogue_state: DialogueState) -> RewriteResult:
        """使用规则方法重写查询

        Args:
            query: 查询文本
            context_analysis: 上下文分析结果
            dialogue_state: 对话状态

        Returns:
            规则重写结果
        """
        rewritten_query = query
        used_entities = []
        used_slots = []
        rewrite_operations = []

        # 规则1: 补全主机信息
        if "状态" in query or "情况" in query:
            hostname_entities = [e for e in context_analysis.get("relevant_entities", [])
                               if e.entity_type == EntityType.HOSTNAME]
            if hostname_entities and "主机" not in query and hostname_entities[0].text not in query:
                rewritten_query = f"{hostname_entities[0].text}的{query}"
                used_entities.append(hostname_entities[0])
                rewrite_operations.append("补全主机信息")

        # 规则2: 补全服务信息
        if "服务" in query:
            service_entities = [e for e in context_analysis.get("relevant_entities", [])
                              if e.entity_type == EntityType.SERVICE]
            if service_entities and service_entities[0].text not in query:
                rewritten_query = rewritten_query.replace("服务", f"{service_entities[0].text}服务")
                used_entities.append(service_entities[0])
                rewrite_operations.append("补全服务信息")

        # 规则3: 补全指标信息
        if any(word in query for word in ["CPU", "内存", "磁盘"]):
            metric_slots = [s for s in context_analysis.get("relevant_slots", [])
                           if s.entity_type == EntityType.METRIC]
            if metric_slots and metric_slots[0].value not in query:
                rewritten_query = f"{rewritten_query} {metric_slots[0].value}"
                used_slots.append(metric_slots[0])
                rewrite_operations.append("补全指标信息")

        # 规则4: 补全时间范围
        if "最近" in query or "今天" in query:
            time_slots = [s for s in context_analysis.get("relevant_slots", [])
                         if s.entity_type == EntityType.TIME_RANGE]
            if time_slots and time_slots[0].value not in query:
                rewritten_query = rewritten_query.replace("最近", f"最近{time_slots[0].value}")
                used_slots.append(time_slots[0])
                rewrite_operations.append("补全时间范围")

        # 计算置信度
        confidence = 0.7 if rewrite_operations else 0.9
        rewrite_type = "rule_based" if rewrite_operations else "no_change"

        return RewriteResult(
            original_query=query,
            rewritten_query=rewritten_query,
            confidence=confidence,
            rewrite_type=rewrite_type,
            used_entities=used_entities,
            used_slots=used_slots,
            reasoning=f"规则重写: {', '.join(rewrite_operations)}" if rewrite_operations else "无需重写",
            metadata={"operations": rewrite_operations}
        )

    def _build_context_info(self, context_analysis: Dict[str, Any],
                           dialogue_state: DialogueState) -> str:
        """构建上下文信息字符串

        Args:
            context_analysis: 上下文分析结果
            dialogue_state: 对话状态

        Returns:
            格式化的上下文信息
        """
        context_parts = []

        # 添加相关实体信息
        relevant_entities = context_analysis.get("relevant_entities", [])
        if relevant_entities:
            entity_info = "相关实体: " + ", ".join([f"{e.entity_type.value}:{e.text}" for e in relevant_entities[:3]])
            context_parts.append(entity_info)

        # 添加相关槽位信息
        relevant_slots = context_analysis.get("relevant_slots", [])
        if relevant_slots:
            slot_info = "已知信息: " + ", ".join([f"{s.name}={s.value}" for s in relevant_slots[:3]])
            context_parts.append(slot_info)

        # 添加最近的意图
        if dialogue_state.last_intent:
            intent_info = f"上一个意图: {dialogue_state.last_intent.intent_type.value}"
            context_parts.append(intent_info)

        # 添加上下文焦点
        if dialogue_state.context_focus:
            focus_info = f"当前焦点: {dialogue_state.context_focus}"
            context_parts.append(focus_info)

        return "\n".join(context_parts) if context_parts else "无特殊上下文信息"


class QueryRewriteManager:
    """查询重写管理器

    提供完整的查询重写功能，整合所有组件
    """

    def __init__(self, llm_manager: LLMManager):
        """初始化查询重写管理器

        Args:
            llm_manager: LLM管理器实例
        """
        self.logger = get_logger("QueryRewriteManager")
        self.rewriter = QueryRewriter(llm_manager)

        self.logger.info("查询重写管理器初始化完成")

    def process_query(self, query: str, dialogue_state: DialogueState,
                     resolved_references: List[Reference] = None) -> Dict[str, Any]:
        """处理查询重写

        Args:
            query: 原始查询
            dialogue_state: 对话状态
            resolved_references: 已解析的指代关系

        Returns:
            处理结果
        """
        try:
            # 执行查询重写
            rewrite_result = self.rewriter.rewrite_query(query, dialogue_state, resolved_references)

            # 构建返回结果
            result = {
                "original_query": query,
                "rewritten_query": rewrite_result.rewritten_query,
                "rewrite_needed": rewrite_result.original_query != rewrite_result.rewritten_query,
                "confidence": rewrite_result.confidence,
                "rewrite_type": rewrite_result.rewrite_type,
                "reasoning": rewrite_result.reasoning,
                "used_entities_count": len(rewrite_result.used_entities) if rewrite_result.used_entities else 0,
                "used_slots_count": len(rewrite_result.used_slots) if rewrite_result.used_slots else 0,
                "processing_time": time.time(),
                "rewrite_details": rewrite_result.to_dict()
            }

            self.logger.debug(f"查询重写完成: {query} -> {rewrite_result.rewritten_query}")
            return result

        except Exception as e:
            self.logger.error(f"查询重写处理失败: {e}")
            return {
                "original_query": query,
                "rewritten_query": query,
                "rewrite_needed": False,
                "confidence": 0.5,
                "rewrite_type": "error",
                "reasoning": f"处理失败: {str(e)}",
                "error": str(e)
            }
