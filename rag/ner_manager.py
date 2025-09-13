"""
命名实体识别(NER)管理模块

该模块负责从用户输入中识别命名实体，支持：
1. 使用LLM进行通用实体识别（人名、地名、组织等）
2. 使用正则表达式识别结构化实体（IP地址、邮箱、电话等）
3. 使用LLM进行领域特定实体识别（服务器名、错误码等）
4. 多轮对话中的实体跟踪和更新

数据流：
1. 用户输入 -> 预处理 -> 多种NER方法并行处理
2. 正则匹配 -> 结构化实体提取
3. LLM调用 -> 通用实体识别 -> 领域实体识别
4. 实体合并 -> 去重 -> 置信度评估 -> 返回结果

学习要点：
1. 多种NER技术的组合使用
2. LLM在实体识别中的应用
3. 实体类型的分类和管理
4. 置信度评估和结果合并
"""

import re
import json
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from logger import get_logger
from llm import LLMManager


class EntityType(Enum):
    """实体类型枚举"""

    # 通用实体类型
    PERSON = "PERSON"  # 人名
    ORGANIZATION = "ORG"  # 组织机构
    LOCATION = "LOC"  # 地点
    DATE = "DATE"  # 日期
    TIME = "TIME"  # 时间
    MONEY = "MONEY"  # 金额

    # 技术相关实体
    IP_ADDRESS = "IP_ADDRESS"  # IP地址
    EMAIL = "EMAIL"  # 邮箱地址
    PHONE = "PHONE"  # 电话号码
    URL = "URL"  # 网址
    HOSTNAME = "HOSTNAME"  # 主机名
    SERVICE = "SERVICE"  # 服务名
    ERROR_CODE = "ERROR_CODE"  # 错误码

    # 业务相关实体
    USER_ID = "USER_ID"  # 用户ID
    ORDER_ID = "ORDER_ID"  # 订单号
    PRODUCT = "PRODUCT"  # 产品名
    METRIC = "METRIC"  # 指标名
    TIME_RANGE = "TIME_RANGE"  # 时间范围

    # 其他
    OTHER = "OTHER"  # 其他类型


@dataclass
class Entity:
    """实体数据结构"""

    text: str  # 实体文本
    entity_type: EntityType  # 实体类型
    start_pos: int  # 开始位置
    end_pos: int  # 结束位置
    confidence: float  # 置信度 (0.0-1.0)
    source: str  # 识别来源 (regex/llm_general/llm_domain)
    metadata: Dict[str, Any] = None  # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result["entity_type"] = self.entity_type.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """从字典创建实体对象"""
        data["entity_type"] = EntityType(data["entity_type"])
        return cls(**data)


class RegexEntityExtractor:
    """正则表达式实体提取器

    用于识别具有固定模式的结构化实体，如IP地址、邮箱等
    """

    def __init__(self):
        """初始化正则表达式模式"""
        self.logger = get_logger("RegexEntityExtractor")

        # 定义正则表达式模式
        self.patterns = {
            EntityType.IP_ADDRESS: [
                r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            ],
            EntityType.EMAIL: [
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            ],
            EntityType.PHONE: [
                r"\b(?:\+86[-\s]?)?(?:1[3-9]\d{9}|(?:0\d{2,3}[-\s]?)?\d{7,8})\b",  # 中国手机号和固话
                r"\b\+?[1-9]\d{1,14}\b",  # 国际电话号码
            ],
            EntityType.URL: [
                r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?",
            ],
            EntityType.HOSTNAME: [
                r"\b[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\b",
            ],
            EntityType.ERROR_CODE: [
                r"\b[A-Z]{2,5}[-_]?\d{3,6}\b",  # 如 HTTP-404, ERR_001
                r"\b\d{3,5}\b",  # 纯数字错误码
            ],
            EntityType.USER_ID: [
                r"\buser[-_]?\d+\b",
                r"\b[a-zA-Z0-9]{8,32}\b",  # 通用ID格式
            ],
            EntityType.ORDER_ID: [
                r"\border[-_]?\d+\b",
                r"\b[A-Z0-9]{10,20}\b",  # 订单号格式
            ],
        }

        # 编译正则表达式
        self.compiled_patterns = {}
        for entity_type, patterns in self.patterns.items():
            self.compiled_patterns[entity_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def extract(self, text: str) -> List[Entity]:
        """从文本中提取实体

        Args:
            text: 输入文本

        Returns:
            提取的实体列表
        """
        entities = []

        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = Entity(
                        text=match.group(),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.95,  # 正则匹配置信度较高
                        source="regex",
                    )
                    entities.append(entity)

        return entities


class LLMEntityExtractor:
    """LLM实体提取器

    使用大语言模型进行实体识别，支持通用实体和领域特定实体
    """

    def __init__(self, llm_manager: LLMManager):
        """初始化LLM实体提取器

        Args:
            llm_manager: LLM管理器实例
        """
        self.logger = get_logger("LLMEntityExtractor")
        self.llm = llm_manager

        # 通用实体识别提示模板
        self.general_ner_prompt = """你是一个专业的命名实体识别助手。请从以下文本中识别出所有的命名实体。

请识别以下类型的实体：
- PERSON: 人名
- ORG: 组织机构名
- LOC: 地点名
- DATE: 日期
- TIME: 时间
- MONEY: 金额

文本: "{text}"

请以JSON格式返回结果，格式如下：
{{
    "entities": [
        {{
            "text": "实体文本",
            "type": "实体类型",
            "start": 开始位置,
            "end": 结束位置,
            "confidence": 置信度(0.0-1.0)
        }}
    ]
}}

只返回JSON，不要其他解释。"""

        # 领域特定实体识别提示模板
        self.domain_ner_prompt = """你是一个IT运维领域的实体识别专家。请从以下文本中识别出技术相关的实体。

请识别以下类型的实体：
- HOSTNAME: 主机名、服务器名 (如: web-server-01, db-server-03)
- SERVICE: 服务名 (如: Nginx, MySQL, Redis)
- METRIC: 性能指标 (如: CPU使用率, 内存占用, 磁盘空间)
- TIME_RANGE: 时间范围 (如: 过去3小时, 最近一周, 昨天)
- PRODUCT: 产品名称
- OTHER: 其他技术相关实体

文本: "{text}"

请以JSON格式返回结果，格式如下：
{{
    "entities": [
        {{
            "text": "实体文本",
            "type": "实体类型", 
            "start": 开始位置,
            "end": 结束位置,
            "confidence": 置信度(0.0-1.0)
        }}
    ]
}}

只返回JSON，不要其他解释。"""

    def extract_general_entities(self, text: str) -> List[Entity]:
        """使用LLM提取通用实体

        Args:
            text: 输入文本

        Returns:
            提取的实体列表
        """
        try:
            prompt = self.general_ner_prompt.format(text=text)
            response = self.llm.generate(prompt)

            # 解析LLM响应
            entities = self._parse_llm_response(response, "llm_general")
            self.logger.debug(f"LLM通用实体识别完成，识别出 {len(entities)} 个实体")
            return entities

        except Exception as e:
            self.logger.error(f"LLM通用实体识别失败: {e}")
            return []

    def extract_domain_entities(self, text: str) -> List[Entity]:
        """使用LLM提取领域特定实体

        Args:
            text: 输入文本

        Returns:
            提取的实体列表
        """
        try:
            prompt = self.domain_ner_prompt.format(text=text)
            response = self.llm.generate(prompt)

            # 解析LLM响应
            entities = self._parse_llm_response(response, "llm_domain")
            self.logger.debug(f"LLM领域实体识别完成，识别出 {len(entities)} 个实体")
            return entities

        except Exception as e:
            self.logger.error(f"LLM领域实体识别失败: {e}")
            return []

    def _parse_llm_response(self, response: str, source: str) -> List[Entity]:
        """解析LLM响应，提取实体信息

        Args:
            response: LLM响应文本
            source: 识别来源标识

        Returns:
            解析出的实体列表
        """
        entities = []

        try:
            # 尝试解析JSON响应
            response_data = json.loads(response.strip())

            if "entities" in response_data:
                for entity_data in response_data["entities"]:
                    try:
                        # 映射实体类型
                        entity_type_str = entity_data.get("type", "OTHER")
                        entity_type = self._map_entity_type(entity_type_str)

                        entity = Entity(
                            text=entity_data["text"],
                            entity_type=entity_type,
                            start_pos=entity_data.get("start", 0),
                            end_pos=entity_data.get("end", 0),
                            confidence=entity_data.get("confidence", 0.8),
                            source=source,
                        )
                        entities.append(entity)

                    except Exception as e:
                        self.logger.warning(f"解析单个实体失败: {e}")
                        continue

        except json.JSONDecodeError as e:
            self.logger.error(f"解析LLM响应JSON失败: {e}")
            # 尝试从响应中提取可能的实体信息
            entities = self._fallback_parse(response, source)

        return entities

    def _map_entity_type(self, type_str: str) -> EntityType:
        """映射字符串到实体类型枚举

        Args:
            type_str: 实体类型字符串

        Returns:
            对应的实体类型枚举
        """
        type_mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME,
            "MONEY": EntityType.MONEY,
            "HOSTNAME": EntityType.HOSTNAME,
            "SERVICE": EntityType.SERVICE,
            "METRIC": EntityType.METRIC,
            "TIME_RANGE": EntityType.TIME_RANGE,
            "PRODUCT": EntityType.PRODUCT,
            "OTHER": EntityType.OTHER,
        }

        return type_mapping.get(type_str.upper(), EntityType.OTHER)

    def _fallback_parse(self, response: str, source: str) -> List[Entity]:
        """当JSON解析失败时的备用解析方法

        Args:
            response: LLM响应文本
            source: 识别来源标识

        Returns:
            解析出的实体列表
        """
        entities = []

        # 简单的文本解析逻辑，寻找可能的实体提及
        lines = response.split("\n")
        for line in lines:
            if ":" in line and any(
                    keyword in line.lower() for keyword in ["实体", "entity", "识别"]
            ):
                # 尝试提取实体信息
                parts = line.split(":")
                if len(parts) >= 2:
                    entity_text = parts[1].strip()
                    if entity_text:
                        entity = Entity(
                            text=entity_text,
                            entity_type=EntityType.OTHER,
                            start_pos=0,
                            end_pos=len(entity_text),
                            confidence=0.5,  # 较低的置信度
                            source=source,
                        )
                        entities.append(entity)

        return entities


class NERManager:
    """命名实体识别管理器

    整合多种NER方法，提供统一的实体识别接口
    """

    def __init__(self, llm_manager: LLMManager):
        """初始化NER管理器

        Args:
            llm_manager: LLM管理器实例
        """
        self.logger = get_logger("NERManager")

        # 初始化各种实体提取器
        self.regex_extractor = RegexEntityExtractor()
        self.llm_extractor = LLMEntityExtractor(llm_manager)

        self.logger.info("NER管理器初始化完成")

    def extract_entities(
            self,
            text: str,
            use_regex: bool = True,
            use_llm_general: bool = True,
            use_llm_domain: bool = True,
    ) -> List[Entity]:
        """从文本中提取所有实体

        Args:
            text: 输入文本
            use_regex: 是否使用正则表达式提取
            use_llm_general: 是否使用LLM进行通用实体识别
            use_llm_domain: 是否使用LLM进行领域实体识别

        Returns:
            去重后的实体列表
        """
        all_entities = []

        try:
            self.logger.info(f"开始实体识别: {text[:50]}...")

            # 1. 正则表达式提取结构化实体
            if use_regex:
                regex_entities = self.regex_extractor.extract(text)
                all_entities.extend(regex_entities)
                self.logger.debug(f"正则提取到 {len(regex_entities)} 个实体")

            # 2. LLM通用实体识别
            if use_llm_general:
                general_entities = self.llm_extractor.extract_general_entities(text)
                all_entities.extend(general_entities)
                self.logger.debug(f"LLM通用识别到 {len(general_entities)} 个实体")

            # 3. LLM领域特定实体识别
            if use_llm_domain:
                domain_entities = self.llm_extractor.extract_domain_entities(text)
                all_entities.extend(domain_entities)
                self.logger.debug(f"LLM领域识别到 {len(domain_entities)} 个实体")

            # 4. 合并和去重
            merged_entities = self._merge_and_deduplicate(all_entities)

            self.logger.info(
                f"实体识别完成，共识别出 {len(merged_entities)} 个唯一实体"
            )
            return merged_entities

        except Exception as e:
            self.logger.error(f"实体识别失败: {e}")
            return []

    def _merge_and_deduplicate(self, entities: List[Entity]) -> List[Entity]:
        """合并和去重实体列表

        Args:
            entities: 原始实体列表

        Returns:
            去重后的实体列表
        """
        if not entities:
            return []

        # 按文本内容分组
        entity_groups = {}
        for entity in entities:
            key = entity.text.lower().strip()
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)

        # 对每组实体进行合并
        merged_entities = []
        for group in entity_groups.values():
            merged_entity = self._merge_entity_group(group)
            if merged_entity:
                merged_entities.append(merged_entity)

        # 按置信度排序
        merged_entities.sort(key=lambda x: x.confidence, reverse=True)

        return merged_entities

    def _merge_entity_group(self, entities: List[Entity]) -> Optional[Entity]:
        """合并同一文本的多个实体

        Args:
            entities: 同一文本的实体列表

        Returns:
            合并后的实体
        """
        if not entities:
            return None

        if len(entities) == 1:
            return entities[0]

        # 选择置信度最高的实体作为基础
        best_entity = max(entities, key=lambda x: x.confidence)

        # 合并来源信息
        sources = list(set(entity.source for entity in entities))
        merged_source = "+".join(sources)

        # 选择最具体的实体类型（非OTHER类型优先）
        entity_types = [
            entity.entity_type
            for entity in entities
            if entity.entity_type != EntityType.OTHER
        ]
        if entity_types:
            merged_type = entity_types[0]  # 选择第一个非OTHER类型
        else:
            merged_type = EntityType.OTHER

        # 创建合并后的实体
        merged_entity = Entity(
            text=best_entity.text,
            entity_type=merged_type,
            start_pos=best_entity.start_pos,
            end_pos=best_entity.end_pos,
            confidence=min(1.0, best_entity.confidence + 0.1),  # 略微提升置信度
            source=merged_source,
            metadata={"merged_from": len(entities)},
        )

        return merged_entity

    def get_entities_by_type(
            self, entities: List[Entity], entity_type: EntityType
    ) -> List[Entity]:
        """根据类型筛选实体

        Args:
            entities: 实体列表
            entity_type: 目标实体类型

        Returns:
            指定类型的实体列表
        """
        return [entity for entity in entities if entity.entity_type == entity_type]

    def get_high_confidence_entities(
            self, entities: List[Entity], threshold: float = 0.8
    ) -> List[Entity]:
        """获取高置信度实体

        Args:
            entities: 实体列表
            threshold: 置信度阈值

        Returns:
            高置信度实体列表
        """
        return [entity for entity in entities if entity.confidence >= threshold]
