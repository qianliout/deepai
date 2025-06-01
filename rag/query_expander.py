"""
查询扩展模块

使用手动维护的同义词词典进行查询扩展，提高检索召回率。
这是一个简化的实现，用于学习RAG中查询扩展的核心概念。

数据流：
原始查询 -> 分词 -> 同义词查找 -> 查询扩展 -> 扩展后查询

学习要点：
1. 查询扩展的基本原理：通过添加同义词提高召回率
2. 同义词词典的构建和维护
3. 扩展策略的设计和优化
"""

from typing import List, Dict
from dataclasses import dataclass
import time
import re

from logger import get_logger


@dataclass
class QueryExpansionResult:
    """查询扩展结果数据类"""
    original_query: str              # 原始查询
    expanded_query: str              # 扩展后查询
    expansion_terms: List[str]       # 扩展词汇
    synonym_pairs: Dict[str, List[str]]  # 同义词对应关系
    processing_time: float           # 处理时间
    method: str                      # 扩展方法


class SimpleQueryExpander:
    """简单查询扩展器
    
    使用手动维护的同义词词典进行查询扩展，用于学习RAG中查询扩展的核心概念。
    
    学习要点：
    1. 查询扩展的基本原理：通过添加同义词提高召回率
    2. 同义词词典的构建和维护
    3. 扩展策略的设计和优化
    """
    
    def __init__(self, enable_expansion: bool = True):
        """初始化查询扩展器
        
        Args:
            enable_expansion: 是否启用查询扩展
        """
        self.logger = get_logger("SimpleQueryExpander")
        self.enable_expansion = enable_expansion
        
        # 手动维护的同义词词典 - 这是学习重点
        self.synonym_dict = self._build_synonym_dict()
        
        self.logger.info(f"简单查询扩展器初始化完成，词典大小: {len(self.synonym_dict)}")
    
    def _build_synonym_dict(self) -> Dict[str, List[str]]:
        """构建同义词词典
        
        这是一个手动维护的同义词词典，在实际项目中可以：
        1. 从外部文件加载
        2. 通过机器学习方法自动构建
        3. 结合领域知识人工维护
        
        Returns:
            同义词词典
        """
        return {
            # AI相关
            "人工智能": ["AI", "机器智能", "智能系统"],
            "机器学习": ["ML", "机器学习算法", "自动学习"],
            "深度学习": ["DL", "神经网络", "深度神经网络"],
            "自然语言处理": ["NLP", "文本处理", "语言理解"],
            
            # RAG相关
            "检索": ["搜索", "查找", "检索系统"],
            "生成": ["产生", "创建", "生成模型"],
            "向量": ["嵌入", "特征向量", "向量表示"],
            "相似度": ["相关性", "匹配度", "相似性"],
            
            # 技术相关
            "算法": ["方法", "技术", "策略"],
            "模型": ["系统", "框架", "架构"],
            "数据": ["信息", "资料", "内容"],
            "文档": ["文件", "资料", "材料"],
            
            # 常用词汇
            "问题": ["疑问", "难题", "课题"],
            "方法": ["方式", "途径", "手段"],
            "效果": ["结果", "成效", "表现"],
            "优化": ["改进", "提升", "增强"],
        }
    
    def expand_query(
        self, 
        query: str, 
        max_synonyms_per_word: int = 2,
        max_expansion_ratio: float = 2.0
    ) -> QueryExpansionResult:
        """扩展查询
        
        Args:
            query: 原始查询
            max_synonyms_per_word: 每个词的最大同义词数量
            max_expansion_ratio: 最大扩展比例
            
        Returns:
            查询扩展结果
        """
        start_time = time.time()
        
        if not self.enable_expansion:
            return QueryExpansionResult(
                original_query=query,
                expanded_query=query,
                expansion_terms=[],
                synonym_pairs={},
                processing_time=time.time() - start_time,
                method="disabled"
            )
        
        try:
            # 简单分词（按空格和标点分割）
            words = self._simple_tokenize(query)
            
            # 查找同义词
            synonym_pairs = {}
            expansion_terms = []
            
            for word in words:
                if word in self.synonym_dict:
                    synonyms = self.synonym_dict[word][:max_synonyms_per_word]
                    if synonyms:
                        synonym_pairs[word] = synonyms
                        expansion_terms.extend(synonyms)
            
            # 构建扩展查询
            expanded_query = self._build_expanded_query(
                query, 
                synonym_pairs, 
                max_expansion_ratio
            )
            
            processing_time = time.time() - start_time
            
            result = QueryExpansionResult(
                original_query=query,
                expanded_query=expanded_query,
                expansion_terms=expansion_terms,
                synonym_pairs=synonym_pairs,
                processing_time=processing_time,
                method="manual_synonyms"
            )
            
            self.logger.debug(
                f"查询扩展完成 | 原始: '{query}' | "
                f"扩展: '{expanded_query}' | 扩展词数: {len(expansion_terms)}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"查询扩展失败: {e}")
            return QueryExpansionResult(
                original_query=query,
                expanded_query=query,
                expansion_terms=[],
                synonym_pairs={},
                processing_time=time.time() - start_time,
                method="error"
            )
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """简单分词
        
        使用正则表达式进行简单的中文分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果
        """
        # 匹配中文词汇、英文单词、数字
        pattern = r'[\u4e00-\u9fff]+|[a-zA-Z]+|[0-9]+'
        tokens = re.findall(pattern, text)
        
        # 过滤长度小于2的词汇
        return [token for token in tokens if len(token) >= 2]
    
    def _build_expanded_query(
        self, 
        original_query: str, 
        synonym_pairs: Dict[str, List[str]], 
        max_expansion_ratio: float
    ) -> str:
        """构建扩展查询
        
        Args:
            original_query: 原始查询
            synonym_pairs: 同义词对应关系
            max_expansion_ratio: 最大扩展比例
            
        Returns:
            扩展后的查询
        """
        if not synonym_pairs:
            return original_query
        
        # 计算最大扩展长度
        max_length = int(len(original_query) * max_expansion_ratio)
        
        # 构建扩展查询
        expanded_parts = [original_query]
        
        # 添加同义词
        for word, synonyms in synonym_pairs.items():
            for synonym in synonyms:
                expanded_parts.append(synonym)
                
                # 检查长度限制
                current_query = " ".join(expanded_parts)
                if len(current_query) > max_length:
                    break
            
            # 检查长度限制
            current_query = " ".join(expanded_parts)
            if len(current_query) > max_length:
                break
        
        return " ".join(expanded_parts)