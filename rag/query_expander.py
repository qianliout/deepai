"""
查询扩展模块

使用同义词扩展查询，提高检索召回率。
使用Synonyms库进行中文同义词扩展。

数据流：
原始查询 -> 分词 -> 同义词查找 -> 查询扩展 -> 扩展后查询
"""

from typing import List, Set, Dict, Any
from dataclasses import dataclass
import time

# 同义词库导入
try:
    import synonyms
    SYNONYMS_AVAILABLE = True
except ImportError:
    SYNONYMS_AVAILABLE = False

from config import config
from logger import get_logger
from chinese_tokenizer import ChineseTokenizer, TokenizerType


@dataclass
class QueryExpansionResult:
    """查询扩展结果数据类"""
    original_query: str              # 原始查询
    expanded_query: str              # 扩展后查询
    expansion_terms: List[str]       # 扩展词汇
    synonym_pairs: Dict[str, List[str]]  # 同义词对应关系
    processing_time: float           # 处理时间
    method: str                      # 扩展方法


class QueryExpander:
    """查询扩展器
    
    使用同义词扩展查询，提高检索效果
    """
    
    def __init__(self, enable_synonyms: bool = True):
        """初始化查询扩展器
        
        Args:
            enable_synonyms: 是否启用同义词扩展
        """
        self.logger = get_logger("QueryExpander")
        self.enable_synonyms = enable_synonyms
        
        # 初始化分词器
        self.tokenizer = ChineseTokenizer(TokenizerType.JIEBA)
        
        # 初始化同义词库
        if enable_synonyms and SYNONYMS_AVAILABLE:
            self._init_synonyms()
        elif enable_synonyms and not SYNONYMS_AVAILABLE:
            self.logger.warning("synonyms库未安装，同义词扩展功能不可用")
            self.enable_synonyms = False
        
        self.logger.info(f"查询扩展器初始化完成，同义词扩展: {self.enable_synonyms}")
    
    def _init_synonyms(self):
        """初始化同义词库"""
        try:
            # 测试synonyms库是否正常工作
            test_synonyms = synonyms.nearby("测试")
            self.logger.info(f"同义词库初始化成功，测试词汇: {test_synonyms[:3] if test_synonyms else '无'}")
        except Exception as e:
            self.logger.error(f"同义词库初始化失败: {e}")
            self.enable_synonyms = False
    
    def expand_query(
        self, 
        query: str, 
        max_synonyms_per_word: int = 2,
        similarity_threshold: float = 0.7,
        max_expansion_ratio: float = 2.0
    ) -> QueryExpansionResult:
        """扩展查询
        
        Args:
            query: 原始查询
            max_synonyms_per_word: 每个词的最大同义词数量
            similarity_threshold: 同义词相似度阈值
            max_expansion_ratio: 最大扩展比例
            
        Returns:
            查询扩展结果
        """
        start_time = time.time()
        
        try:
            # 分词
            tokenize_result = self.tokenizer.tokenize(query, remove_stop_words=False)
            tokens = tokenize_result.tokens
            
            if not self.enable_synonyms:
                # 不启用同义词扩展，直接返回原查询
                return QueryExpansionResult(
                    original_query=query,
                    expanded_query=query,
                    expansion_terms=[],
                    synonym_pairs={},
                    processing_time=time.time() - start_time,
                    method="none"
                )
            
            # 获取同义词
            synonym_pairs = {}
            expansion_terms = []
            
            for token in tokens:
                if len(token) > 1 and self._is_meaningful_word(token):
                    synonyms_list = self._get_synonyms(
                        token, 
                        max_synonyms_per_word, 
                        similarity_threshold
                    )
                    
                    if synonyms_list:
                        synonym_pairs[token] = synonyms_list
                        expansion_terms.extend(synonyms_list)
            
            # 构建扩展查询
            expanded_query = self._build_expanded_query(
                query, 
                tokens, 
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
                method="synonyms"
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
    
    def _get_synonyms(
        self, 
        word: str, 
        max_count: int, 
        threshold: float
    ) -> List[str]:
        """获取词汇的同义词
        
        Args:
            word: 目标词汇
            max_count: 最大同义词数量
            threshold: 相似度阈值
            
        Returns:
            同义词列表
        """
        if not SYNONYMS_AVAILABLE:
            return []
        
        try:
            # 获取同义词
            nearby_words = synonyms.nearby(word)
            
            if not nearby_words:
                return []
            
            # 过滤同义词
            filtered_synonyms = []
            for synonym, score in nearby_words:
                if (score >= threshold and 
                    synonym != word and 
                    len(synonym) > 1 and
                    self._is_meaningful_word(synonym)):
                    
                    filtered_synonyms.append(synonym)
                    
                    if len(filtered_synonyms) >= max_count:
                        break
            
            return filtered_synonyms
            
        except Exception as e:
            self.logger.debug(f"获取同义词失败 '{word}': {e}")
            return []
    
    def _is_meaningful_word(self, word: str) -> bool:
        """判断是否为有意义的词汇"""
        # 过滤掉单字符、纯数字、纯标点等
        if len(word) <= 1:
            return False
        
        if word.isdigit():
            return False
        
        if not any(c.isalnum() for c in word):
            return False
        
        # 过滤常见的无意义词汇
        meaningless_words = {
            '这个', '那个', '什么', '怎么', '为什么', '哪里', '哪个',
            '多少', '几个', '一些', '很多', '非常', '特别', '比较',
            '可能', '应该', '或者', '但是', '因为', '所以', '如果'
        }
        
        return word not in meaningless_words
    
    def _build_expanded_query(
        self, 
        original_query: str, 
        tokens: List[str], 
        synonym_pairs: Dict[str, List[str]], 
        max_expansion_ratio: float
    ) -> str:
        """构建扩展查询
        
        Args:
            original_query: 原始查询
            tokens: 分词结果
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
    
    def batch_expand(
        self, 
        queries: List[str], 
        **kwargs
    ) -> List[QueryExpansionResult]:
        """批量查询扩展
        
        Args:
            queries: 查询列表
            **kwargs: 扩展参数
            
        Returns:
            扩展结果列表
        """
        results = []
        for query in queries:
            result = self.expand_query(query, **kwargs)
            results.append(result)
        
        self.logger.info(f"批量查询扩展完成，处理 {len(queries)} 个查询")
        return results
    
    def get_expansion_stats(self, results: List[QueryExpansionResult]) -> Dict[str, Any]:
        """获取扩展统计信息
        
        Args:
            results: 扩展结果列表
            
        Returns:
            统计信息
        """
        if not results:
            return {}
        
        total_queries = len(results)
        expanded_queries = sum(1 for r in results if r.expansion_terms)
        total_expansion_terms = sum(len(r.expansion_terms) for r in results)
        avg_processing_time = sum(r.processing_time for r in results) / total_queries
        
        return {
            "total_queries": total_queries,
            "expanded_queries": expanded_queries,
            "expansion_rate": expanded_queries / total_queries if total_queries > 0 else 0,
            "total_expansion_terms": total_expansion_terms,
            "avg_expansion_terms": total_expansion_terms / expanded_queries if expanded_queries > 0 else 0,
            "avg_processing_time": avg_processing_time
        }
