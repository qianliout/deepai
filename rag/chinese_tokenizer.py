"""
中文分词模块

提供两种中文分词方式：
1. 简单手工分词：基于标点符号和空格
2. jieba分词器：使用jieba库进行精确分词

数据流：
文本输入 -> 分词方式选择 -> 分词处理 -> 词汇列表输出
"""

import re
from typing import List, Optional, Set
from enum import Enum
from dataclasses import dataclass

# jieba分词器导入
try:
    import jieba
    import jieba.posseg as pseg
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

from config import config
from logger import get_logger


class TokenizerType(str, Enum):
    """分词器类型"""
    MANUAL = "manual"      # 手工分词
    JIEBA = "jieba"        # jieba分词


@dataclass
class TokenizeResult:
    """分词结果数据类"""
    tokens: List[str]           # 分词结果
    token_count: int           # 词汇数量
    method: str                # 分词方法
    processing_time: float     # 处理时间(秒)
    filtered_count: int        # 过滤掉的词汇数量


class ChineseTokenizer:
    """中文分词器
    
    支持多种分词策略，针对中文文本进行优化处理
    """
    
    def __init__(self, tokenizer_type: TokenizerType = TokenizerType.JIEBA):
        """初始化中文分词器
        
        Args:
            tokenizer_type: 分词器类型
        """
        self.logger = get_logger("ChineseTokenizer")
        self.tokenizer_type = tokenizer_type
        
        # 停用词集合
        self.stop_words = self._load_stop_words()
        
        # 初始化jieba分词器
        if tokenizer_type == TokenizerType.JIEBA and JIEBA_AVAILABLE:
            self._init_jieba()
        elif tokenizer_type == TokenizerType.JIEBA and not JIEBA_AVAILABLE:
            self.logger.warning("jieba未安装，回退到手工分词")
            self.tokenizer_type = TokenizerType.MANUAL
        
        self.logger.info(f"中文分词器初始化完成，使用方法: {self.tokenizer_type}")
    
    def _load_stop_words(self) -> Set[str]:
        """加载停用词表"""
        # 基础中文停用词
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '里', '就是', '还', '把', '比', '或者', '因为', '所以',
            '但是', '如果', '这样', '那样', '可以', '应该', '能够', '已经', '正在',
            '之后', '之前', '当时', '现在', '以后', '以前', '今天', '明天', '昨天',
            '年', '月', '日', '时', '分', '秒', '点', '个', '些', '每', '各', '另',
            '其他', '等等', '什么', '怎么', '为什么', '哪里', '哪个', '多少', '几',
            '第一', '第二', '第三', '最', '更', '最好', '最大', '最小', '非常', '特别',
            '。', '，', '！', '？', '；', '：', '"', '"', ''', ''', '（', '）', '【', '】',
            ' ', '\t', '\n', '\r'
        }
        return stop_words
    
    def _init_jieba(self):
        """初始化jieba分词器"""
        try:
            # 设置jieba日志级别
            jieba.setLogLevel(20)  # INFO级别
            
            # 预加载词典
            jieba.initialize()
            
            self.logger.info("jieba分词器初始化完成")
        except Exception as e:
            self.logger.error(f"jieba初始化失败: {e}")
            self.tokenizer_type = TokenizerType.MANUAL
    
    def tokenize(self, text: str, remove_stop_words: bool = True) -> TokenizeResult:
        """对文本进行分词
        
        Args:
            text: 输入文本
            remove_stop_words: 是否移除停用词
            
        Returns:
            分词结果
        """
        import time
        start_time = time.time()
        
        try:
            if self.tokenizer_type == TokenizerType.JIEBA:
                tokens = self._jieba_tokenize(text)
            else:
                tokens = self._manual_tokenize(text)
            
            # 过滤停用词
            original_count = len(tokens)
            if remove_stop_words:
                tokens = [token for token in tokens if token not in self.stop_words]
            
            processing_time = time.time() - start_time
            filtered_count = original_count - len(tokens)
            
            result = TokenizeResult(
                tokens=tokens,
                token_count=len(tokens),
                method=self.tokenizer_type.value,
                processing_time=processing_time,
                filtered_count=filtered_count
            )
            
            self.logger.debug(
                f"分词完成 | 方法: {self.tokenizer_type} | "
                f"原始词数: {original_count} | 过滤后: {len(tokens)} | "
                f"耗时: {processing_time:.4f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"分词失败: {e}")
            # 返回空结果
            return TokenizeResult(
                tokens=[],
                token_count=0,
                method=self.tokenizer_type.value,
                processing_time=time.time() - start_time,
                filtered_count=0
            )
    
    def _manual_tokenize(self, text: str) -> List[str]:
        """手工分词实现
        
        基于标点符号、空格和中文字符特征进行分词
        """
        # 清理文本
        text = text.strip()
        
        # 使用正则表达式分词
        # 匹配中文词汇、英文单词、数字
        pattern = r'[\u4e00-\u9fff]+|[a-zA-Z]+|[0-9]+(?:\.[0-9]+)?'
        tokens = re.findall(pattern, text)
        
        # 进一步处理中文词汇（简单的基于长度的分割）
        processed_tokens = []
        for token in tokens:
            if self._is_chinese(token):
                # 中文词汇按字符分割（简化处理）
                if len(token) > 4:  # 长词汇进行分割
                    # 尝试按2-3字符分割
                    for i in range(0, len(token), 2):
                        if i + 2 <= len(token):
                            processed_tokens.append(token[i:i+2])
                        elif i < len(token):
                            processed_tokens.append(token[i:])
                else:
                    processed_tokens.append(token)
            else:
                processed_tokens.append(token)
        
        return [token for token in processed_tokens if len(token.strip()) > 0]
    
    def _jieba_tokenize(self, text: str) -> List[str]:
        """jieba分词实现"""
        if not JIEBA_AVAILABLE:
            return self._manual_tokenize(text)
        
        try:
            # 使用jieba精确模式分词
            tokens = list(jieba.cut(text, cut_all=False))
            
            # 过滤空白和单字符标点
            filtered_tokens = []
            for token in tokens:
                token = token.strip()
                if len(token) > 0 and not (len(token) == 1 and not token.isalnum()):
                    filtered_tokens.append(token)
            
            return filtered_tokens
            
        except Exception as e:
            self.logger.warning(f"jieba分词失败，回退到手工分词: {e}")
            return self._manual_tokenize(text)
    
    def _is_chinese(self, text: str) -> bool:
        """判断文本是否包含中文字符"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    def add_user_dict(self, words: List[str]):
        """添加用户自定义词典
        
        Args:
            words: 词汇列表
        """
        if self.tokenizer_type == TokenizerType.JIEBA and JIEBA_AVAILABLE:
            for word in words:
                jieba.add_word(word)
            self.logger.info(f"添加用户词典: {len(words)} 个词汇")
        else:
            self.logger.warning("当前分词器不支持用户词典")
    
    def get_word_frequency(self, text: str, top_k: int = 20) -> List[tuple]:
        """获取词频统计
        
        Args:
            text: 输入文本
            top_k: 返回前k个高频词
            
        Returns:
            (词汇, 频次) 元组列表
        """
        from collections import Counter
        
        result = self.tokenize(text, remove_stop_words=True)
        word_freq = Counter(result.tokens)
        
        return word_freq.most_common(top_k)
    
    def batch_tokenize(self, texts: List[str], remove_stop_words: bool = True) -> List[TokenizeResult]:
        """批量分词
        
        Args:
            texts: 文本列表
            remove_stop_words: 是否移除停用词
            
        Returns:
            分词结果列表
        """
        results = []
        for text in texts:
            result = self.tokenize(text, remove_stop_words)
            results.append(result)
        
        self.logger.info(f"批量分词完成，处理 {len(texts)} 个文本")
        return results
