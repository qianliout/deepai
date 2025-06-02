"""
中文分词模块

提供两种中文分词方式，用于学习不同分词策略的特点和应用场景：
1. SimpleTokenizer：简单手工分词，基于规则和正则表达式
2. JiebaTokenizer：jieba分词器，使用成熟的中文分词库

数据流：
文本输入 -> 分词器选择 -> 分词处理 -> 词汇列表输出

学习要点：
1. 中文分词的基本原理和挑战
2. 规则分词 vs 统计分词的区别
3. 分词效果对RAG系统的影响
"""

import re
from typing import List, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
# jieba分词器导入
import jieba
from logger import get_logger


@dataclass
class TokenizeResult:
    """分词结果数据类"""
    tokens: List[str]  # 分词结果
    token_count: int  # 词汇数量
    method: str  # 分词方法
    processing_time: float  # 处理时间(秒)
    filtered_count: int  # 过滤掉的词汇数量


class BaseTokenizer(ABC):
    """分词器基类
    
    定义分词器的通用接口，用于学习不同分词策略的实现
    """

    def __init__(self):
        """初始化分词器"""
        self.logger = get_logger(self.__class__.__name__)
        self.stop_words = self._load_stop_words()

    @abstractmethod
    def tokenize(self, text: str, remove_stop_words: bool = True) -> TokenizeResult:
        """分词接口
        
        Args:
            text: 输入文本
            remove_stop_words: 是否移除停用词
            
        Returns:
            分词结果
        """
        pass

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


class SimpleTokenizer(BaseTokenizer):
    """简单分词器
    
    基于规则和正则表达式的简单中文分词实现。
    
    学习要点：
    1. 规则分词的基本原理
    2. 正则表达式在文本处理中的应用
    3. 简单分词的优缺点
    """

    def __init__(self):
        """初始化简单分词器"""
        super().__init__()
        self.logger.info("简单分词器初始化完成")

    def tokenize(self, text: str, remove_stop_words: bool = True) -> TokenizeResult:
        """简单分词实现
        
        Args:
            text: 输入文本
            remove_stop_words: 是否移除停用词
            
        Returns:
            分词结果
        """
        import time
        start_time = time.time()

        try:
            # 使用正则表达式进行简单分词
            tokens = self._simple_tokenize(text)

            # 过滤停用词
            original_count = len(tokens)
            if remove_stop_words:
                tokens = [token for token in tokens if token not in self.stop_words]

            processing_time = time.time() - start_time
            filtered_count = original_count - len(tokens)

            result = TokenizeResult(
                tokens=tokens,
                token_count=len(tokens),
                method="simple",
                processing_time=processing_time,
                filtered_count=filtered_count
            )

            self.logger.debug(
                f"简单分词完成 | 原始词数: {original_count} | "
                f"过滤后: {len(tokens)} | 耗时: {processing_time:.4f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"简单分词失败: {e}")
            return TokenizeResult(
                tokens=[],
                token_count=0,
                method="simple",
                processing_time=time.time() - start_time,
                filtered_count=0
            )

    def _simple_tokenize(self, text: str) -> List[str]:
        """简单分词实现
        
        基于正则表达式的简单中文分词
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
                            processed_tokens.append(token[i:i + 2])
                        elif i < len(token):
                            processed_tokens.append(token[i:])
                else:
                    processed_tokens.append(token)
            else:
                processed_tokens.append(token)

        return [token for token in processed_tokens if len(token.strip()) > 0]

    def _is_chinese(self, text: str) -> bool:
        """判断文本是否包含中文字符"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))


class JiebaTokenizer(BaseTokenizer):
    """Jieba分词器
    
    基于jieba库的中文分词实现，使用统计和机器学习方法。
    
    学习要点：
    1. 统计分词的基本原理
    2. HMM和CRF在中文分词中的应用
    3. 词典和统计相结合的分词策略
    """

    def __init__(self):
        """初始化jieba分词器"""
        super().__init__()

        # 初始化jieba
        self._init_jieba()
        self.logger.info("Jieba分词器初始化完成")

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
            raise

    def tokenize(self, text: str, remove_stop_words: bool = True) -> TokenizeResult:
        """jieba分词实现
        
        Args:
            text: 输入文本
            remove_stop_words: 是否移除停用词
            
        Returns:
            分词结果
        """
        import time
        start_time = time.time()

        try:
            # 使用jieba进行分词
            tokens = self._jieba_tokenize(text)

            # 过滤停用词
            original_count = len(tokens)
            if remove_stop_words:
                tokens = [token for token in tokens if token not in self.stop_words]

            processing_time = time.time() - start_time
            filtered_count = original_count - len(tokens)

            result = TokenizeResult(
                tokens=tokens,
                token_count=len(tokens),
                method="jieba",
                processing_time=processing_time,
                filtered_count=filtered_count
            )

            self.logger.debug(
                f"jieba分词完成 | 原始词数: {original_count} | "
                f"过滤后: {len(tokens)} | 耗时: {processing_time:.4f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"jieba分词失败: {e}")
            return TokenizeResult(
                tokens=[],
                token_count=0,
                method="jieba",
                processing_time=time.time() - start_time,
                filtered_count=0
            )

    def _jieba_tokenize(self, text: str) -> List[str]:
        """jieba分词实现"""
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
            self.logger.error(f"jieba分词失败: {e}")
            return []

    def add_user_dict(self, words: List[str]):
        """添加用户自定义词典
        
        Args:
            words: 词汇列表
        """
        for word in words:
            jieba.add_word(word)
        self.logger.info(f"添加用户词典: {len(words)} 个词汇")


# 为了兼容性，保留原来的类名和工厂函数
def create_tokenizer(tokenizer_type: str = "jieba") -> BaseTokenizer:
    """创建分词器工厂函数
    
    Args:
        tokenizer_type: 分词器类型 ("simple" 或 "jieba")
        
    Returns:
        分词器实例
    """
    if tokenizer_type.lower() == "simple":
        return SimpleTokenizer()
    elif tokenizer_type.lower() == "jieba":
        return JiebaTokenizer()
    else:
        raise ValueError(f"不支持的分词器类型: {tokenizer_type}")
