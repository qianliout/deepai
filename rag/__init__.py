"""
工具模块

该模块包含RAG系统的工具函数：
- logger: 日志工具
- metrics: 评估指标
- helpers: 辅助函数
"""

from .logger import get_logger

try:
    from .metrics import RAGMetrics
except ImportError:
    RAGMetrics = None

try:
    from .helpers import *
except ImportError:
    pass

__all__ = [
    "get_logger"
]

if RAGMetrics:
    __all__.append("RAGMetrics")
