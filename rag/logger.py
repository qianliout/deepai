"""
日志工具模块

提供统一的日志配置和管理功能，支持文件和控制台输出。
使用loguru库实现高性能日志记录。

数据流：
1. 日志配置加载 -> 日志器初始化 -> 日志输出
2. 支持按模块、级别、时间进行日志分类和过滤
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# 确保日志目录存在
Path('logs').mkdir(exist_ok=True)

# 使用标准库logging，避免loguru的复杂性
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


class LoggerManager:
    """日志管理器

    统一管理RAG系统的日志配置和输出
    """

    def __init__(self):
        self._initialized = True  # 使用标准库logging，已经初始化
        self._loggers = {}

    def setup_logger(self, name: Optional[str] = None) -> None:
        """设置日志器

        Args:
            name: 日志器名称，用于区分不同模块的日志
        """
        # 使用标准库logging，无需额外设置
        pass

    def get_logger(self, name: str):
        """获取指定名称的日志器

        Args:
            name: 日志器名称

        Returns:
            配置好的日志器实例
        """
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)

        return self._loggers[name]

    def log_performance(self, operation: str, duration: float, **kwargs):
        """记录性能日志

        Args:
            operation: 操作名称
            duration: 执行时间(秒)
            **kwargs: 额外的性能指标
        """
        perf_logger = logging.getLogger("performance")
        perf_logger.info(f"性能监控 | 操作: {operation} | 耗时: {duration:.4f}s | 详情: {kwargs}")


# 全局日志管理器实例
_logger_manager = LoggerManager()


# 便捷函数
def setup_logger(name: Optional[str] = None) -> None:
    """设置日志器的便捷函数"""
    _logger_manager.setup_logger(name)


def get_logger(name: str):
    """获取日志器的便捷函数"""
    log =  _logger_manager.get_logger(name)
    return log


def log_performance(operation: str, duration: float, **kwargs):
    """记录性能日志的便捷函数"""
    _logger_manager.log_performance(operation, duration, **kwargs)


# 装饰器：自动记录函数执行时间
def log_execution_time(operation_name: Optional[str] = None):
    """装饰器：自动记录函数执行时间

    Args:
        operation_name: 操作名称，默认使用函数名
    """
    import time
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                log_performance(op_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                log_performance(f"{op_name}_ERROR", duration, error=str(e))
                raise

        return wrapper

    return decorator


# 上下文管理器：记录代码块执行时间
class LogExecutionTime:
    """上下文管理器：记录代码块执行时间"""

    def __init__(self, operation_name: str, **extra_info):
        self.operation_name = operation_name
        self.extra_info = extra_info
        self.start_time = None

    def __enter__(self):
        import time

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        duration = time.time() - self.start_time
        if exc_type is None:
            log_performance(self.operation_name, duration, **self.extra_info)
        else:
            log_performance(f"{self.operation_name}_ERROR", duration, error=str(exc_val), **self.extra_info)
