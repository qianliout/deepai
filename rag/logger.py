"""
简化日志模块 - 参考BERT项目的日志方式
使用标准库logging，简单直接
"""

import logging


def get_logger(name: str):
    """获取日志器 - 简化版本"""
    return logging.getLogger(name)


def log_execution_time(operation_name: str = None):
    """装饰器：记录函数执行时间"""
    import time
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger = get_logger("performance")
                logger.info(f"操作: {operation_name or func.__name__} | 耗时: {duration:.4f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger = get_logger("performance")
                logger.error(f"操作失败: {operation_name or func.__name__} | 耗时: {duration:.4f}s | 错误: {e}")
                raise
        return wrapper
    return decorator


class LogExecutionTime:
    """上下文管理器：记录代码块执行时间"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        logger = get_logger("performance")
        if exc_type is None:
            logger.info(f"操作: {self.operation_name} | 耗时: {duration:.4f}s")
        else:
            logger.error(f"操作失败: {self.operation_name} | 耗时: {duration:.4f}s | 错误: {exc_val}")
