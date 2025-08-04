"""
RAG2项目统一日志配置
使用loguru进行日志管理
"""

import os
import sys
import time
from pathlib import Path
from loguru import logger
from typing import Optional

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ..config.config import get_config
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from config.config import get_config

class LoggerConfig:
    """日志配置类"""
    
    def __init__(self):
        self.config = get_config()
        self.log_dir = Path("data/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 移除默认的logger
        logger.remove()
        
        # 配置日志格式
        self.log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        # 配置控制台日志
        self._setup_console_logging()
        
        # 配置文件日志
        self._setup_file_logging()
        
        # 配置特殊日志
        self._setup_special_logging()
    
    def _setup_console_logging(self):
        """配置控制台日志"""
        log_level = "DEBUG" if self.config.debug else "INFO"
        
        logger.add(
            sys.stdout,
            format=self.log_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
    
    def _setup_file_logging(self):
        """配置文件日志"""
        # 应用主日志
        logger.add(
            self.log_dir / "rag2.log",
            format=self.log_format,
            level="INFO",
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
        )
        
        # 错误日志
        logger.add(
            self.log_dir / "error.log",
            format=self.log_format,
            level="ERROR",
            rotation="50 MB",
            retention="60 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
        )
        
        # 调试日志（仅开发环境）
        if self.config.debug:
            logger.add(
                self.log_dir / "debug.log",
                format=self.log_format,
                level="DEBUG",
                rotation="200 MB",
                retention="7 days",
                compression="zip",
                backtrace=True,
                diagnose=True,
            )
    
    def _setup_special_logging(self):
        """配置特殊用途的日志"""
        # API访问日志
        logger.add(
            self.log_dir / "api_access.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
            level="INFO",
            rotation="50 MB",
            retention="30 days",
            compression="zip",
            filter=lambda record: record["extra"].get("logger_type") == "api_access"
        )
        
        # 查询日志
        logger.add(
            self.log_dir / "query.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
            level="INFO",
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            filter=lambda record: record["extra"].get("logger_type") == "query"
        )
        
        # 检索日志
        logger.add(
            self.log_dir / "retrieval.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
            level="INFO",
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            filter=lambda record: record["extra"].get("logger_type") == "retrieval"
        )
        
        # 模型调用日志
        logger.add(
            self.log_dir / "model.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
            level="INFO",
            rotation="50 MB",
            retention="30 days",
            compression="zip",
            filter=lambda record: record["extra"].get("logger_type") == "model"
        )
        
        # 性能日志
        logger.add(
            self.log_dir / "performance.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
            level="INFO",
            rotation="50 MB",
            retention="30 days",
            compression="zip",
            filter=lambda record: record["extra"].get("logger_type") == "performance"
        )

# 初始化日志配置
_logger_config = LoggerConfig()

# 创建专用logger
def get_logger(name: str = "rag2"):
    """获取logger实例"""
    return logger.bind(name=name)

def get_api_logger():
    """获取API访问logger"""
    return logger.bind(logger_type="api_access")

def get_query_logger():
    """获取查询logger"""
    return logger.bind(logger_type="query")

def get_retrieval_logger():
    """获取检索logger"""
    return logger.bind(logger_type="retrieval")

def get_model_logger():
    """获取模型调用logger"""
    return logger.bind(logger_type="model")

def get_performance_logger():
    """获取性能logger"""
    return logger.bind(logger_type="performance")

# 日志装饰器
def log_function_call(logger_instance=None):
    """函数调用日志装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            log = logger_instance or logger
            log.info(f"调用函数: {func.__name__}, 参数: args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                log.info(f"函数 {func.__name__} 执行成功")
                return result
            except Exception as e:
                log.error(f"函数 {func.__name__} 执行失败: {str(e)}")
                raise
        return wrapper
    return decorator

def log_performance(logger_instance=None):
    """性能监控装饰器"""
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            log = logger_instance or get_performance_logger()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                
                log.info(f"函数: {func.__name__}, 执行时间: {execution_time:.4f}秒")
                return result
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                log.error(f"函数: {func.__name__}, 执行失败, 执行时间: {execution_time:.4f}秒, 错误: {str(e)}")
                raise
        return wrapper
    return decorator

# 上下文管理器
class LogContext:
    """日志上下文管理器"""
    
    def __init__(self, operation: str, logger_instance=None, **context):
        self.operation = operation
        self.logger = logger_instance or logger
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"开始操作: {self.operation}", **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"操作完成: {self.operation}, 耗时: {duration:.4f}秒", **self.context)
        else:
            self.logger.error(f"操作失败: {self.operation}, 耗时: {duration:.4f}秒, 错误: {exc_val}", **self.context)

# 便捷函数
def log_query(query: str, user_id: Optional[str] = None, session_id: Optional[str] = None):
    """记录用户查询"""
    query_log = get_query_logger()
    query_log.info(f"用户查询: {query}", user_id=user_id, session_id=session_id)

def log_retrieval(query: str, retrieved_count: int, method: str, duration: float):
    """记录检索结果"""
    retrieval_log = get_retrieval_logger()
    retrieval_log.info(f"检索完成: 查询='{query}', 方法={method}, 结果数={retrieved_count}, 耗时={duration:.4f}秒")

def log_model_call(model_name: str, input_tokens: int, output_tokens: int, duration: float):
    """记录模型调用"""
    model_log = get_model_logger()
    model_log.info(f"模型调用: {model_name}, 输入tokens={input_tokens}, 输出tokens={output_tokens}, 耗时={duration:.4f}秒")

def log_api_access(method: str, path: str, status_code: int, duration: float, user_id: Optional[str] = None):
    """记录API访问"""
    api_log = get_api_logger()
    api_log.info(f"API访问: {method} {path}, 状态码={status_code}, 耗时={duration:.4f}秒", user_id=user_id)
