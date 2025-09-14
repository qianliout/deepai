"""
Logging configuration and utilities.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import colorlog


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)
        max_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatters
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    file_formatter = logging.Formatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_file_logging(
    log_file: str,
    level: str = "INFO",
    max_size: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up file-only logging (no console output).
    
    Args:
        log_file: Path to log file
        level: Logging level
        max_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("file_logger")
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


class TranslationLogger:
    """Specialized logger for translation operations."""
    
    def __init__(self, name: str = "translation"):
        self.logger = get_logger(name)
        self.stats = {
            "files_processed": 0,
            "files_successful": 0,
            "files_failed": 0,
            "total_tokens": 0,
            "start_time": None,
            "end_time": None
        }
    
    def start_translation(self, total_files: int):
        """Log start of translation process."""
        import time
        self.stats["start_time"] = time.time()
        self.logger.info(f"Starting translation of {total_files} files")
    
    def end_translation(self):
        """Log end of translation process."""
        import time
        self.stats["end_time"] = time.time()
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        self.logger.info(
            f"Translation completed in {duration:.2f} seconds. "
            f"Processed: {self.stats['files_processed']}, "
            f"Successful: {self.stats['files_successful']}, "
            f"Failed: {self.stats['files_failed']}, "
            f"Total tokens: {self.stats['total_tokens']}"
        )
    
    def log_file_start(self, file_path: str):
        """Log start of file translation."""
        self.logger.info(f"Translating: {file_path}")
    
    def log_file_success(self, file_path: str, tokens: int = 0):
        """Log successful file translation."""
        self.stats["files_processed"] += 1
        self.stats["files_successful"] += 1
        self.stats["total_tokens"] += tokens
        self.logger.info(f"✓ Successfully translated: {file_path}")
    
    def log_file_error(self, file_path: str, error: str):
        """Log file translation error."""
        self.stats["files_processed"] += 1
        self.stats["files_failed"] += 1
        self.logger.error(f"✗ Failed to translate {file_path}: {error}")
    
    def log_progress(self, current: int, total: int):
        """Log translation progress."""
        percentage = (current / total) * 100
        self.logger.info(f"Progress: {current}/{total} ({percentage:.1f}%)")
    
    def get_stats(self) -> dict:
        """Get translation statistics."""
        return self.stats.copy()
