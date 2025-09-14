"""
Custom exceptions for the translation system.
"""


class TranslationError(Exception):
    """Base exception for translation-related errors."""
    pass


class FileProcessingError(TranslationError):
    """Exception raised for file processing errors."""
    pass


class LLMServiceError(TranslationError):
    """Exception raised for LLM service errors."""
    pass


class ConfigurationError(TranslationError):
    """Exception raised for configuration errors."""
    pass


class ModelNotFoundError(LLMServiceError):
    """Exception raised when a model is not found."""
    pass


class APIKeyError(LLMServiceError):
    """Exception raised when API key is missing or invalid."""
    pass


class NetworkError(LLMServiceError):
    """Exception raised for network-related errors."""
    pass


class TranslationTimeoutError(LLMServiceError):
    """Exception raised when translation times out."""
    pass


class FileSizeError(FileProcessingError):
    """Exception raised when file is too large."""
    pass


class UnsupportedFileTypeError(FileProcessingError):
    """Exception raised for unsupported file types."""
    pass


class BackupError(FileProcessingError):
    """Exception raised when backup creation fails."""
    pass


class ValidationError(FileProcessingError):
    """Exception raised when file validation fails."""
    pass
