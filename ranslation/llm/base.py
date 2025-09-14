"""
Base LLM client interface and common types.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    CLAUDE = "claude"
    QWEN = "qwen"


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    provider: LLMProvider
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TranslationRequest:
    """Translation request parameters."""
    text: str
    source_lang: str = "en"
    target_lang: str = "zh"
    context: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.3


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs
        
    @abstractmethod
    async def translate(self, request: TranslationRequest) -> LLMResponse:
        """
        Translate text using the LLM.
        
        Args:
            request: Translation request parameters
            
        Returns:
            LLMResponse with translated text
            
        Raises:
            LLMError: If translation fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM service is available.
        
        Returns:
            True if service is available, False otherwise
        """
        pass
    
    @abstractmethod
    def get_provider(self) -> LLMProvider:
        """Get the provider type."""
        pass
    
    def _build_translation_prompt(self, request: TranslationRequest) -> str:
        """Build translation prompt based on request."""
        context_part = f"\n\n上下文: {request.context}" if request.context else ""
        
        prompt = f"""
        Translate the following English text to Chinese. Keep the markdown formatting exactly the same. 
        Only output the translated text:

{request.text}"""
        
        return prompt
    
    def _validate_request(self, request: TranslationRequest) -> None:
        """Validate translation request."""
        if not request.text or not request.text.strip():
            raise LLMError("Translation text cannot be empty")
        
        if len(request.text) > 1000000:  # 1000KB limit
            raise LLMError("Text too long for translation (max 1000KB)")
    
    def __str__(self) -> str:
        return f"{self.get_provider().value}:{self.model}"
