"""
Tests for base LLM client.
"""

import pytest
from llm.base import BaseLLMClient, LLMResponse, LLMProvider, TranslationRequest, LLMError


class TestLLMResponse:
    """Test LLMResponse dataclass."""
    
    def test_llm_response_creation(self):
        """Test creating LLMResponse."""
        response = LLMResponse(
            content="Hello, World!",
            model="test-model",
            provider=LLMProvider.OLLAMA
        )
        
        assert response.content == "Hello, World!"
        assert response.model == "test-model"
        assert response.provider == LLMProvider.OLLAMA
        assert response.usage is None
        assert response.metadata is None
    
    def test_llm_response_with_usage(self):
        """Test LLMResponse with usage information."""
        response = LLMResponse(
            content="Hello, World!",
            model="test-model",
            provider=LLMProvider.OLLAMA,
            usage={"total_tokens": 100},
            metadata={"finish_reason": "stop"}
        )
        
        assert response.usage == {"total_tokens": 100}
        assert response.metadata == {"finish_reason": "stop"}


class TestTranslationRequest:
    """Test TranslationRequest dataclass."""
    
    def test_translation_request_defaults(self):
        """Test TranslationRequest with default values."""
        request = TranslationRequest(text="Hello, World!")
        
        assert request.text == "Hello, World!"
        assert request.source_lang == "en"
        assert request.target_lang == "zh"
        assert request.context is None
        assert request.max_tokens is None
        assert request.temperature == 0.3
    
    def test_translation_request_custom(self):
        """Test TranslationRequest with custom values."""
        request = TranslationRequest(
            text="Hello, World!",
            source_lang="en",
            target_lang="fr",
            context="Test context",
            max_tokens=1000,
            temperature=0.5
        )
        
        assert request.text == "Hello, World!"
        assert request.source_lang == "en"
        assert request.target_lang == "fr"
        assert request.context == "Test context"
        assert request.max_tokens == 1000
        assert request.temperature == 0.5


class TestLLMProvider:
    """Test LLMProvider enum."""
    
    def test_llm_provider_values(self):
        """Test LLMProvider enum values."""
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.CLAUDE.value == "claude"
        assert LLMProvider.QWEN.value == "qwen"


class TestLLMError:
    """Test LLMError exception."""
    
    def test_llm_error_creation(self):
        """Test creating LLMError."""
        error = LLMError("Test error message")
        assert str(error) == "Test error message"
    
    def test_llm_error_inheritance(self):
        """Test LLMError inheritance."""
        error = LLMError("Test error message")
        assert isinstance(error, Exception)


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing base class functionality."""
    
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        self._health_check_result = True
        self._translate_response = None
        self._translate_exception = None
    
    def set_health_check_result(self, result: bool):
        """Set health check result."""
        self._health_check_result = result
    
    def set_translate_response(self, response: LLMResponse):
        """Set translate response."""
        self._translate_response = response
    
    def set_translate_exception(self, exception: Exception):
        """Set translate exception."""
        self._translate_exception = exception
    
    async def translate(self, request: TranslationRequest) -> LLMResponse:
        """Mock translate method."""
        if self._translate_exception:
            raise self._translate_exception
        return self._translate_response
    
    async def health_check(self) -> bool:
        """Mock health check method."""
        return self._health_check_result
    
    def get_provider(self) -> LLMProvider:
        """Mock get provider method."""
        return LLMProvider.OLLAMA


class TestBaseLLMClient:
    """Test BaseLLMClient abstract class."""
    
    def test_base_llm_client_creation(self):
        """Test creating BaseLLMClient."""
        client = MockLLMClient("test-model", api_key="test-key")
        
        assert client.model == "test-model"
        assert client.config == {"api_key": "test-key"}
    
    def test_build_translation_prompt(self):
        """Test building translation prompt."""
        client = MockLLMClient("test-model")
        request = TranslationRequest(
            text="Hello, World!",
            context="Test context"
        )
        
        prompt = client._build_translation_prompt(request)
        
        assert "Hello, World!" in prompt
        assert "Test context" in prompt
        assert "translate" in prompt.lower()
        assert "chinese" in prompt.lower()
    
    def test_build_translation_prompt_no_context(self):
        """Test building translation prompt without context."""
        client = MockLLMClient("test-model")
        request = TranslationRequest(text="Hello, World!")
        
        prompt = client._build_translation_prompt(request)
        
        assert "Hello, World!" in prompt
        assert "Context:" not in prompt
    
    def test_validate_request_valid(self):
        """Test validating valid request."""
        client = MockLLMClient("test-model")
        request = TranslationRequest(text="Hello, World!")
        
        # Should not raise exception
        client._validate_request(request)
    
    def test_validate_request_empty_text(self):
        """Test validating request with empty text."""
        client = MockLLMClient("test-model")
        request = TranslationRequest(text="")
        
        with pytest.raises(LLMError, match="Translation text cannot be empty"):
            client._validate_request(request)
    
    def test_validate_request_whitespace_text(self):
        """Test validating request with whitespace-only text."""
        client = MockLLMClient("test-model")
        request = TranslationRequest(text="   \n\t   ")
        
        with pytest.raises(LLMError, match="Translation text cannot be empty"):
            client._validate_request(request)
    
    def test_validate_request_too_large(self):
        """Test validating request with text too large."""
        client = MockLLMClient("test-model")
        large_text = "x" * 100001  # Over 100KB limit
        request = TranslationRequest(text=large_text)
        
        with pytest.raises(LLMError, match="Text too long for translation"):
            client._validate_request(request)
    
    def test_str_representation(self):
        """Test string representation."""
        client = MockLLMClient("test-model")
        
        assert str(client) == "ollama:test-model"
