"""
Mock tests for LLM clients (when actual services are not available).
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from llm.base import LLMResponse, LLMProvider, TranslationRequest, LLMError


class TestMockLLMClient:
    """Test mock LLM client functionality."""
    
    @pytest.mark.asyncio
    async def test_mock_translation_success(self, mock_llm_client, sample_translation_request):
        """Test successful mock translation."""
        response = await mock_llm_client.translate(sample_translation_request)
        
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert response.model == "test-model"
        assert response.provider == LLMProvider.OLLAMA
    
    @pytest.mark.asyncio
    async def test_mock_translation_failure(self, mock_llm_client_failing, sample_translation_request):
        """Test mock translation failure."""
        with pytest.raises(Exception, match="Translation failed"):
            await mock_llm_client_failing.translate(sample_translation_request)
    
    @pytest.mark.asyncio
    async def test_mock_health_check_success(self, mock_llm_client):
        """Test successful mock health check."""
        result = await mock_llm_client.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_mock_health_check_failure(self, mock_llm_client_failing):
        """Test failed mock health check."""
        result = await mock_llm_client_failing.health_check()
        assert result is False
    
    def test_mock_provider_type(self, mock_llm_client):
        """Test mock provider type."""
        provider = mock_llm_client.get_provider()
        assert provider == LLMProvider.OLLAMA


class TestLLMClientMocking:
    """Test LLM client mocking patterns."""
    
    @pytest.mark.asyncio
    async def test_openai_client_mock(self):
        """Test mocking OpenAI client."""
        with patch('llm.openai_client.AsyncOpenAI') as mock_openai:
            # Mock the client and response
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Translated text"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15
            
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client
            
            # Import and test
            from llm.openai_client import OpenAIClient
            
            client = OpenAIClient("gpt-3.5-turbo", "test-key")
            request = TranslationRequest(text="Hello, World!")
            
            response = await client.translate(request)
            
            assert response.content == "Translated text"
            assert response.provider == LLMProvider.OPENAI
    
    @pytest.mark.asyncio
    async def test_claude_client_mock(self):
        """Test mocking Claude client."""
        with patch('llm.claude_client.AsyncAnthropic') as mock_anthropic:
            # Mock the client and response
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = "Translated text"
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 5
            
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client
            
            # Import and test
            from llm.claude_client import ClaudeClient
            
            client = ClaudeClient("claude-3-sonnet-20240229", "test-key")
            request = TranslationRequest(text="Hello, World!")
            
            response = await client.translate(request)
            
            assert response.content == "Translated text"
            assert response.provider == LLMProvider.CLAUDE
    
    @pytest.mark.asyncio
    async def test_qwen_client_mock(self):
        """Test mocking Qwen client."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock the session and response
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "choices": [{"message": {"content": "Translated text"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            })
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            # Import and test
            from llm.qwen_client import QwenClient
            
            client = QwenClient("qwen-turbo", "test-key")
            request = TranslationRequest(text="Hello, World!")
            
            response = await client.translate(request)
            
            assert response.content == "Translated text"
            assert response.provider == LLMProvider.QWEN


class TestLLMErrorHandling:
    """Test LLM error handling with mocks."""
    
    @pytest.mark.asyncio
    async def test_network_error_mock(self):
        """Test network error handling."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.side_effect = Exception("Network error")
            
            from llm.ollama_client import OllamaClient
            
            client = OllamaClient("test-model")
            request = TranslationRequest(text="Hello, World!")
            
            with pytest.raises(LLMError, match="Network error"):
                await client.translate(request)
    
    @pytest.mark.asyncio
    async def test_timeout_error_mock(self):
        """Test timeout error handling."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.side_effect = asyncio.TimeoutError()
            
            from llm.ollama_client import OllamaClient
            
            client = OllamaClient("test-model")
            request = TranslationRequest(text="Hello, World!")
            
            with pytest.raises(LLMError, match="timed out"):
                await client.translate(request)
    
    @pytest.mark.asyncio
    async def test_api_error_mock(self):
        """Test API error handling."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            from llm.ollama_client import OllamaClient
            
            client = OllamaClient("test-model")
            request = TranslationRequest(text="Hello, World!")
            
            with pytest.raises(LLMError, match="API error 500"):
                await client.translate(request)
