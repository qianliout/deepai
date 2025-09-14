"""
Claude API client implementation.
"""

import asyncio
from typing import Optional, Dict, Any
from llm.base import BaseLLMClient, LLMResponse, LLMProvider, TranslationRequest, LLMError

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:
    anthropic = None
    AsyncAnthropic = None


class ClaudeClient(BaseLLMClient):
    """Client for Claude API."""
    
    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(model, **kwargs)
        
        if not anthropic:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
        
        self.client = AsyncAnthropic(api_key=api_key)
    
    async def translate(self, request: TranslationRequest) -> LLMResponse:
        """Translate text using Claude API."""
        self._validate_request(request)
        
        prompt = self._build_translation_prompt(request)
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens or 4000,
                temperature=request.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                timeout=300  # 5 minutes timeout
            )
            
            content = response.content[0].text
            if not content:
                raise LLMError("Empty response from Claude API")
            
            return LLMResponse(
                content=content.strip(),
                model=self.model,
                provider=LLMProvider.CLAUDE,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                metadata={
                    "stop_reason": response.stop_reason,
                    "model": response.model
                }
            )
            
        except anthropic.APIError as e:
            raise LLMError(f"Claude API error: {e}")
        except asyncio.TimeoutError:
            raise LLMError("Claude request timed out")
        except Exception as e:
            raise LLMError(f"Unexpected error in Claude translation: {e}")
    
    async def health_check(self) -> bool:
        """Check if Claude API is available."""
        try:
            # Simple test request
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}],
                timeout=10
            )
            return True
        except Exception:
            return False
    
    def get_provider(self) -> LLMProvider:
        """Get the provider type."""
        return LLMProvider.CLAUDE
