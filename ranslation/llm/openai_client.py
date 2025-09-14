"""
OpenAI API client implementation.
"""

import asyncio
from typing import Optional, Dict, Any
from llm.base import BaseLLMClient, LLMResponse, LLMProvider, TranslationRequest, LLMError

try:
    import openai
    from openai import AsyncOpenAI
except ImportError:
    openai = None
    AsyncOpenAI = None


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API."""
    
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        
        if not openai:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    async def translate(self, request: TranslationRequest) -> LLMResponse:
        """Translate text using OpenAI API."""
        self._validate_request(request)
        
        prompt = self._build_translation_prompt(request)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional translator specializing in technical documentation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request.max_tokens or 4000,
                temperature=request.temperature,
                timeout=300  # 5 minutes timeout
            )
            
            content = response.choices[0].message.content
            if not content:
                raise LLMError("Empty response from OpenAI API")
            
            return LLMResponse(
                content=content.strip(),
                model=self.model,
                provider=LLMProvider.OPENAI,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model
                }
            )
            
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}")
        except asyncio.TimeoutError:
            raise LLMError("OpenAI request timed out")
        except Exception as e:
            raise LLMError(f"Unexpected error in OpenAI translation: {e}")
    
    async def health_check(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            # Simple test request
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=10
            )
            return True
        except Exception:
            return False
    
    def get_provider(self) -> LLMProvider:
        """Get the provider type."""
        return LLMProvider.OPENAI
