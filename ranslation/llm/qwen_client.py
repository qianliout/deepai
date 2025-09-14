"""
Qwen (通义千问) API client implementation.
"""

import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any
from llm.base import BaseLLMClient, LLMResponse, LLMProvider, TranslationRequest, LLMError


class QwenClient(BaseLLMClient):
    """Client for Qwen (通义千问) API."""
    
    def __init__(self, model: str, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/api/v1", **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def translate(self, request: TranslationRequest) -> LLMResponse:
        """Translate text using Qwen API."""
        self._validate_request(request)
        
        prompt = self._build_translation_prompt(request)
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个专业的翻译专家，专门翻译技术文档。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": request.max_tokens or 4000,
            "temperature": request.temperature,
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"Qwen API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "choices" not in result or not result["choices"]:
                        raise LLMError("Invalid response from Qwen API")
                    
                    content = result["choices"][0]["message"]["content"]
                    if not content:
                        raise LLMError("Empty response from Qwen API")
                    
                    return LLMResponse(
                        content=content.strip(),
                        model=self.model,
                        provider=LLMProvider.QWEN,
                        usage={
                            "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                            "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
                            "total_tokens": result.get("usage", {}).get("total_tokens", 0)
                        },
                        metadata={
                            "finish_reason": result["choices"][0].get("finish_reason"),
                            "model": result.get("model")
                        }
                    )
                    
        except aiohttp.ClientError as e:
            raise LLMError(f"Network error connecting to Qwen API: {e}")
        except asyncio.TimeoutError:
            raise LLMError("Qwen request timed out")
        except Exception as e:
            raise LLMError(f"Unexpected error in Qwen translation: {e}")
    
    async def health_check(self) -> bool:
        """Check if Qwen API is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 1
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    
    def get_provider(self) -> LLMProvider:
        """Get the provider type."""
        return LLMProvider.QWEN
