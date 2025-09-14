"""
Ollama local LLM client implementation.
"""

import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any
from llm.base import BaseLLMClient, LLMResponse, LLMProvider, TranslationRequest, LLMError


class OllamaClient(BaseLLMClient):
    """Client for Ollama local LLM service."""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"
        
    async def translate(self, request: TranslationRequest) -> LLMResponse:
        """Translate text using Ollama."""
        self._validate_request(request)
        
        prompt = self._build_translation_prompt(request)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Lower temperature for more consistent translation
                "num_predict": request.max_tokens or 4000
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"Ollama API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "response" not in result:
                        raise LLMError("Invalid response from Ollama API")
                    
                    # Clean up the response by removing thinking tags and extra content
                    content = result["response"].strip()
                    content = self._clean_translation_output(content)
                    
                    return LLMResponse(
                        content=content,
                        model=self.model,
                        provider=LLMProvider.OLLAMA,
                        usage={
                            "prompt_tokens": result.get("prompt_eval_count", 0),
                            "completion_tokens": result.get("eval_count", 0),
                            "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                        },
                        metadata={
                            "eval_duration": result.get("eval_duration", 0),
                            "load_duration": result.get("load_duration", 0),
                            "total_duration": result.get("total_duration", 0)
                        }
                    )
                    
        except aiohttp.ClientError as e:
            raise LLMError(f"Network error connecting to Ollama: {e}")
        except asyncio.TimeoutError:
            raise LLMError("Ollama request timed out")
        except Exception as e:
            raise LLMError(f"Unexpected error in Ollama translation: {e}")
    
    async def health_check(self) -> bool:
        """Check if Ollama service is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    
    def get_provider(self) -> LLMProvider:
        """Get the provider type."""
        return LLMProvider.OLLAMA
    
    def _clean_translation_output(self, content: str) -> str:
        """Clean up translation output by removing thinking tags and extra content."""
        # Remove <think> tags and their content
        import re
        
        # Remove <think>...</think> blocks
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Remove any remaining <think> tags
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        
        # Remove common prefixes that models might add
        prefixes_to_remove = [
            "Here's a structured summary",
            "Here's the translation",
            "Here's the translated text",
            "Translation:",
            "Translated text:",
            "以下是翻译结果",
            "翻译结果：",
            "翻译："
        ]
        
        for prefix in prefixes_to_remove:
            if content.startswith(prefix):
                content = content[len(prefix):].strip()
        
        # Remove any leading/trailing whitespace and newlines
        content = content.strip()
        
        return content
    
    async def list_models(self) -> list:
        """List available models in Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return [model["name"] for model in result.get("models", [])]
                    return []
        except Exception:
            return []
