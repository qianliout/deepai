"""
Translation engine for markdown content.
"""

import asyncio
import logging
from typing import Optional, List
from pathlib import Path

from llm.base import BaseLLMClient, TranslationRequest, LLMResponse, LLMError
from core.file_scanner import MarkdownFile


class Translator:
    """Handles translation of markdown content using LLM clients."""
    
    def __init__(self, llm_client: BaseLLMClient, logger: Optional[logging.Logger] = None):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        self.translation_stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "total_tokens": 0
        }
    
    async def translate_file(self, file_info: MarkdownFile, context: Optional[str] = None) -> str:
        """
        Translate a single markdown file.
        
        Args:
            file_info: MarkdownFile object containing file information
            context: Optional context for translation
            
        Returns:
            Translated content as string
            
        Raises:
            LLMError: If translation fails
        """
        self.logger.info(f"Starting translation of: {file_info.relative_path}")
        
        try:
            # Read file content
            content = file_info.path.read_text(encoding='utf-8')
            self.logger.debug(f"Read {len(content)} characters from {file_info.relative_path}")
            
            # Create translation request
            request = TranslationRequest(
                text=content,
                source_lang="en",
                target_lang="zh",
                context=context,
                temperature=0.3
            )
            
            # Perform translation
            response = await self.llm_client.translate(request)
            
            # Update statistics
            self.translation_stats["total_files"] += 1
            self.translation_stats["successful"] += 1
            if response.usage:
                self.translation_stats["total_tokens"] += response.usage.get("total_tokens", 0)
            
            self.logger.info(
                f"Successfully translated {file_info.relative_path} "
                f"({len(response.content)} characters)"
            )
            
            if response.usage:
                self.logger.debug(
                    f"Translation used {response.usage.get('total_tokens', 0)} tokens"
                )
            
            return response.content
            
        except Exception as e:
            self.translation_stats["total_files"] += 1
            self.translation_stats["failed"] += 1
            
            error_msg = f"Failed to translate {file_info.relative_path}: {e}"
            self.logger.error(error_msg)
            raise LLMError(error_msg) from e
    
    async def translate_files(
        self, 
        file_list: List[MarkdownFile], 
        context: Optional[str] = None,
        max_concurrent: int = 3
    ) -> List[tuple[MarkdownFile, str]]:
        """
        Translate multiple files with concurrency control.
        
        Args:
            file_list: List of MarkdownFile objects
            context: Optional context for translation
            max_concurrent: Maximum number of concurrent translations
            
        Returns:
            List of tuples (MarkdownFile, translated_content)
        """
        self.logger.info(f"Starting translation of {len(file_list)} files")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def translate_single(file_info: MarkdownFile) -> tuple[MarkdownFile, str]:
            async with semaphore:
                try:
                    content = await self.translate_file(file_info, context)
                    return (file_info, content)
                except Exception as e:
                    self.logger.error(f"Translation failed for {file_info.relative_path}: {e}")
                    raise
        
        # Create tasks for all files
        tasks = [translate_single(file_info) for file_info in file_list]
        
        try:
            # Execute all translations concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Separate successful results from exceptions
            successful_results = []
            failed_count = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_count += 1
                    self.logger.error(f"Translation failed for {file_list[i].relative_path}: {result}")
                else:
                    successful_results.append(result)
            
            self.logger.info(
                f"Translation completed: {len(successful_results)} successful, "
                f"{failed_count} failed"
            )
            
            return successful_results
            
        except Exception as e:
            self.logger.error(f"Error during batch translation: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if the translation service is available."""
        try:
            return await self.llm_client.health_check()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get translation statistics."""
        return self.translation_stats.copy()
    
    def reset_stats(self):
        """Reset translation statistics."""
        self.translation_stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "total_tokens": 0
        }
    
    def get_success_rate(self) -> float:
        """Get translation success rate."""
        if self.translation_stats["total_files"] == 0:
            return 0.0
        
        return self.translation_stats["successful"] / self.translation_stats["total_files"]
