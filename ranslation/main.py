#!/usr/bin/env python3
"""
Main entry point for the markdown translation system.

This script provides a command-line interface for translating markdown files
using various LLM providers including Ollama, OpenAI, Claude, and Qwen.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional, List

from config.settings import load_settings, get_default_config_path
from config.models import load_model_configs, get_model_config
from llm import OllamaClient, OpenAIClient, ClaudeClient, QwenClient
from core import FileScanner, Translator, FileProcessor
from utils.logger_config import setup_logging, TranslationLogger
from utils.exceptions import TranslationError, ConfigurationError


class TranslationApp:
    """Main application class for markdown translation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the translation application."""
        self.settings = load_settings(config_path)
        self.logger = setup_logging(
            level=self.settings.logging.level,
            log_file=self.settings.logging.file,
            log_format=self.settings.logging.format
        )
        self.translation_logger = TranslationLogger()
        
        # Initialize components
        self.file_scanner = FileScanner(self.logger)
        self.file_processor = FileProcessor(self.logger)
        
        # LLM client will be initialized when needed
        self.llm_client = None
        self.translator = None
    
    def _create_llm_client(self) -> None:
        """Create LLM client based on configuration."""
        try:
            if self.settings.model_provider == "ollama":
                self.llm_client = OllamaClient(
                    model=self.settings.model_name,
                    base_url=self.settings.ollama_url
                )
            elif self.settings.model_provider == "openai":
                if not self.settings.openai_api_key:
                    raise ConfigurationError("OpenAI API key not provided")
                self.llm_client = OpenAIClient(
                    model=self.settings.model_name,
                    api_key=self.settings.openai_api_key,
                    base_url=self.settings.openai_base_url
                )
            elif self.settings.model_provider == "claude":
                if not self.settings.claude_api_key:
                    raise ConfigurationError("Claude API key not provided")
                self.llm_client = ClaudeClient(
                    model=self.settings.model_name,
                    api_key=self.settings.claude_api_key
                )
            elif self.settings.model_provider == "qwen":
                if not self.settings.qwen_api_key:
                    raise ConfigurationError("Qwen API key not provided")
                self.llm_client = QwenClient(
                    model=self.settings.model_name,
                    api_key=self.settings.qwen_api_key,
                    base_url=self.settings.qwen_base_url
                )
            else:
                raise ConfigurationError(f"Unsupported model provider: {self.settings.model_provider}")
            
            self.translator = Translator(self.llm_client, self.logger)
            self.logger.info(f"Initialized {self.settings.model_provider} client with model {self.settings.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create LLM client: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if all services are healthy."""
        try:
            if not self.llm_client:
                self._create_llm_client()
            
            is_healthy = await self.translator.health_check()
            if is_healthy:
                self.logger.info("All services are healthy")
            else:
                self.logger.warning("Some services are not available")
            
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def translate_directory(
        self, 
        directory: Path, 
        output_dir: Optional[Path] = None,
        recursive: bool = True,
        skip_existing: bool = True,
        context: Optional[str] = None
    ) -> List[Path]:
        """
        Translate all markdown files in a directory.
        
        Args:
            directory: Directory to scan for markdown files
            output_dir: Output directory for translated files
            recursive: Whether to scan subdirectories
            skip_existing: Whether to skip files with existing translations
            context: Optional context for translation
            
        Returns:
            List of paths to translated files
        """
        self.logger.info(f"Starting translation of directory: {directory}")
        
        try:
            # Initialize LLM client
            if not self.llm_client:
                self._create_llm_client()
            
            # Scan for markdown files
            if recursive:
                files = self.file_scanner.scan_git_repo(directory)
            else:
                files = self.file_scanner.scan_directory(directory, recursive=False)
            
            if not files:
                self.logger.warning("No markdown files found")
                return []
            
            # Filter existing translations if requested
            if skip_existing:
                files = self.file_scanner.filter_existing_translations(files)
                if not files:
                    self.logger.info("All files already have translations")
                    return []
            
            # Log file statistics
            stats = self.file_scanner.get_file_stats(files)
            self.logger.info(f"Found {stats['total_files']} files to translate")
            self.logger.info(f"Total size: {stats['total_size']} bytes")
            
            # Start translation
            self.translation_logger.start_translation(len(files))
            
            # Translate files
            translations = await self.translator.translate_files(
                files, 
                context=context,
                max_concurrent=self.settings.translation.max_concurrent
            )
            
            # Process translations (save to files)
            results = self.file_processor.process_translations(translations, output_dir)
            
            # Log results
            successful = [r.output_path for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            self.logger.info(f"Translation completed: {len(successful)} successful, {len(failed)} failed")
            
            if failed:
                self.logger.warning("Failed translations:")
                for result in failed:
                    self.logger.warning(f"  - {result.file_info.relative_path}: {result.error_message}")
            
            # End translation logging
            self.translation_logger.end_translation()
            
            return successful
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            raise
    
    async def translate_file(
        self, 
        file_path: Path, 
        output_path: Optional[Path] = None,
        context: Optional[str] = None
    ) -> Optional[Path]:
        """
        Translate a single markdown file.
        
        Args:
            file_path: Path to markdown file
            output_path: Output path for translated file
            context: Optional context for translation
            
        Returns:
            Path to translated file or None if failed
        """
        self.logger.info(f"Starting translation of file: {file_path}")
        
        try:
            # Initialize LLM client
            if not self.llm_client:
                self._create_llm_client()
            
            # Create MarkdownFile object
            from core.file_scanner import MarkdownFile
            file_info = MarkdownFile(
                path=file_path,
                size=file_path.stat().st_size,
                relative_path=file_path.name
            )
            
            # Translate file
            translated_content = await self.translator.translate_file(file_info, context)
            
            # Determine output path
            if output_path is None:
                output_path = file_path.parent / f"{file_path.stem}_zh{file_path.suffix}"
            
            # Save translation
            result = self.file_processor.save_translation(file_info, translated_content)
            
            if result.success:
                self.logger.info(f"Successfully translated: {file_path} -> {result.output_path}")
                return result.output_path
            else:
                self.logger.error(f"Failed to save translation: {result.error_message}")
                return None
                
        except Exception as e:
            self.logger.error(f"File translation failed: {e}")
            return None


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Translate markdown files using various LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate all markdown files in current directory
  python main.py translate .

  # Translate specific file
  python main.py translate-file README.md

  # Use specific model
  python main.py translate . --model qwen2.5:7b --provider ollama

  # Translate with custom output directory
  python main.py translate . --output-dir ./translations

  # Check service health
  python main.py health-check
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Translate directory command
    translate_parser = subparsers.add_parser('translate', help='Translate markdown files in directory')
    translate_parser.add_argument('directory', type=Path, help='Directory to scan for markdown files')
    translate_parser.add_argument('--output-dir', type=Path, help='Output directory for translated files')
    translate_parser.add_argument('--no-recursive', action='store_true', help='Do not scan subdirectories')
    translate_parser.add_argument('--no-skip-existing', action='store_true', help='Do not skip existing translations')
    translate_parser.add_argument('--context', type=str, help='Context for translation')
    
    # Translate file command
    translate_file_parser = subparsers.add_parser('translate-file', help='Translate a single markdown file')
    translate_file_parser.add_argument('file', type=Path, help='Markdown file to translate')
    translate_file_parser.add_argument('--output', type=Path, help='Output file path')
    translate_file_parser.add_argument('--context', type=str, help='Context for translation')
    
    # Health check command
    health_parser = subparsers.add_parser('health-check', help='Check service health')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available models')
    
    # Global options
    parser.add_argument('--config', type=Path, help='Configuration file path')
    parser.add_argument('--model', type=str, help='Model name to use')
    parser.add_argument('--provider', type=str, choices=['ollama', 'openai', 'claude', 'qwen'], 
                       help='LLM provider to use')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    return parser


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Load configuration
        config_path = args.config or get_default_config_path()
        app = TranslationApp(config_path)
        
        # Override settings from command line
        if args.model:
            app.settings.model_name = args.model
        if args.provider:
            app.settings.model_provider = args.provider
        if args.log_level:
            app.settings.logging.level = args.log_level
        if args.log_file:
            app.settings.logging.file = args.log_file
        
        # Reconfigure logging if needed
        if args.log_level or args.log_file:
            app.logger = setup_logging(
                level=app.settings.logging.level,
                log_file=app.settings.logging.file,
                log_format=app.settings.logging.format
            )
        
        # Execute command
        if args.command == 'translate':
            result = await app.translate_directory(
                directory=args.directory,
                output_dir=args.output_dir,
                recursive=not args.no_recursive,
                skip_existing=not args.no_skip_existing,
                context=args.context
            )
            print(f"Translated {len(result)} files")
            
        elif args.command == 'translate-file':
            result = await app.translate_file(
                file_path=args.file,
                output_path=args.output,
                context=args.context
            )
            if result:
                print(f"Translated file saved to: {result}")
            else:
                print("Translation failed")
                return 1
                
        elif args.command == 'health-check':
            is_healthy = await app.health_check()
            if is_healthy:
                print("All services are healthy")
                return 0
            else:
                print("Some services are not available")
                return 1
                
        elif args.command == 'list-models':
            try:
                model_configs = load_model_configs()
                providers = {}
                for config in model_configs.values():
                    if config.provider not in providers:
                        providers[config.provider] = []
                    providers[config.provider].append(config.name)
                
                print("Available models:")
                for provider, models in providers.items():
                    print(f"\n{provider.upper()}:")
                    for model in models:
                        print(f"  - {model}")
                        
            except Exception as e:
                print(f"Failed to load models: {e}")
                return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
