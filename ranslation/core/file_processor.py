"""
File processor for handling markdown file operations.
"""

import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from core.file_scanner import MarkdownFile


@dataclass
class ProcessResult:
    """Result of file processing operation."""
    file_info: MarkdownFile
    success: bool
    output_path: Optional[Path] = None
    error_message: Optional[str] = None


class FileProcessor:
    """Handles file I/O operations for translation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.processing_stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0
        }
    
    def save_translation(
        self, 
        file_info: MarkdownFile, 
        translated_content: str,
        output_dir: Optional[Path] = None
    ) -> ProcessResult:
        """
        Save translated content to a new file.
        
        Args:
            file_info: Original file information
            translated_content: Translated content
            output_dir: Optional output directory (defaults to same as original)
            
        Returns:
            ProcessResult with operation status
        """
        self.logger.info(f"Saving translation for: {file_info.relative_path}")
        
        try:
            # Determine output path
            if output_dir:
                output_path = output_dir / f"{file_info.path.stem}_zh{file_info.path.suffix}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_path = file_info.path.parent / f"{file_info.path.stem}_zh{file_info.path.suffix}"
            
            # Check if output file already exists
            if output_path.exists():
                self.logger.warning(f"Output file already exists: {output_path}")
                self.processing_stats["skipped"] += 1
                return ProcessResult(
                    file_info=file_info,
                    success=False,
                    output_path=output_path,
                    error_message="Output file already exists"
                )
            
            # Write translated content
            output_path.write_text(translated_content, encoding='utf-8')
            
            # Verify file was written correctly
            if not output_path.exists():
                raise IOError("Output file was not created")
            
            file_size = output_path.stat().st_size
            self.logger.info(
                f"Successfully saved translation: {output_path} ({file_size} bytes)"
            )
            
            # Update statistics
            self.processing_stats["total_files"] += 1
            self.processing_stats["successful"] += 1
            
            return ProcessResult(
                file_info=file_info,
                success=True,
                output_path=output_path
            )
            
        except Exception as e:
            error_msg = f"Failed to save translation for {file_info.relative_path}: {e}"
            self.logger.error(error_msg)
            
            # Update statistics
            self.processing_stats["total_files"] += 1
            self.processing_stats["failed"] += 1
            
            return ProcessResult(
                file_info=file_info,
                success=False,
                error_message=error_msg
            )
    
    def process_translations(
        self, 
        translations: List[tuple[MarkdownFile, str]], 
        output_dir: Optional[Path] = None
    ) -> List[ProcessResult]:
        """
        Process multiple translations and save them to files.
        
        Args:
            translations: List of (MarkdownFile, translated_content) tuples
            output_dir: Optional output directory
            
        Returns:
            List of ProcessResult objects
        """
        self.logger.info(f"Processing {len(translations)} translations")
        
        results = []
        
        for file_info, translated_content in translations:
            result = self.save_translation(file_info, translated_content, output_dir)
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success and not r.error_message == "Output file already exists")
        skipped = sum(1 for r in results if not r.success and r.error_message == "Output file already exists")
        
        self.logger.info(
            f"Processing completed: {successful} successful, {failed} failed, {skipped} skipped"
        )
        
        return results
    
    def create_backup(self, file_path: Path) -> Optional[Path]:
        """
        Create a backup of the original file.
        
        Args:
            file_path: Path to file to backup
            
        Returns:
            Path to backup file, or None if backup failed
        """
        try:
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
            
            if backup_path.exists():
                self.logger.debug(f"Backup already exists: {backup_path}")
                return backup_path
            
            # Copy file content
            content = file_path.read_text(encoding='utf-8')
            backup_path.write_text(content, encoding='utf-8')
            
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup for {file_path}: {e}")
            return None
    
    def validate_translation(self, original_path: Path, translated_path: Path) -> bool:
        """
        Validate that translation file was created correctly.
        
        Args:
            original_path: Path to original file
            translated_path: Path to translated file
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            if not translated_path.exists():
                self.logger.error(f"Translated file does not exist: {translated_path}")
                return False
            
            # Check file size (should be reasonable)
            original_size = original_path.stat().st_size
            translated_size = translated_path.stat().st_size
            
            # Translated file should not be empty and not extremely large
            if translated_size == 0:
                self.logger.error(f"Translated file is empty: {translated_path}")
                return False
            
            if translated_size > original_size * 3:  # Allow up to 3x original size
                self.logger.warning(
                    f"Translated file is unusually large: {translated_path} "
                    f"({translated_size} vs {original_size} bytes)"
                )
            
            # Check that file contains some Chinese characters
            content = translated_path.read_text(encoding='utf-8')
            chinese_chars = sum(1 for char in content if '\u4e00' <= char <= '\u9fff')
            
            if chinese_chars < 10:  # Should have at least 10 Chinese characters
                self.logger.warning(
                    f"Translated file has few Chinese characters: {translated_path} "
                    f"({chinese_chars} characters)"
                )
            
            self.logger.debug(f"Translation validation passed: {translated_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Translation validation failed for {translated_path}: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        return self.processing_stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0
        }
