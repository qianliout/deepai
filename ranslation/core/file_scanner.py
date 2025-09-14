"""
File scanner for finding and filtering markdown files.
"""

import os
import logging
from pathlib import Path
from typing import List, Iterator, Optional
from dataclasses import dataclass


@dataclass
class MarkdownFile:
    """Represents a markdown file to be translated."""
    path: Path
    size: int
    relative_path: str
    
    def __str__(self) -> str:
        return f"{self.relative_path} ({self.size} bytes)"


class FileScanner:
    """Scans directories for markdown files to translate."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.supported_extensions = {'.md', '.markdown'}
        self.max_file_size = 100 * 1024  # 100KB limit
    
    def scan_directory(self, directory: Path, recursive: bool = True) -> List[MarkdownFile]:
        """
        Scan directory for markdown files.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            
        Returns:
            List of MarkdownFile objects
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")
        
        self.logger.info(f"Scanning directory: {directory}")
        
        markdown_files = []
        
        try:
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for file_path in directory.glob(pattern):
                if self._is_markdown_file(file_path):
                    try:
                        file_size = file_path.stat().st_size
                        relative_path = file_path.relative_to(directory)
                        
                        if file_size > self.max_file_size:
                            self.logger.warning(
                                f"File too large, skipping: {relative_path} ({file_size} bytes)"
                            )
                            continue
                        
                        markdown_file = MarkdownFile(
                            path=file_path,
                            size=file_size,
                            relative_path=str(relative_path)
                        )
                        markdown_files.append(markdown_file)
                        self.logger.debug(f"Found markdown file: {markdown_file}")
                        
                    except OSError as e:
                        self.logger.error(f"Error accessing file {file_path}: {e}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Error scanning directory {directory}: {e}")
            raise
        
        self.logger.info(f"Found {len(markdown_files)} markdown files")
        return markdown_files
    
    def scan_git_repo(self, repo_path: Path) -> List[MarkdownFile]:
        """
        Scan git repository for markdown files, excluding .git directory.
        
        Args:
            repo_path: Path to git repository
            
        Returns:
            List of MarkdownFile objects
        """
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")
        
        self.logger.info(f"Scanning git repository: {repo_path}")
        
        markdown_files = []
        
        try:
            for file_path in repo_path.rglob("*"):
                # Skip .git directory and other hidden directories
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                
                if self._is_markdown_file(file_path):
                    try:
                        file_size = file_path.stat().st_size
                        relative_path = file_path.relative_to(repo_path)
                        
                        if file_size > self.max_file_size:
                            self.logger.warning(
                                f"File too large, skipping: {relative_path} ({file_size} bytes)"
                            )
                            continue
                        
                        markdown_file = MarkdownFile(
                            path=file_path,
                            size=file_size,
                            relative_path=str(relative_path)
                        )
                        markdown_files.append(markdown_file)
                        self.logger.debug(f"Found markdown file: {markdown_file}")
                        
                    except OSError as e:
                        self.logger.error(f"Error accessing file {file_path}: {e}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Error scanning git repository {repo_path}: {e}")
            raise
        
        self.logger.info(f"Found {len(markdown_files)} markdown files in repository")
        return markdown_files
    
    def _is_markdown_file(self, file_path: Path) -> bool:
        """Check if file is a markdown file."""
        if not file_path.is_file():
            return False
        
        return file_path.suffix.lower() in self.supported_extensions
    
    def filter_existing_translations(self, markdown_files: List[MarkdownFile]) -> List[MarkdownFile]:
        """
        Filter out files that already have translations.
        
        Args:
            markdown_files: List of markdown files
            
        Returns:
            List of files without existing translations
        """
        filtered_files = []
        
        for file_info in markdown_files:
            # Check if _zh version already exists
            zh_path = file_info.path.parent / f"{file_info.path.stem}_zh{file_info.path.suffix}"
            
            if zh_path.exists():
                self.logger.debug(f"Translation already exists, skipping: {file_info.relative_path}")
                continue
            
            filtered_files.append(file_info)
        
        self.logger.info(f"Filtered to {len(filtered_files)} files without existing translations")
        return filtered_files
    
    def get_file_stats(self, markdown_files: List[MarkdownFile]) -> dict:
        """Get statistics about the markdown files."""
        if not markdown_files:
            return {
                "total_files": 0,
                "total_size": 0,
                "average_size": 0,
                "largest_file": None,
                "smallest_file": None
            }
        
        total_size = sum(f.size for f in markdown_files)
        sizes = [f.size for f in markdown_files]
        
        return {
            "total_files": len(markdown_files),
            "total_size": total_size,
            "average_size": total_size // len(markdown_files),
            "largest_file": max(markdown_files, key=lambda f: f.size),
            "smallest_file": min(markdown_files, key=lambda f: f.size)
        }
