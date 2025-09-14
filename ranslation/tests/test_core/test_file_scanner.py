"""
Tests for file scanner module.
"""

import pytest
from pathlib import Path
from core.file_scanner import FileScanner, MarkdownFile


class TestMarkdownFile:
    """Test MarkdownFile dataclass."""
    
    def test_markdown_file_creation(self, sample_markdown_file):
        """Test creating MarkdownFile."""
        file_info = MarkdownFile(
            path=sample_markdown_file,
            size=sample_markdown_file.stat().st_size,
            relative_path="sample.md"
        )
        
        assert file_info.path == sample_markdown_file
        assert file_info.size > 0
        assert file_info.relative_path == "sample.md"
    
    def test_markdown_file_str_representation(self, sample_markdown_file):
        """Test string representation of MarkdownFile."""
        file_info = MarkdownFile(
            path=sample_markdown_file,
            size=1024,
            relative_path="sample.md"
        )
        
        str_repr = str(file_info)
        assert "sample.md" in str_repr
        assert "1024" in str_repr


class TestFileScanner:
    """Test FileScanner class."""
    
    def test_file_scanner_creation(self):
        """Test creating FileScanner."""
        scanner = FileScanner()
        
        assert scanner.supported_extensions == {'.md', '.markdown'}
        assert scanner.max_file_size == 100 * 1024
    
    def test_is_markdown_file_valid(self, sample_markdown_file):
        """Test identifying valid markdown files."""
        scanner = FileScanner()
        
        assert scanner._is_markdown_file(sample_markdown_file) is True
    
    def test_is_markdown_file_invalid(self, temp_dir):
        """Test identifying invalid files."""
        scanner = FileScanner()
        
        # Test non-markdown file
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("Hello, World!")
        assert scanner._is_markdown_file(txt_file) is False
        
        # Test directory
        assert scanner._is_markdown_file(temp_dir) is False
        
        # Test non-existent file
        non_existent = temp_dir / "nonexistent.md"
        assert scanner._is_markdown_file(non_existent) is False
    
    def test_scan_directory_success(self, temp_dir, sample_markdown_content):
        """Test successful directory scanning."""
        # Create test files
        (temp_dir / "test1.md").write_text(sample_markdown_content)
        (temp_dir / "test2.markdown").write_text(sample_markdown_content)
        (temp_dir / "test3.txt").write_text("Not markdown")
        
        scanner = FileScanner()
        files = scanner.scan_directory(temp_dir, recursive=False)
        
        assert len(files) == 2
        file_names = [f.relative_path for f in files]
        assert "test1.md" in file_names
        assert "test2.markdown" in file_names
        assert "test3.txt" not in file_names
    
    def test_scan_directory_recursive(self, temp_dir, sample_markdown_content):
        """Test recursive directory scanning."""
        # Create nested structure
        (temp_dir / "level1.md").write_text(sample_markdown_content)
        (temp_dir / "subdir" / "level2.md").mkdir(parents=True)
        (temp_dir / "subdir" / "level2.md").write_text(sample_markdown_content)
        (temp_dir / "subdir" / "subsubdir" / "level3.md").mkdir(parents=True)
        (temp_dir / "subdir" / "subsubdir" / "level3.md").write_text(sample_markdown_content)
        
        scanner = FileScanner()
        files = scanner.scan_directory(temp_dir, recursive=True)
        
        assert len(files) == 3
        file_names = [f.relative_path for f in files]
        assert "level1.md" in file_names
        assert "subdir/level2.md" in file_names
        assert "subdir/subsubdir/level3.md" in file_names
    
    def test_scan_directory_nonexistent(self):
        """Test scanning non-existent directory."""
        scanner = FileScanner()
        
        with pytest.raises(FileNotFoundError):
            scanner.scan_directory(Path("/nonexistent/directory"))
    
    def test_scan_directory_not_directory(self, sample_markdown_file):
        """Test scanning file instead of directory."""
        scanner = FileScanner()
        
        with pytest.raises(NotADirectoryError):
            scanner.scan_directory(sample_markdown_file)
    
    def test_scan_git_repo(self, git_repo_structure):
        """Test scanning git repository."""
        scanner = FileScanner()
        files = scanner.scan_git_repo(git_repo_structure)
        
        # Should find markdown files but skip .git directory
        assert len(files) == 2
        file_names = [f.relative_path for f in files]
        assert "README.md" in file_names
        assert "docs/guide.md" in file_names
        assert "src/main.py" not in file_names  # Not markdown
    
    def test_scan_git_repo_nonexistent(self):
        """Test scanning non-existent git repository."""
        scanner = FileScanner()
        
        with pytest.raises(FileNotFoundError):
            scanner.scan_git_repo(Path("/nonexistent/repo"))
    
    def test_filter_existing_translations(self, temp_dir, sample_markdown_content):
        """Test filtering files with existing translations."""
        # Create original file
        original_file = temp_dir / "test.md"
        original_file.write_text(sample_markdown_content)
        
        # Create translation file
        translation_file = temp_dir / "test_zh.md"
        translation_file.write_text("Translated content")
        
        scanner = FileScanner()
        
        # Create MarkdownFile objects
        file_info = MarkdownFile(
            path=original_file,
            size=original_file.stat().st_size,
            relative_path="test.md"
        )
        
        files = [file_info]
        filtered_files = scanner.filter_existing_translations(files)
        
        # Should be filtered out because translation exists
        assert len(filtered_files) == 0
    
    def test_filter_existing_translations_no_existing(self, temp_dir, sample_markdown_content):
        """Test filtering when no translations exist."""
        # Create original file only
        original_file = temp_dir / "test.md"
        original_file.write_text(sample_markdown_content)
        
        scanner = FileScanner()
        
        file_info = MarkdownFile(
            path=original_file,
            size=original_file.stat().st_size,
            relative_path="test.md"
        )
        
        files = [file_info]
        filtered_files = scanner.filter_existing_translations(files)
        
        # Should not be filtered out
        assert len(filtered_files) == 1
        assert filtered_files[0] == file_info
    
    def test_get_file_stats(self, temp_dir, sample_markdown_content):
        """Test getting file statistics."""
        # Create test files with different sizes
        file1 = temp_dir / "small.md"
        file1.write_text("Small content")
        
        file2 = temp_dir / "large.md"
        file2.write_text(sample_markdown_content * 10)  # Larger content
        
        scanner = FileScanner()
        
        file_info1 = MarkdownFile(
            path=file1,
            size=file1.stat().st_size,
            relative_path="small.md"
        )
        
        file_info2 = MarkdownFile(
            path=file2,
            size=file2.stat().st_size,
            relative_path="large.md"
        )
        
        files = [file_info1, file_info2]
        stats = scanner.get_file_stats(files)
        
        assert stats["total_files"] == 2
        assert stats["total_size"] == file_info1.size + file_info2.size
        assert stats["average_size"] == (file_info1.size + file_info2.size) // 2
        assert stats["largest_file"] == file_info2
        assert stats["smallest_file"] == file_info1
    
    def test_get_file_stats_empty(self):
        """Test getting stats for empty file list."""
        scanner = FileScanner()
        stats = scanner.get_file_stats([])
        
        assert stats["total_files"] == 0
        assert stats["total_size"] == 0
        assert stats["average_size"] == 0
        assert stats["largest_file"] is None
        assert stats["smallest_file"] is None
    
    def test_large_file_filtering(self, temp_dir, large_markdown_content):
        """Test filtering out large files."""
        # Create a large file
        large_file = temp_dir / "large.md"
        large_file.write_text(large_markdown_content)
        
        scanner = FileScanner()
        files = scanner.scan_directory(temp_dir)
        
        # Large file should be filtered out
        assert len(files) == 0
