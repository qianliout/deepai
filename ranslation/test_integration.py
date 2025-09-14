#!/usr/bin/env python3
"""
Integration test for the markdown translation system.

This script performs basic integration tests to verify the system works correctly.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from main import TranslationApp


async def test_basic_functionality():
    """Test basic functionality without actual LLM calls."""
    print("🧪 Testing basic functionality...")
    
    try:
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test markdown files
            test_files = {
                "simple.md": """# Simple Document

This is a simple test document.

## Features

- Feature 1
- Feature 2

## Code

```python
print("Hello, World!")
```
""",
                "complex.md": """# Complex Document

This is a more complex test document with various markdown features.

## Table

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

## Math

Inline math: $E = mc^2$

Block math:
$$
\\sum_{i=1}^{n} x_i = x_1 + x_2 + \\cdots + x_n
$$

## Links and Images

[Link to example](https://example.com)

![Example image](https://example.com/image.png)

## Conclusion

This document tests various markdown features.
"""
            }
            
            # Write test files
            for filename, content in test_files.items():
                file_path = temp_path / filename
                file_path.write_text(content, encoding='utf-8')
                print(f"  ✅ Created {filename}")
            
            # Test file scanner
            print("  🔍 Testing file scanner...")
            app = TranslationApp()
            files = app.file_scanner.scan_directory(temp_path)
            
            assert len(files) == 2, f"Expected 2 files, found {len(files)}"
            print(f"  ✅ Found {len(files)} markdown files")
            
            # Test file statistics
            stats = app.file_scanner.get_file_stats(files)
            assert stats["total_files"] == 2
            assert stats["total_size"] > 0
            print(f"  ✅ File statistics: {stats['total_files']} files, {stats['total_size']} bytes")
            
            # Test filtering existing translations
            filtered_files = app.file_scanner.filter_existing_translations(files)
            assert len(filtered_files) == 2, "Should not filter out any files"
            print("  ✅ File filtering works correctly")
            
        print("✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_configuration_loading():
    """Test configuration loading."""
    print("🧪 Testing configuration loading...")
    
    try:
        # Test default configuration
        app = TranslationApp()
        assert app.settings.model_provider == "ollama"
        assert app.settings.model_name == "qwen2.5:7b"
        print("  ✅ Default configuration loaded correctly")
        
        # Test configuration file loading
        config_path = Path("config/translation.yaml")
        if config_path.exists():
            app_with_config = TranslationApp(config_path)
            print("  ✅ Configuration file loaded successfully")
        else:
            print("  ⚠️  Configuration file not found, using defaults")
        
        print("✅ Configuration loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False


async def test_llm_client_creation():
    """Test LLM client creation (without actual API calls)."""
    print("🧪 Testing LLM client creation...")
    
    try:
        app = TranslationApp()
        
        # Test Ollama client creation
        app.settings.model_provider = "ollama"
        app.settings.model_name = "qwen2.5:7b"
        app._create_llm_client()
        
        assert app.llm_client is not None
        assert app.llm_client.model == "qwen2.5:7b"
        print("  ✅ Ollama client created successfully")
        
        # Test translator creation
        assert app.translator is not None
        print("  ✅ Translator created successfully")
        
        print("✅ LLM client creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ LLM client creation test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling."""
    print("🧪 Testing error handling...")
    
    try:
        app = TranslationApp()
        
        # Test with invalid model provider
        app.settings.model_provider = "invalid_provider"
        
        try:
            app._create_llm_client()
            print("  ❌ Should have raised ConfigurationError")
            return False
        except Exception as e:
            if "Unsupported model provider" in str(e):
                print("  ✅ Correctly handled invalid model provider")
            else:
                raise
        
        # Test with missing API key for paid services
        app.settings.model_provider = "openai"
        app.settings.openai_api_key = None
        
        try:
            app._create_llm_client()
            print("  ❌ Should have raised ConfigurationError for missing API key")
            return False
        except Exception as e:
            if "API key not provided" in str(e):
                print("  ✅ Correctly handled missing API key")
            else:
                raise
        
        print("✅ Error handling test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


async def test_file_processing():
    """Test file processing without actual translation."""
    print("🧪 Testing file processing...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test file
            test_file = temp_path / "test.md"
            test_content = "# Test Document\n\nThis is a test."
            test_file.write_text(test_content, encoding='utf-8')
            
            # Create MarkdownFile object
            from core.file_scanner import MarkdownFile
            file_info = MarkdownFile(
                path=test_file,
                size=test_file.stat().st_size,
                relative_path="test.md"
            )
            
            # Test file processor
            app = TranslationApp()
            translated_content = "翻译后的内容"
            
            result = app.file_processor.save_translation(
                file_info, 
                translated_content, 
                temp_path
            )
            
            assert result.success, f"File processing failed: {result.error_message}"
            assert result.output_path is not None
            assert result.output_path.exists()
            
            # Verify content
            saved_content = result.output_path.read_text(encoding='utf-8')
            assert saved_content == translated_content
            
            print("  ✅ File processing works correctly")
            
            # Test validation
            is_valid = app.file_processor.validate_translation(test_file, result.output_path)
            print(f"  ✅ File validation: {'passed' if is_valid else 'failed'}")
        
        print("✅ File processing test passed!")
        return True
        
    except Exception as e:
        print(f"❌ File processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all integration tests."""
    print("🚀 Starting Integration Tests")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_configuration_loading,
        test_llm_client_creation,
        test_error_handling,
        test_file_processing,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        
        print()  # Empty line between tests
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed!")
        return False


async def main():
    """Main test function."""
    try:
        success = await run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
