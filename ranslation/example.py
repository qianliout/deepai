#!/usr/bin/env python3
"""
Example usage of the markdown translation system.

This script demonstrates how to use the translation system programmatically.
"""

import asyncio
from pathlib import Path
from main import TranslationApp


async def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===")
    
    # Create app instance
    app = TranslationApp()
    
    # Check if services are healthy
    print("Checking service health...")
    is_healthy = await app.health_check()
    if not is_healthy:
        print("❌ Services are not healthy. Please check your configuration.")
        return
    
    print("✅ Services are healthy!")
    
    # Translate a single file (if it exists)
    sample_file = Path("sample.md")
    if sample_file.exists():
        print(f"Translating {sample_file}...")
        result = await app.translate_file(sample_file)
        if result:
            print(f"✅ Translation saved to: {result}")
        else:
            print("❌ Translation failed")
    else:
        print(f"Sample file {sample_file} not found, skipping single file translation")


async def example_directory_translation():
    """Directory translation example."""
    print("\n=== Directory Translation Example ===")
    
    # Create app instance
    app = TranslationApp()
    
    # Translate current directory
    current_dir = Path(".")
    print(f"Translating all markdown files in {current_dir.absolute()}...")
    
    try:
        results = await app.translate_directory(
            directory=current_dir,
            recursive=True,
            skip_existing=True
        )
        
        if results:
            print(f"✅ Successfully translated {len(results)} files:")
            for result in results:
                print(f"  - {result}")
        else:
            print("ℹ️  No files needed translation")
            
    except Exception as e:
        print(f"❌ Translation failed: {e}")


async def example_with_custom_config():
    """Example with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom config
    config_path = Path("custom_config.yaml")
    
    # This would create a custom config file
    # In practice, you would create this file manually
    if config_path.exists():
        app = TranslationApp(config_path)
        print(f"Using custom config: {config_path}")
    else:
        app = TranslationApp()
        print("Using default configuration")


async def example_error_handling():
    """Example of error handling."""
    print("\n=== Error Handling Example ===")
    
    app = TranslationApp()
    
    # Try to translate a non-existent file
    non_existent_file = Path("non_existent.md")
    print(f"Attempting to translate {non_existent_file}...")
    
    result = await app.translate_file(non_existent_file)
    if result is None:
        print("✅ Error handling worked correctly - file not found")
    else:
        print("❌ Unexpected success")


def create_sample_file():
    """Create a sample markdown file for testing."""
    sample_content = """# Sample Document

This is a sample markdown document for testing the translation system.

## Features

- **Bold text** and *italic text*
- `Code snippets` and `inline code`
- Lists and tables

## Code Block

```python
def hello_world():
    print("Hello, World!")
    return "success"
```

## Table

| Feature | Status | Priority |
|---------|--------|----------|
| Translation | ✅ | High |
| Error Handling | ✅ | High |
| Logging | ✅ | Medium |

## Conclusion

This document demonstrates various markdown features that should be preserved during translation.
"""
    
    sample_file = Path("sample.md")
    sample_file.write_text(sample_content, encoding='utf-8')
    print(f"Created sample file: {sample_file}")


async def main():
    """Main example function."""
    print("Markdown Translation System - Usage Examples")
    print("=" * 50)
    
    # Create a sample file for testing
    create_sample_file()
    
    try:
        # Run examples
        await example_basic_usage()
        await example_directory_translation()
        await example_with_custom_config()
        await example_error_handling()
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
