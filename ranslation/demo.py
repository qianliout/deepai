#!/usr/bin/env python3
"""
演示脚本：展示Markdown翻译系统的功能

这个脚本演示了如何使用翻译系统来翻译Markdown文件。
"""

import asyncio
from pathlib import Path
from main import TranslationApp


async def demo_single_file_translation():
    """演示单文件翻译功能"""
    print("🎯 演示1：单文件翻译")
    print("=" * 50)
    
    # 创建测试文件
    test_content = """# Welcome to Our Project

This is a sample markdown document for demonstration.

## Features

- **Bold text** and *italic text*
- `Code snippets` and `inline code`
- Lists and tables

## Code Example

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

This document demonstrates various markdown features.
"""
    
    # 创建测试文件
    test_file = Path("demo_test.md")
    test_file.write_text(test_content, encoding='utf-8')
    print(f"✅ 创建测试文件: {test_file}")
    
    # 翻译文件
    app = TranslationApp()
    app.settings.model_name = "qwen3:8b"  # 使用正确的模型
    result = await app.translate_file(test_file)
    
    if result:
        print(f"✅ 翻译成功: {result}")
        
        # 显示翻译结果的前几行
        with open(result, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]
            print("\n📄 翻译结果预览:")
            print("-" * 30)
            for line in lines:
                print(line.rstrip())
            print("...")
    else:
        print("❌ 翻译失败")
    
    # 清理测试文件
    test_file.unlink()
    if result:
        result.unlink()
    
    print()


async def demo_directory_translation():
    """演示目录翻译功能"""
    print("🎯 演示2：目录翻译")
    print("=" * 50)
    
    # 创建测试目录和文件
    test_dir = Path("demo_test_dir")
    test_dir.mkdir(exist_ok=True)
    
    files_content = {
        "README.md": """# Project Documentation

This is the main documentation file.

## Getting Started

Follow these steps to get started:

1. Install dependencies
2. Run the application
3. Configure settings

## API Reference

See the API documentation for more details.
""",
        "CHANGELOG.md": """# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2024-01-01

### Added
- Initial release
- Basic functionality
- Documentation

### Changed
- Updated dependencies
- Improved performance
""",
        "CONTRIBUTING.md": """# Contributing

Thank you for your interest in contributing to this project!

## How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Code Style

Please follow the existing code style and conventions.
"""
    }
    
    # 创建测试文件
    for filename, content in files_content.items():
        file_path = test_dir / filename
        file_path.write_text(content, encoding='utf-8')
        print(f"✅ 创建文件: {file_path}")
    
    # 翻译目录
    app = TranslationApp()
    app.settings.model_name = "qwen3:8b"  # 使用正确的模型
    results = await app.translate_directory(test_dir, recursive=False)
    
    print(f"\n✅ 翻译完成: {len(results)} 个文件")
    for result in results:
        print(f"  - {result}")
    
    # 显示一个翻译结果
    if results:
        print(f"\n📄 翻译结果预览 ({results[0].name}):")
        print("-" * 30)
        with open(results[0], 'r', encoding='utf-8') as f:
            lines = f.readlines()[:8]
            for line in lines:
                print(line.rstrip())
        print("...")
    
    # 清理测试目录
    import shutil
    shutil.rmtree(test_dir)
    
    print()


async def demo_health_check():
    """演示健康检查功能"""
    print("🎯 演示3：健康检查")
    print("=" * 50)
    
    app = TranslationApp()
    is_healthy = await app.health_check()
    
    if is_healthy:
        print("✅ 所有服务正常运行")
    else:
        print("❌ 服务不可用")
    
    print()


async def demo_configuration():
    """演示配置功能"""
    print("🎯 演示4：配置信息")
    print("=" * 50)
    
    app = TranslationApp()
    
    print(f"📋 当前配置:")
    print(f"  - 模型提供商: {app.settings.model_provider}")
    print(f"  - 模型名称: {app.settings.model_name}")
    print(f"  - 最大文件大小: {app.settings.translation.max_file_size} 字节")
    print(f"  - 最大并发数: {app.settings.translation.max_concurrent}")
    print(f"  - 温度参数: {app.settings.translation.temperature}")
    
    print()


async def main():
    """主演示函数"""
    print("🚀 Markdown翻译系统演示")
    print("=" * 60)
    print()
    
    try:
        # 运行所有演示
        await demo_health_check()
        await demo_configuration()
        await demo_single_file_translation()
        await demo_directory_translation()
        
        print("🎉 演示完成！")
        print("\n💡 使用提示:")
        print("  - 翻译单个文件: python main.py translate-file <文件路径>")
        print("  - 翻译目录: python main.py translate <目录路径>")
        print("  - 健康检查: python main.py health-check")
        print("  - 查看帮助: python main.py --help")
        
    except KeyboardInterrupt:
        print("\n⚠️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
