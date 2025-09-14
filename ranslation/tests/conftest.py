"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Generator, AsyncGenerator

from llm.base import BaseLLMClient, LLMResponse, LLMProvider, TranslationRequest
from core.file_scanner import MarkdownFile


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_markdown_content() -> str:
    """Sample markdown content for testing."""
    return """# Sample Document

This is a sample markdown document for testing translation.

## Features

- Feature 1
- Feature 2
- Feature 3

## Code Example

```python
def hello_world():
    print("Hello, World!")
```

## Conclusion

This document demonstrates various markdown features.
"""


@pytest.fixture
def sample_markdown_file(temp_dir: Path, sample_markdown_content: str) -> Path:
    """Create a sample markdown file for testing."""
    file_path = temp_dir / "sample.md"
    file_path.write_text(sample_markdown_content, encoding='utf-8')
    return file_path


@pytest.fixture
def markdown_file_info(sample_markdown_file: Path) -> MarkdownFile:
    """Create a MarkdownFile object for testing."""
    return MarkdownFile(
        path=sample_markdown_file,
        size=sample_markdown_file.stat().st_size,
        relative_path="sample.md"
    )


@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """Mock LLM response for testing."""
    return LLMResponse(
        content="# 示例文档\n\n这是一个用于测试翻译的示例markdown文档。\n\n## 功能\n\n- 功能 1\n- 功能 2\n- 功能 3",
        model="test-model",
        provider=LLMProvider.OLLAMA,
        usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50}
    )


@pytest.fixture
def mock_llm_client(mock_llm_response: LLMResponse) -> Mock:
    """Mock LLM client for testing."""
    client = Mock(spec=BaseLLMClient)
    client.model = "test-model"
    client.get_provider.return_value = LLMProvider.OLLAMA
    client.health_check = AsyncMock(return_value=True)
    client.translate = AsyncMock(return_value=mock_llm_response)
    return client


@pytest.fixture
def mock_llm_client_failing() -> Mock:
    """Mock LLM client that fails for testing error handling."""
    client = Mock(spec=BaseLLMClient)
    client.model = "test-model"
    client.get_provider.return_value = LLMProvider.OLLAMA
    client.health_check = AsyncMock(return_value=False)
    client.translate = AsyncMock(side_effect=Exception("Translation failed"))
    return client


@pytest.fixture
def sample_translation_request() -> TranslationRequest:
    """Sample translation request for testing."""
    return TranslationRequest(
        text="Hello, World!",
        source_lang="en",
        target_lang="zh",
        temperature=0.3
    )


@pytest.fixture
def large_markdown_content() -> str:
    """Large markdown content for testing (simulates 50KB file)."""
    base_content = """# Large Document

This is a large markdown document for testing translation performance.

## Section 1

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

"""
    
    # Repeat content to create a large file
    content = base_content
    for i in range(100):  # This should create a file around 50KB
        content += f"""
## Section {i + 2}

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

### Subsection {i + 1}.1

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.

### Subsection {i + 1}.2

Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt.

"""
    
    return content


@pytest.fixture
def large_markdown_file(temp_dir: Path, large_markdown_content: str) -> Path:
    """Create a large markdown file for testing."""
    file_path = temp_dir / "large_sample.md"
    file_path.write_text(large_markdown_content, encoding='utf-8')
    return file_path


@pytest.fixture
def git_repo_structure(temp_dir: Path) -> Path:
    """Create a mock git repository structure for testing."""
    # Create .git directory
    git_dir = temp_dir / ".git"
    git_dir.mkdir()
    
    # Create some markdown files
    (temp_dir / "README.md").write_text("# Project README\n\nThis is a project README.", encoding='utf-8')
    (temp_dir / "docs" / "guide.md").mkdir(parents=True)
    (temp_dir / "docs" / "guide.md").write_text("# User Guide\n\nThis is a user guide.", encoding='utf-8')
    (temp_dir / "src" / "main.py").mkdir(parents=True)
    (temp_dir / "src" / "main.py").write_text("print('Hello, World!')", encoding='utf-8')
    
    return temp_dir


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "llm: Tests requiring LLM API")
