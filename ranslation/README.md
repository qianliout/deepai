# Markdown Translation System

一个基于大语言模型的Markdown文件翻译系统，支持多种LLM提供商，包括Ollama、OpenAI、Claude和通义千问。

## 功能特性

- 🔄 支持多种LLM提供商（Ollama、OpenAI、Claude、通义千问）
- 📁 批量翻译目录中的所有Markdown文件
- 🎯 单文件翻译支持
- 🔍 智能文件扫描和过滤
- 📊 详细的翻译统计和日志
- ⚡ 并发翻译提高效率
- 🛡️ 完善的错误处理和重试机制
- 🔧 灵活的配置管理

## 安装

1. 克隆或下载项目到本地
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 配置环境变量（可选）：

```bash
export OPENAI_API_KEY="your-openai-key"
export CLAUDE_API_KEY="your-claude-key"
export QWEN_API_KEY="your-qwen-key"
```

## 快速开始

### 使用Ollama（推荐）

1. 安装并启动Ollama：
```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下载Qwen模型
ollama pull qwen2.5:7b

# 启动Ollama服务
ollama serve
```

2. 翻译当前目录的所有Markdown文件：
```bash
python main.py translate .
```

### 使用其他LLM提供商

```bash
# 使用OpenAI
python main.py translate . --provider openai --model gpt-3.5-turbo

# 使用Claude
python main.py translate . --provider claude --model claude-3-sonnet-20240229

# 使用通义千问
python main.py translate . --provider qwen --model qwen-turbo
```

## 命令行使用

### 基本命令

```bash
# 翻译目录中的所有Markdown文件
python main.py translate <目录路径>

# 翻译单个文件
python main.py translate-file <文件路径>

# 检查服务健康状态
python main.py health-check

# 列出可用模型
python main.py list-models
```

### 高级选项

```bash
# 指定输出目录
python main.py translate . --output-dir ./translations

# 不扫描子目录
python main.py translate . --no-recursive

# 不跳过已存在的翻译
python main.py translate . --no-skip-existing

# 添加翻译上下文
python main.py translate . --context "技术文档翻译"

# 指定配置文件
python main.py translate . --config ./my_config.yaml

# 设置日志级别
python main.py translate . --log-level DEBUG

# 保存日志到文件
python main.py translate . --log-file ./translation.log
```

## 配置文件

系统支持YAML配置文件来自定义设置。创建 `config/translation.yaml`：

```yaml
model:
  provider: "ollama"
  name: "qwen2.5:7b"

api:
  ollama_url: "http://localhost:11434"
  openai_api_key: "your-key"
  claude_api_key: "your-key"
  qwen_api_key: "your-key"

translation:
  max_file_size: 102400  # 100KB
  max_concurrent: 3
  temperature: 0.3
  max_tokens: 4000

file:
  supported_extensions: [".md", ".markdown"]
  create_backup: true
  validate_translation: true
  output_suffix: "_zh"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/translation.log"
  max_size: 10485760  # 10MB
  backup_count: 5
```

## 支持的模型

### Ollama模型
- qwen2.5:7b
- qwen2.5:14b
- llama3.1:8b

### OpenAI模型
- gpt-4
- gpt-3.5-turbo

### Claude模型
- claude-3-sonnet-20240229
- claude-3-haiku-20240307

### 通义千问模型
- qwen-turbo
- qwen-plus

## 项目结构

```
ranslation/
├── main.py                    # 主程序入口
├── config/                    # 配置管理
│   ├── settings.py
│   ├── models.py
│   └── models.yaml
├── llm/                       # LLM客户端
│   ├── base.py
│   ├── ollama_client.py
│   ├── openai_client.py
│   ├── claude_client.py
│   └── qwen_client.py
├── core/                      # 核心功能
│   ├── file_scanner.py
│   ├── translator.py
│   └── file_processor.py
├── utils/                     # 工具模块
│   ├── logger_config.py
│   └── exceptions.py
├── tests/                     # 测试套件
├── requirements.txt
└── README.md
```

## 使用示例

### 翻译Git仓库

```bash
# 翻译整个Git仓库的文档
python main.py translate /path/to/your/repo

# 只翻译根目录的文档
python main.py translate /path/to/your/repo --no-recursive
```

### 翻译单个文件

```bash
# 翻译README文件
python main.py translate-file README.md

# 指定输出文件名
python main.py translate-file README.md --output README_zh.md
```

### 批量翻译多个目录

```bash
# 翻译多个目录
for dir in docs/ api-docs/ user-guide/; do
    python main.py translate "$dir" --output-dir "translations/$dir"
done
```

## 注意事项

1. **文件大小限制**：默认最大文件大小为100KB，可在配置中调整
2. **并发控制**：默认最大并发数为3，避免过度占用资源
3. **翻译质量**：建议使用较大的模型（如qwen2.5:14b）获得更好的翻译质量
4. **网络连接**：使用远程API时需要稳定的网络连接
5. **API密钥**：使用付费API时请妥善保管API密钥

## 故障排除

### 常见问题

1. **Ollama连接失败**
   - 确保Ollama服务正在运行：`ollama serve`
   - 检查端口是否正确：默认11434

2. **API密钥错误**
   - 检查环境变量是否正确设置
   - 确认API密钥有效且有足够额度

3. **翻译质量不佳**
   - 尝试使用更大的模型
   - 调整temperature参数
   - 添加更多上下文信息

4. **文件过大**
   - 检查文件大小是否超过限制
   - 考虑分割大文件

### 日志分析

查看详细日志：
```bash
python main.py translate . --log-level DEBUG --log-file debug.log
```

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

MIT License
