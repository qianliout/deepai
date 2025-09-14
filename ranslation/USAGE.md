# 使用说明

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动Ollama服务
```bash
# 确保Ollama已安装并启动
ollama serve

# 下载模型（如果还没有）
ollama pull qwen3:8b
```

### 3. 翻译文件

#### 翻译单个文件
```bash
python main.py translate-file README.md
```

#### 翻译整个目录
```bash
python main.py translate .
```

#### 翻译指定目录
```bash
python main.py translate /path/to/your/docs
```

## 常用命令

### 健康检查
```bash
python main.py health-check
```

### 列出可用模型
```bash
python main.py list-models
```

### 查看帮助
```bash
python main.py --help
```

## 配置选项

### 使用不同模型
```bash
# 使用qwen3:8b（默认）
python main.py translate . --model qwen3:8b

# 使用其他Ollama模型
python main.py translate . --model llama3.1:8b
```

### 指定输出目录
```bash
python main.py translate . --output-dir ./translations
```

### 不扫描子目录
```bash
python main.py translate . --no-recursive
```

### 不跳过已存在的翻译
```bash
python main.py translate . --no-skip-existing
```

### 添加翻译上下文
```bash
python main.py translate . --context "技术文档翻译"
```

## 配置文件

编辑 `config/translation.yaml` 来自定义设置：

```yaml
model:
  provider: "ollama"
  name: "qwen3:8b"

translation:
  max_file_size: 102400  # 100KB
  max_concurrent: 3
  temperature: 0.3
```

## 注意事项

1. **文件大小限制**：默认最大100KB，可在配置中调整
2. **并发控制**：默认最多3个并发翻译
3. **翻译质量**：建议使用较大的模型获得更好效果
4. **格式保持**：系统会保持原始Markdown格式

## 故障排除

### Ollama连接问题
```bash
# 检查Ollama是否运行
ollama list

# 重启Ollama
ollama serve
```

### 模型不存在
```bash
# 查看可用模型
ollama list

# 下载模型
ollama pull qwen3:8b
```

### 翻译质量不佳
- 尝试使用更大的模型
- 调整temperature参数
- 添加更多上下文信息
