# RAG个人知识库系统 - 简化版

一个基于LangChain的简化版检索增强生成(RAG)个人知识库系统，专注于核心功能，易于使用和部署。

## 🌟 主要特性

- **简化架构**: 只支持TXT文档，专注核心RAG功能
- **ChromaDB存储**: 使用ChromaDB作为唯一向量存储，底层SQLite持久化
- **BGE嵌入模型**: 使用BAAI/bge-small-zh-v1.5中文嵌入模型
- **通义百炼LLM**: 集成阿里云通义百炼大语言模型
- **Redis会话**: 支持对话历史和会话管理
- **一键测试**: 提供quick命令快速体验系统功能
- **配置统一**: 所有配置集中在config.py中管理
- **扁平结构**: 所有模块文件都在根目录下，简化导入关系

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 4GB+ RAM
- Apple Silicon GPU (MPS) 或 CPU

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置设置

设置环境变量：
```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
```

### 基本使用

1. **测试导入** (验证环境)：
```bash
python test_imports.py
```

2. **快速测试** (推荐首次使用)：
```bash
python main.py quick
```

3. **构建知识库**：
```bash
python main.py build --docs ./your_txt_documents_folder
```

4. **查询问答**：
```bash
python main.py query "你的问题"
```

5. **交互式对话**：
```bash
python main.py chat
```

## 📁 项目结构

```
rag/
├── config.py              # 统一配置管理
├── main.py                # 主程序入口
├── requirements.txt       # 简化依赖列表
├── embeddings.py          # BGE嵌入模型
├── vector_store.py        # ChromaDB向量存储
├── llm.py                 # 通义百炼LLM
├── rag_chain.py           # RAG链
├── retriever.py           # 检索器
├── document_loader.py     # TXT文档加载器
├── text_splitter.py       # 文本分割器
├── logger.py              # 日志管理
├── test_imports.py        # 导入测试脚本
└── data/                  # 数据目录
    ├── documents/         # TXT文档存储
    ├── vectorstore/       # ChromaDB数据
    ├── models/           # 模型缓存
    └── logs/             # 日志文件
```

## 🔧 配置说明

所有配置都在 `config.py` 中统一管理：

### 嵌入模型配置
```python
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-zh-v1.5"  # BGE中文嵌入模型
    device: DeviceType = DeviceType.MPS         # Apple Silicon GPU
    max_length: int = 512                       # 最大文本长度
    batch_size: int = 32                        # 批处理大小
```

### ChromaDB配置
```python
class ChromaDBConfig:
    persist_directory: str = "data/vectorstore"  # 数据持久化目录
    collection_name: str = "knowledge_base"      # 集合名称
    top_k: int = 5                              # 检索返回数量
    score_threshold: float = 0.7                # 相似度阈值
```

### LLM配置
```python
class LLMConfig:
    api_key: str = ""                    # 通义百炼API密钥
    model_name: str = "qwen-max"         # 模型名称
    temperature: float = 0.7             # 采样温度
    max_tokens: int = 2048              # 最大生成token数
```

## 📖 使用示例

### 快速测试流程

```bash
# 1. 测试环境
python test_imports.py

# 2. 快速测试 - 自动创建示例文档并测试
python main.py quick

# 输出示例：
# 🚀 RAG系统快速测试
# ✅ RAG系统初始化完成
# 📚 开始构建知识库
# 📄 成功加载 1 个文档片段
# 🔍 正在构建向量索引...
# ✅ 知识库构建完成
# 🧪 开始测试查询...
# 测试问题: 什么是RAG？
# 回答: RAG（Retrieval-Augmented Generation）是一种结合了检索和生成的AI技术...
```

### Python API使用

```python
from main import RAGSystem

# 初始化系统
rag = RAGSystem()
rag.initialize()

# 构建知识库
rag.build_knowledge_base("./documents", clear_existing=True)

# 查询
answer = rag.query_knowledge_base("什么是机器学习？")
print(answer)

# 获取统计信息
stats = rag.vector_store.get_stats()
print(f"文档数量: {stats['document_count']}")
```

## 🎯 核心功能

### 文档处理
- 只支持TXT格式文档
- 自动编码检测 (UTF-8, GBK, GB2312, Latin-1)
- 智能文本分割，保持语义完整性
- 批量文档加载和处理

### 向量检索
- ChromaDB向量存储，SQLite持久化
- BGE中文嵌入模型，支持中英文
- 余弦相似度计算
- 可配置的检索数量和阈值

### 智能问答
- 通义百炼大语言模型
- 基于检索上下文的生成
- 支持对话历史管理
- 可配置的生成参数

## 🔍 监控和调试

### 查看系统状态
```bash
python main.py status
```

### 日志配置
日志自动保存到 `data/logs/` 目录，支持不同级别：
- DEBUG: 详细调试信息
- INFO: 一般信息 (默认)
- WARNING: 警告信息
- ERROR: 错误信息

### 性能监控
```python
# 获取详细统计
stats = rag_system.vector_store.get_stats()
print(f"""
存储类型: {stats['store_type']}
文档数量: {stats['document_count']}
嵌入维度: {stats['embedding_dim']}
集合名称: {stats['collection_name']}
""")
```

## 🚨 注意事项

1. **API密钥**: 需要配置通义百炼API密钥才能使用LLM功能
2. **文档格式**: 只支持TXT格式，其他格式需要先转换
3. **内存使用**: BGE模型需要约1GB内存
4. **设备支持**: 优先使用Apple Silicon GPU (MPS)，回退到CPU
5. **扁平结构**: 所有模块都在根目录下，简化了导入关系

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用框架
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [BGE](https://github.com/FlagOpen/FlagEmbedding) - 中文嵌入模型
- [通义百炼](https://dashscope.aliyun.com/) - 大语言模型API

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！
