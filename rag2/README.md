# RAG2 AIOps智能助手

基于RAG（Retrieval-Augmented Generation）技术的智能运维助手，专门为AIOps场景设计，集成了多种检索策略、知识图谱、向量数据库等先进技术。

## 🎯 项目特色

### 统一技术栈设计
- **主LLM**: DeepSeek-V2.5 (生产) / Qwen2.5:7B (开发)
- **嵌入模型**: BAAI/bge-large-zh-v1.5 (生产) / BAAI/bge-base-zh-v1.5 (开发)
- **重排序**: BAAI/bge-reranker-v2-m3 (生产) / BAAI/bge-reranker-base (开发)
- **开发框架**: LangChain + FastAPI
- **Mac M1优化**: 支持MPS加速

### 核心功能
- 🔍 **多模态检索**: 语义检索 + 关键词检索 + 图谱检索
- 🧠 **智能决策**: 自适应RAG + Self RAG
- 📚 **知识管理**: 文档处理、分块、向量化
- 💬 **对话管理**: 多轮对话、上下文压缩
- 📊 **性能监控**: 完整的日志和指标体系

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI Web  │    │   RAG Pipeline  │    │   LLM Client    │
│     Service     │───▶│     Manager     │───▶│   (DeepSeek)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Retrieval     │    │   Embedding     │
│   Processor     │    │   Manager       │    │   Manager       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Storage Layer                                │
│  PostgreSQL  │  MySQL  │  Redis  │  Elasticsearch  │  Neo4j    │
│  (向量存储)   │ (对话)   │ (缓存)   │   (全文检索)     │ (知识图谱) │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求
- Python 3.9+
- Docker & Docker Compose
- Anaconda (推荐使用aideep2环境)
- Mac M1 (项目已优化)

### 1. 克隆项目
```bash
cd /path/to/your/workspace
git clone <repository-url>
cd rag2
```

### 2. 环境配置
```bash
# 激活conda环境
conda activate aideep2

# 安装依赖
pip install -r requirements.txt
```

### 3. 启动数据库服务
```bash
# 启动所有数据库服务
docker-compose up -d

# 检查服务状态
docker ps
```

### 4. 配置环境变量
```bash
# 复制环境配置文件
cp .env.example .env

# 编辑配置文件
vim .env
```

必需的环境变量：
```bash
# 环境设置
RAG_ENV=development  # 或 production

# DeepSeek API (生产环境)
DEEPSEEK_API_KEY=your-deepseek-api-key

# 模型设备 (Mac M1)
MODEL_DEVICE=mps  # 或 cpu
```

### 5. 运行测试
```bash
# 简单测试（推荐先运行）
python test_simple.py

# 使用统一启动脚本测试
python start.py test

# 如果依赖完整，可以运行完整测试
python test_basic_setup.py
python test_complete_system.py
```

### 6. 启动API服务
```bash
# 使用统一启动脚本（推荐）
python start.py api

# 或直接启动
python run_api.py

# 或使用uvicorn（如果导入正常）
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📖 使用指南

### API接口

#### 1. 健康检查
```bash
curl http://localhost:8000/health
```

#### 2. 创建会话
```bash
curl -X POST http://localhost:8000/api/v1/query/session \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "session_name": "测试会话"}'
```

#### 3. 提问查询
```bash
curl -X POST http://localhost:8000/api/v1/query/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是容器安全？",
    "session_id": "your-session-id",
    "user_id": "test_user"
  }'
```

#### 4. 流式查询
```bash
curl -X POST http://localhost:8000/api/v1/query/ask/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何修复CVE-2024-1234漏洞？",
    "session_id": "your-session-id"
  }'
```

### Python SDK使用

```python
import asyncio
from core.rag_pipeline import get_rag_pipeline

async def main():
    # 获取RAG管道
    rag = await get_rag_pipeline()
    
    # 执行查询
    result = await rag.query("什么是AIOps安全？")
    
    print(f"问题: {result['query']}")
    print(f"回答: {result['response']}")
    print(f"检索到 {len(result['retrieved_documents'])} 个相关文档")

asyncio.run(main())
```

### 文档处理

```python
import asyncio
from core.document_processor import DocumentProcessor

async def process_documents():
    processor = DocumentProcessor()
    
    # 处理单个文档
    result = await processor.process_document("path/to/document.txt")
    
    # 处理目录中的所有文档
    results = await processor.process_documents_batch([
        "doc1.txt", "doc2.pdf", "doc3.md"
    ])
    
    return results

asyncio.run(process_documents())
```

## 🔧 配置说明

### 模型配置
项目支持开发和生产环境的不同模型配置：

**开发环境** (RAG_ENV=development):
- LLM: Qwen2.5:7B (Ollama本地部署)
- 嵌入: BAAI/bge-base-zh-v1.5 (768维)
- 重排序: BAAI/bge-reranker-base
- 设备: MPS (Mac M1) 或 CPU

**生产环境** (RAG_ENV=production):
- LLM: DeepSeek-V2.5 (API调用)
- 嵌入: BAAI/bge-large-zh-v1.5 (1024维)
- 重排序: BAAI/bge-reranker-v2-m3
- 设备: CUDA 或 MPS

### 数据库配置
- **PostgreSQL**: 向量存储 (pgvector扩展)
- **MySQL**: 对话历史和会话管理
- **Redis**: 实时缓存和会话状态
- **Elasticsearch**: 全文检索和关键词搜索
- **Neo4j**: 知识图谱和实体关系

## 📊 性能优化

### Mac M1优化
1. **MPS加速**: 自动检测并使用Metal Performance Shaders
2. **内存优化**: 针对M1芯片的统一内存架构优化
3. **模型量化**: 支持FP16量化减少内存占用

### 检索优化
1. **多阶段检索**: 粗排 + 精排
2. **缓存策略**: Redis缓存热点查询
3. **批量处理**: 支持文档批量处理和向量化

## 🧪 测试

### 运行所有测试
```bash
# 基础组件测试
python test_basic_setup.py

# 完整系统测试  
python test_complete_system.py

# API测试
pytest tests/
```

### 测试覆盖
- ✅ 配置系统
- ✅ 日志系统  
- ✅ 数据库连接
- ✅ 模型加载
- ✅ 文档处理
- ✅ 检索系统
- ✅ RAG管道
- ✅ API接口

## 📝 开发指南

### 添加新的检索器
```python
from retrieval.base_retriever import BaseRetriever

class CustomRetriever(BaseRetriever):
    def __init__(self):
        super().__init__("custom")
    
    async def retrieve(self, query: str, top_k: int = 10, **kwargs):
        # 实现检索逻辑
        return results
    
    async def add_documents(self, documents):
        # 实现文档添加逻辑
        return True
```

### 自定义文档处理器
```python
from core.document_processor import DocumentProcessor

class CustomProcessor(DocumentProcessor):
    def load_document(self, file_path: str):
        # 自定义文档加载逻辑
        return document
```

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   ```bash
   # 检查设备支持
   python -c "import torch; print(torch.backends.mps.is_available())"
   
   # 降级到CPU
   export MODEL_DEVICE=cpu
   ```

2. **数据库连接失败**
   ```bash
   # 检查Docker服务
   docker ps
   docker logs rag2_postgres
   
   # 重启服务
   docker-compose restart
   ```

3. **内存不足**
   ```bash
   # 使用小模型
   export RAG_ENV=development
   
   # 减少批处理大小
   # 在配置中调整chunk_size和batch_size
   ```

## 📚 学习资源

### 核心概念
1. **RAG技术**: 检索增强生成
2. **向量检索**: 语义相似度搜索
3. **重排序**: 提高检索精度
4. **上下文压缩**: 优化token使用
5. **多轮对话**: 会话状态管理

### 技术文档
- [LangChain官方文档](https://python.langchain.com/)
- [BAAI/BGE模型介绍](https://github.com/FlagOpen/FlagEmbedding)
- [DeepSeek API文档](https://platform.deepseek.com/api-docs/)
- [FastAPI官方文档](https://fastapi.tiangolo.com/)

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - RAG开发框架
- [BAAI/BGE](https://github.com/FlagOpen/FlagEmbedding) - 中文嵌入模型
- [DeepSeek](https://www.deepseek.com/) - 大语言模型
- [FastAPI](https://github.com/tiangolo/fastapi) - Web框架
