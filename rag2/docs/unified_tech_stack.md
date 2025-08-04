# RAG2项目统一技术栈说明

## 核心设计原则

为了简化部署和维护，RAG2项目采用统一的技术栈，最大化复用相同的模型和框架。

## 1. 统一的大模型选择

### 1.1 主LLM：DeepSeek系列
- **主要模型**: `DeepSeek-V2.5`
- **代码任务**: `DeepSeek-Coder-V2` (可选)
- **API接口**: 兼容OpenAI API格式
- **优势**: 
  - 中文支持优秀
  - API成本较低
  - 性能强劲
  - 统一接口，减少集成复杂度

### 1.2 嵌入模型：BAAI/BGE系列
- **主要模型**: `BAAI/bge-large-zh-v1.5`
- **多语言**: `BAAI/bge-m3` (备选)
- **部署方式**: HuggingFace Transformers本地部署
- **优势**:
  - 中文嵌入效果优秀
  - 开源免费
  - 与重排序模型同系列，兼容性好

### 1.3 重排序模型：BAAI/BGE Reranker
- **主要模型**: `BAAI/bge-reranker-v2-m3`
- **备选模型**: `BAAI/bge-reranker-large`
- **部署方式**: HuggingFace Transformers本地部署
- **优势**:
  - 与嵌入模型同系列
  - 多语言支持
  - 重排序效果优秀

## 2. 统一的开发框架

### 2.1 RAG开发框架：LangChain
- **核心包**: `langchain`
- **社区扩展**: `langchain-community`
- **文本分割**: `langchain-text-splitters`
- **实验功能**: `langchain-experimental`

**选择理由**:
- 成熟的RAG开发生态
- 丰富的组件和集成
- 活跃的社区支持
- 标准化的接口设计

### 2.2 Web服务框架：FastAPI
- **核心框架**: `fastapi`
- **ASGI服务器**: `uvicorn`
- **数据验证**: `pydantic`

**选择理由**:
- 高性能异步框架
- 自动API文档生成
- 类型提示支持
- 易于部署和扩展

## 3. 模型部署策略

### 3.1 本地部署模型
```python
# BAAI/BGE嵌入模型
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# BAAI/BGE重排序模型
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
```

### 3.2 API调用模型
```python
# DeepSeek LLM
from openai import OpenAI
client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com/v1"
)
```

## 4. LangChain集成示例

### 4.1 嵌入模型集成
```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
```

### 4.2 LLM集成
```python
from langchain.llms import OpenAI

llm = OpenAI(
    openai_api_key="your-deepseek-api-key",
    openai_api_base="https://api.deepseek.com/v1",
    model_name="deepseek-chat"
)
```

### 4.3 重排序器集成
```python
from langchain.retrievers.document_compressors import BGERerank

compressor = BGERerank(
    model_name="BAAI/bge-reranker-v2-m3",
    top_k=5
)
```

### 4.4 完整RAG链
```python
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever

# 压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever()
)

# RAG链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True
)
```

## 5. 分环境配置管理

### 5.1 开发环境配置
```python
# config/dev_config.py
DEV_MODEL_CONFIG = {
    "llm": {
        "provider": "ollama",  # 本地部署
        "model_name": "qwen2.5:7b",
        "api_base": "http://localhost:11434/v1",
        "temperature": 0.1,
        "max_tokens": 2048
    },
    "embedding": {
        "model_name": "BAAI/bge-base-zh-v1.5",  # 较小模型
        "device": "cpu",  # 开发环境可用CPU
        "normalize_embeddings": True,
        "dimensions": 768
    },
    "reranker": {
        "model_name": "BAAI/bge-reranker-base",  # 轻量级
        "top_k": 5,
        "use_fp16": False,  # CPU环境
        "device": "cpu"
    }
}
```

### 5.2 生产环境配置
```python
# config/prod_config.py
PROD_MODEL_CONFIG = {
    "llm": {
        "provider": "deepseek",
        "model_name": "deepseek-chat",
        "api_base": "https://api.deepseek.com/v1",
        "temperature": 0.1,
        "max_tokens": 4096
    },
    "embedding": {
        "model_name": "BAAI/bge-large-zh-v1.5",  # 大模型
        "device": "cuda",
        "normalize_embeddings": True,
        "dimensions": 1024
    },
    "reranker": {
        "model_name": "BAAI/bge-reranker-v2-m3",  # 最新版本
        "top_k": 10,
        "use_fp16": True,
        "device": "cuda"
    }
}
```

### 5.3 统一配置加载器
```python
# config/config_loader.py
import os
from .dev_config import DEV_MODEL_CONFIG
from .prod_config import PROD_MODEL_CONFIG

def get_model_config():
    env = os.getenv("RAG_ENV", "development")

    if env == "production":
        return PROD_MODEL_CONFIG
    else:
        return DEV_MODEL_CONFIG

# 使用示例
MODEL_CONFIG = get_model_config()
```

### 5.4 环境变量配置
```bash
# .env.development (开发环境)
RAG_ENV=development
OLLAMA_API_BASE=http://localhost:11434/v1
DEEPSEEK_API_KEY=your-deepseek-api-key  # 备选

# 模型缓存目录
HF_HOME=/path/to/huggingface/cache
TRANSFORMERS_CACHE=/path/to/transformers/cache

# 硬件配置
CUDA_VISIBLE_DEVICES=0
MODEL_DEVICE=cpu  # 开发环境使用CPU
```

```bash
# .env.production (生产环境)
RAG_ENV=production
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# 模型缓存目录
HF_HOME=/data/models/huggingface
TRANSFORMERS_CACHE=/data/models/transformers

# 硬件配置
CUDA_VISIBLE_DEVICES=0,1
MODEL_DEVICE=cuda  # 生产环境使用GPU
```

### 5.5 Ollama开发环境设置
```bash
# 安装Ollama (开发环境)
curl -fsSL https://ollama.ai/install.sh | sh

# 拉取开发用模型
ollama pull qwen2.5:7b

# 启动Ollama服务
ollama serve
```

## 6. 部署优势

### 6.1 简化的模型管理
- 只需部署3个核心模型
- 统一的HuggingFace模型格式
- 一致的API接口

### 6.2 降低资源需求
- 复用相同的模型权重
- 减少GPU显存占用
- 简化模型版本管理

### 6.3 统一的监控和维护
- 一致的日志格式
- 统一的性能指标
- 简化的故障排查

## 7. 扩展性考虑

### 7.1 模型替换
如需替换模型，只需修改配置文件，无需改动核心代码：
```python
# 替换为其他嵌入模型
"embedding": {
    "model_name": "text-embedding-3-large",  # OpenAI模型
    "provider": "openai"
}
```

### 7.2 多模型支持
可以通过配置支持多个模型并行：
```python
"embedding": {
    "primary": "BAAI/bge-large-zh-v1.5",
    "fallback": "text-embedding-3-small"
}
```

这种统一的技术栈设计确保了项目的一致性、可维护性和部署简便性，同时保持了足够的灵活性以适应未来的需求变化。
