# RAG2 AIOps项目详细实现计划

## 项目概述
基于现有rag项目经验，实现一个面向AIOps场景的高级RAG系统，集成多种检索策略、知识图谱、向量数据库等技术。

## 大模型配置

### 分环境的模型配置策略：

#### **开发环境 (本地调试)**
1. **主LLM**:
   - `Qwen2.5:7B` (Ollama本地部署，快速调试)
   - `DeepSeek-V2-Lite-Chat` (API调用，轻量版本)

2. **嵌入模型**:
   - `BAAI/bge-base-zh-v1.5` (384维，较小模型)
   - `BAAI/bge-small-zh-v1.5` (512维，最小模型，备选)

3. **重排序模型**:
   - `BAAI/bge-reranker-base` (轻量级重排序)

#### **生产环境 (高性能)**
1. **主LLM**:
   - `DeepSeek-V2.5` (API调用，完整版本)
   - `DeepSeek-Coder-V2` (代码相关任务)

2. **嵌入模型**:
   - `BAAI/bge-large-zh-v1.5` (1024维，高精度)
   - `BAAI/bge-m3` (多语言支持，备选)

3. **重排序模型**:
   - `BAAI/bge-reranker-v2-m3` (最新版本，高性能)
   - `BAAI/bge-reranker-large` (大模型版本，备选)

#### **通用模型 (开发+生产)**
4. **NER模型**:
   - `bert-base-chinese` (中文实体识别)
   - `spacy zh_core_web_sm` (中文NLP管道)

5. **指代消解模型**:
   - `hfl/chinese-roberta-wwm-ext` (中文指代消解)

6. **分类模型**:
   - 基于主LLM的Few-shot分类 (问题类型分类)
   - `distilbert-base-chinese` (轻量级分类器，备选)

### 技术框架选择：
- **开发框架**: LangChain (统一的RAG开发框架)
- **Web框架**: FastAPI (高性能异步API服务)
- **模型部署**: HuggingFace Transformers + DeepSeek API + Ollama (开发环境)

### 环境配置策略：
- **开发环境**: 使用小模型，支持本地调试，快速迭代
- **生产环境**: 使用大模型，追求最佳性能和准确性
- **配置驱动**: 通过环境变量和配置文件实现模型切换
- **兼容性保证**: 相同的API接口，无缝环境切换

## 一、项目目录结构设计

```
rag2/
├── README.md                           # 项目总体介绍
├── requirements.txt                    # Python依赖
├── docker-compose.yml                  # 容器编排文件
├── config/                            # 配置文件目录
│   ├── __init__.py
│   ├── config.py                      # 主配置文件
│   ├── database_config.py             # 数据库配置
│   └── model_config.py                # 模型配置
├── core/                              # 核心功能模块
│   ├── __init__.py
│   ├── document_processor.py          # 文档处理和分块
│   ├── context_enhancer.py            # 上下文增强
│   ├── query_processor.py             # 查询重写和纠错
│   ├── reranker.py                    # 重排序模块
│   ├── context_compressor.py          # 上下文压缩
│   ├── session_manager.py             # 多轮会话管理
│   ├── feedback_loop.py               # 反馈回流
│   ├── adaptive_rag.py                # 自适应RAG
│   ├── self_rag.py                    # 自我决策RAG
│   └── fusion_retriever.py            # 结果融合检索
├── knowledge_graph/                   # 知识图谱模块
│   ├── __init__.py
│   ├── neo4j_manager.py               # Neo4j连接管理
│   ├── graph_builder.py               # 图谱构建
│   ├── graph_retriever.py             # 图谱检索
│   └── entity_linker.py               # 实体链接
├── nlp/                               # NLP处理模块
│   ├── __init__.py
│   ├── ner_processor.py               # 命名实体识别
│   ├── coreference_resolver.py        # 指代消解
│   └── text_analyzer.py               # 文本分析工具
├── storage/                           # 存储层模块
│   ├── __init__.py
│   ├── elasticsearch_manager.py       # ES管理
│   ├── pgvector_manager.py            # PgVector管理
│   ├── redis_manager.py               # Redis管理
│   ├── mysql_manager.py               # MySQL管理
│   └── vector_store.py                # 向量存储抽象层
├── retrieval/                         # 检索模块
│   ├── __init__.py
│   ├── base_retriever.py              # 基础检索器
│   ├── semantic_retriever.py          # 语义检索
│   ├── keyword_retriever.py           # 关键词检索
│   ├── hybrid_retriever.py            # 混合检索
│   └── graph_retriever.py             # 图谱检索
├── models/                            # 模型相关
│   ├── __init__.py
│   ├── embeddings.py                  # 嵌入模型
│   ├── llm_client.py                  # LLM客户端
│   └── rerank_models.py               # 重排序模型
├── api/                               # API接口
│   ├── __init__.py
│   ├── main.py                        # FastAPI主入口
│   ├── routes/                        # 路由模块
│   │   ├── __init__.py
│   │   ├── query.py                   # 查询接口
│   │   ├── document.py                # 文档管理接口
│   │   └── admin.py                   # 管理接口
│   └── schemas/                       # 数据模型
│       ├── __init__.py
│       ├── query_schemas.py
│       └── response_schemas.py
├── deployment/                        # 部署相关
│   ├── docker/                        # Docker配置
│   │   ├── Dockerfile
│   │   ├── redis.conf
│   │   └── nginx.conf
│   ├── scripts/                       # 部署脚本
│   │   ├── setup_databases.sh
│   │   ├── init_neo4j.sh
│   │   └── load_mock_data.sh
│   ├── sql/                           # SQL脚本
│   │   ├── init_mysql.sql
│   │   ├── init_postgresql.sql
│   │   └── mock_data.sql
│   └── neo4j/                         # Neo4j初始化
│       ├── init_schema.cypher
│       └── load_mock_data.cypher
├── data/                              # 数据目录
│   ├── documents/                     # 原始文档文件
│   │   ├── aiops_knowledge/           # AIOps知识库文档
│   │   │   ├── security_best_practices.txt
│   │   │   ├── vulnerability_analysis.txt
│   │   │   ├── container_security.txt
│   │   │   ├── incident_response.txt
│   │   │   └── compliance_guidelines.txt
│   │   └── technical_docs/            # 技术文档
│   │       ├── docker_security.txt
│   │       ├── kubernetes_hardening.txt
│   │       └── network_security.txt
│   ├── mock/                          # 模拟数据
│   │   ├── structured/                # 结构化数据
│   │   │   ├── hosts.json             # 主机数据
│   │   │   ├── images.json            # 镜像数据
│   │   │   ├── vulnerabilities.json   # 漏洞数据
│   │   │   └── relationships.json     # 实体关系数据
│   │   ├── scripts/                   # 数据导入脚本
│   │   │   ├── load_pg_data.py        # PostgreSQL数据导入
│   │   │   ├── load_es_data.py        # Elasticsearch数据导入
│   │   │   ├── load_neo4j_data.py     # Neo4j数据导入
│   │   │   ├── load_mysql_data.py     # MySQL数据导入
│   │   │   └── generate_vectors.py    # 向量数据生成
│   │   └── sample_queries.json        # 测试查询样例
│   └── logs/                          # 日志文件
├── tests/                             # 测试用例
│   ├── __init__.py
│   ├── unit/                          # 单元测试
│   │   ├── test_document_processor.py
│   │   ├── test_query_processor.py
│   │   ├── test_reranker.py
│   │   ├── test_ner_processor.py
│   │   └── test_knowledge_graph.py
│   ├── integration/                   # 集成测试
│   │   ├── test_rag_pipeline.py
│   │   ├── test_api_endpoints.py
│   │   └── test_database_integration.py
│   └── performance/                   # 性能测试
│       ├── test_retrieval_speed.py
│       └── test_memory_usage.py
├── docs/                              # 文档目录
│   ├── architecture.md                # 架构设计文档
│   ├── knowledge_points_mapping.md    # 知识点代码映射
│   ├── api_documentation.md           # API文档
│   ├── deployment_guide.md            # 部署指南
│   └── user_guide.md                  # 用户指南
└── utils/                             # 工具模块
    ├── __init__.py
    ├── logger.py                      # 日志工具
    ├── metrics.py                     # 性能指标
    └── helpers.py                     # 辅助函数
```

## 二、核心技术实现计划

### 2.1 文档分块 (Document Chunking)
**实现文件**: `core/document_processor.py`
**依赖包**:
```python
langchain==0.1.0
langchain-community==0.0.20
langchain-text-splitters==0.0.1
transformers==4.36.0
jieba==0.42.1
pypdf2==3.0.1
python-docx==0.8.11
markdown==3.5.1
beautifulsoup4==4.12.2
```

**主要工具**:
- LangChain RecursiveCharacterTextSplitter
- LangChain SemanticChunker (基于BAAI/bge嵌入)
- JiebaTokenizer (中文分词)
- 自定义语义分块算法

**功能特性**:
- 支持多种文档格式 (txt, pdf, docx, md)
- 基于LangChain的智能语义分块
- 使用BAAI/bge模型进行语义相似度计算
- 重叠窗口策略，避免信息丢失
- 元数据提取和保存

### 2.2 上下文增强 (Context Enhancement)
**实现文件**: `core/context_enhancer.py`
**依赖包**:
```python
scikit-learn==1.3.2
gensim==4.3.2
textrank4zh==0.3
keybert==0.8.3
```

**主要工具**:
- 实体识别增强
- 关键词提取
- 主题建模

**功能特性**:
- 基于实体的上下文扩展
- 历史对话上下文融合
- 动态上下文权重调整

### 2.3 查询重写和智能纠错 (Query Rewriting & Error Correction)
**实现文件**: `core/query_processor.py`
**依赖包**:
```python
pyspellchecker==0.7.2
synonyms==3.17.0
fuzzywuzzy==0.18.0
python-levenshtein==0.21.1
opencc-python-reimplemented==0.1.7
```

**主要工具**:
- 同义词词典
- 拼写检查器
- LLM查询扩展

**功能特性**:
- 查询意图识别
- 同义词替换和扩展
- 拼写错误自动纠正
- 多Query生成策略

### 2.4 重排序系统 (Reranking)
**实现文件**: `core/reranker.py`
**依赖包**:
```python
langchain==0.1.0
langchain-community==0.0.20
sentence-transformers==2.2.2
transformers==4.36.0
torch==2.1.0
FlagEmbedding==1.2.5
```

**主要工具**:
- LangChain BGERerank (基于BAAI/bge-reranker-v2-m3)
- BAAI/bge-reranker-large Cross-Encoder
- 自定义多阶段重排序器

**功能特性**:
- 基于LangChain的重排序集成
- 使用BAAI/bge重排序模型
- 多阶段重排序流程
- 相关性分数融合
- 多样性保证机制

### 2.5 上下文压缩 (Context Compression)
**实现文件**: `core/context_compressor.py`
**依赖包**:
```python
llmlingua==0.1.4
tiktoken==0.5.2
nltk==3.8.1
```

**主要工具**:
- LLMLingua
- 关键信息提取
- 动态压缩比调整

**功能特性**:
- 智能内容压缩
- 关键信息保留
- Token使用优化

### 2.6 多轮会话管理 (Multi-turn Conversation)
**实现文件**: `core/session_manager.py`
**依赖包**:
```python
redis==5.0.1
pydantic==2.5.0
uuid==1.30
```

**主要工具**:
- Redis会话存储
- 对话状态跟踪
- 上下文窗口管理

**功能特性**:
- 会话状态持久化
- 对话历史压缩
- 上下文相关性评分

### 2.7 反馈回流 (Feedback Loop)
**实现文件**: `core/feedback_loop.py`
**依赖包**:
```python
mlflow==2.8.1
prometheus-client==0.19.0
numpy==1.24.3
```

**主要工具**:
- 用户反馈收集
- 模型性能监控
- 自动调优机制

**功能特性**:
- 实时反馈收集
- 检索质量评估
- 动态参数调整

### 2.8 自适应RAG (Adaptive RAG)
**实现文件**: `core/adaptive_rag.py`
**依赖包**:
```python
langchain==0.1.0
langchain-community==0.0.20
openai==1.6.1  # DeepSeek API兼容
```

**主要工具**:
- DeepSeek-V2.5 问题类型分类
- LangChain RouterChain 策略选择
- 基于规则的决策引擎

**功能特性**:
- 使用DeepSeek进行问题类型识别
- 动态检索策略选择
- 基于LangChain的路由机制
- 性能监控和调优

### 2.9 自我决策RAG (Self RAG)
**实现文件**: `core/self_rag.py`
**依赖包**:
```python
langchain==0.1.0
langchain-community==0.0.20
openai==1.6.1  # DeepSeek API兼容
```

**主要工具**:
- DeepSeek-V2.5 知识需求判断
- LangChain SelfQueryRetriever
- 置信度评估机制

**功能特性**:
- 使用DeepSeek判断是否需要检索
- 基于LangChain的自查询机制
- 答案置信度评估
- 检索必要性决策

### 2.10 知识图谱 (Knowledge Graph)
**实现文件**: `knowledge_graph/`目录
**依赖包**:
```python
neo4j==5.14.1
py2neo==2021.2.4
networkx==3.2.1
rdflib==7.0.0
```

**主要工具**:
- Neo4j图数据库
- 实体抽取工具
- 关系抽取模型

**功能特性**:
- 三元组构建和存储
- 图谱检索和推理
- 实体关系可视化

### 2.11 结果融合检索 (Fusion Retrieval)
**实现文件**: `core/fusion_retriever.py`
**依赖包**:
```python
rank-bm25==0.2.2
scipy==1.11.4
```

**主要工具**:
- 多通道检索器
- 分数融合算法
- 排序优化

**功能特性**:
- 语义+关键词双通道
- 智能分数融合
- 结果去重和排序

### 2.12 NLP增强功能
**实现文件**: `nlp/`目录
**依赖包**:
```python
spacy==3.7.2
zh-core-web-sm==3.7.0
transformers==4.36.0
torch==2.1.0
allennlp==2.10.1
```

**主要工具**:
- spaCy NER模型
- 指代消解模型
- 中文NLP工具包

**功能特性**:
- 实体识别和分类
- 指代关系解析
- 文本预处理优化

## 三、数据存储架构

### 3.1 PostgreSQL + pgvector
**依赖包**:
```python
psycopg2-binary==2.9.9
pgvector==0.2.4
sqlalchemy==2.0.23
```
- 向量数据存储
- 文档元数据管理
- 相似性搜索

### 3.2 Elasticsearch
**依赖包**:
```python
elasticsearch==8.11.0
elasticsearch-dsl==8.11.0
```
- 全文检索
- 关键词搜索
- 文档索引管理

### 3.3 MySQL
**依赖包**:
```python
pymysql==1.1.0
mysql-connector-python==8.2.0
```
- 对话历史存储
- 用户会话管理
- 系统配置数据

### 3.4 Redis
**依赖包**:
```python
redis==5.0.1
redis-py-cluster==2.1.3
```
- 会话缓存
- 查询结果缓存
- 实时数据存储

### 3.5 Neo4j
**依赖包**:
```python
neo4j==5.14.1
py2neo==2021.2.4
```
- 知识图谱存储
- 实体关系管理
- 图谱查询和推理

## 四、API和Web服务

### 4.1 FastAPI Web服务
**依赖包**:
```python
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
```

### 4.2 其他通用依赖
**依赖包**:
```python
requests==2.31.0
aiohttp==3.9.1
python-dotenv==1.0.0
loguru==0.7.2
pytest==7.4.3
pytest-asyncio==0.21.1
```

## 五、Mock数据脚本详细说明

### 5.1 PostgreSQL数据导入脚本 (`data/mock/scripts/load_pg_data.py`)
**功能**:
- 创建向量表结构
- 导入文档向量数据
- 建立向量索引
- 导入文档元数据

**数据内容**:
- 1000条AIOps知识文档向量
- 文档分块后的嵌入向量
- 文档元数据(标题、来源、时间戳等)

### 5.2 Elasticsearch数据导入脚本 (`data/mock/scripts/load_es_data.py`)
**功能**:
- 创建文档索引
- 导入全文检索数据
- 配置中文分词器
- 建立搜索模板

**数据内容**:
- 同步PostgreSQL中的文档内容
- 支持全文检索的文档分块
- 关键词索引和倒排索引

### 5.3 Neo4j数据导入脚本 (`data/mock/scripts/load_neo4j_data.py`)
**功能**:
- 创建节点和关系
- 导入主机、镜像、漏洞数据
- 建立实体关系图谱
- 创建图谱索引

**数据内容**:
- 100台主机节点
- 200个镜像节点
- 500个漏洞节点
- 主机-镜像、镜像-漏洞关系

### 5.4 MySQL数据导入脚本 (`data/mock/scripts/load_mysql_data.py`)
**功能**:
- 创建用户会话表
- 导入历史对话数据
- 创建反馈数据表
- 导入系统配置

**数据内容**:
- 50个用户会话记录
- 200条历史对话数据
- 用户反馈和评分数据

## 六、部署和运维

### 6.1 Docker容器化
**依赖包**:
```python
docker==6.1.3
docker-compose==1.29.2
```
- 多服务容器编排
- 环境隔离和管理
- 一键部署脚本

### 6.2 数据初始化
- 数据库schema创建
- 模拟数据生成
- 索引构建脚本

### 6.3 监控和日志
**依赖包**:
```python
prometheus-client==0.19.0
grafana-api==1.0.3
```
- 详细的操作日志
- 性能指标监控
- 错误追踪和报警

## 五、测试策略

### 5.1 单元测试
- 每个核心模块的独立测试
- 功能正确性验证
- 边界条件测试

### 5.2 集成测试
- 端到端流程测试
- 数据库集成测试
- API接口测试

### 5.3 性能测试
- 检索速度测试
- 内存使用监控
- 并发性能测试

## 六、文档和学习资源

### 6.1 知识点映射文档
详细记录每个RAG知识点在代码中的具体实现位置，便于学习和理解。

### 6.2 架构设计文档
系统整体架构说明，模块间的交互关系，数据流向等。

### 6.3 API文档
完整的接口文档，包括请求参数、响应格式、使用示例等。

### 6.4 部署指南
详细的部署步骤，环境配置，故障排除等。

## 七、开发优先级

1. **第一阶段**: 基础架构搭建 (存储层、配置管理、日志系统)
2. **第二阶段**: 核心检索功能 (文档处理、向量检索、基础RAG)
3. **第三阶段**: 高级功能 (重排序、查询重写、上下文增强)
4. **第四阶段**: 知识图谱集成 (Neo4j、图谱检索)
5. **第五阶段**: 智能化功能 (自适应RAG、Self RAG、反馈回流)
6. **第六阶段**: NLP增强 (NER、指代消解)
7. **第七阶段**: 性能优化和测试完善

## 八、完整依赖包列表

项目已创建完整的 `requirements.txt` 文件，包含所有必需的Python包：

### 统一技术栈核心组件：
1. **Web框架**: FastAPI + Uvicorn (高性能异步API)
2. **RAG框架**: LangChain生态 (统一开发框架)
3. **主LLM**: DeepSeek-V2.5 (统一的语言模型)
4. **嵌入模型**: BAAI/bge-large-zh-v1.5 (HuggingFace部署)
5. **重排序**: BAAI/bge-reranker-v2-m3 (同系列模型)
6. **中文NLP**: jieba, spacy (中文处理优化)
7. **数据库**: PostgreSQL+pgvector, ES, MySQL, Redis, Neo4j
8. **开发工具**: pytest, black, flake8, mypy

### 分环境技术栈优势：
- **开发效率**: 开发环境使用小模型，快速调试和迭代
- **成本控制**: 本地Ollama部署，减少API调用成本
- **性能保证**: 生产环境使用大模型，确保最佳效果
- **无缝切换**: 统一的API接口，配置驱动的环境切换
- **资源优化**: 根据环境自动调整资源使用策略
- **模型兼容**: BAAI/BGE系列保证不同规模模型的兼容性

### Mock数据生成：
- 已创建 `generate_all_mock_data.py` 脚本
- 生成100台主机、200个镜像、500个漏洞的测试数据
- 包含实体关系和测试查询样例
- 提供专门的数据导入脚本到各个数据库

### 知识库文档：
- 创建了AIOps安全最佳实践指南
- 漏洞分析与风险评估指南
- 后续将添加更多技术文档

## 九、项目特色和创新点

### 9.1 统一技术栈设计
1. **模型统一**: 使用DeepSeek作为主LLM，BAAI/BGE系列作为嵌入和重排序
2. **框架统一**: 基于LangChain的统一RAG开发框架
3. **接口统一**: DeepSeek兼容OpenAI API，简化集成
4. **部署统一**: 最小化模型数量，降低资源需求

### 9.2 技术创新特点
1. **多模态检索融合**: 语义+关键词+图谱的三重检索策略
2. **智能化决策**: 基于DeepSeek的自适应RAG和Self RAG
3. **中文优化**: 针对中文场景的专门优化(jieba分词、BGE中文模型)
4. **AIOps专业化**: 面向运维场景的专业知识图谱和领域知识
5. **LangChain深度集成**: 充分利用LangChain生态的组件和工具

### 9.3 工程化优势
1. **配置驱动**: 统一的配置管理，支持模型热切换
2. **监控完善**: 基于FastAPI的完整监控和日志体系
3. **测试覆盖**: 完整的单元测试、集成测试、性能测试
4. **文档齐全**: 详细的技术文档和知识点映射
5. **部署简化**: Docker容器化，一键部署脚本

### 9.4 学习价值
1. **知识点全覆盖**: 涵盖RAG的14个核心知识点
2. **代码可读性**: 清晰的模块划分和代码注释
3. **实践导向**: 基于真实AIOps场景的应用实践
4. **技术前沿**: 集成最新的RAG技术和模型

这个统一技术栈的设计确保了项目的一致性、可维护性和部署简便性，同时保持了足够的技术深度和学习价值。通过使用相同系列的模型和统一的开发框架，大大简化了系统的复杂度，提高了开发和运维效率。
