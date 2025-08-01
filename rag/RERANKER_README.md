# RAG系统重排序器使用指南

本文档介绍如何在RAG系统中使用双编码器和交叉编码器进行重排序，以提升检索结果的相关性和准确性。

## 概述

重排序器是RAG系统中的重要组件，用于在初始检索结果的基础上进行精细化排序。本系统实现了三种重排序器：

1. **双编码器重排序器 (BiEncoderReranker)**: 使用双编码器模型分别编码查询和文档，计算向量相似度
2. **交叉编码器重排序器 (CrossEncoderReranker)**: 使用交叉编码器模型联合编码查询-文档对，直接输出相关性分数
3. **混合重排序器 (HybridReranker)**: 结合双编码器和交叉编码器的优势，实现多阶段重排序

## 架构设计

```
初始检索结果 → 双编码器粗排 → 交叉编码器精排 → 分数融合 → 最终结果
     ↓              ↓              ↓           ↓
   50个候选      →  20个候选    →   10个结果  →  排序结果
```

### 重排序流程

1. **双编码器粗排**: 快速筛选大量候选文档，计算效率高
2. **交叉编码器精排**: 对筛选后的候选进行精确排序，准确性高
3. **分数融合**: 综合原始检索分数、双编码器分数和交叉编码器分数

## 配置说明

### 重排序器配置 (config.py)

```python
class RerankerConfig(BaseModel):
    # 双编码器配置
    bi_encoder_model: str = "BAAI/bge-reranker-base"
    bi_encoder_enabled: bool = True
    bi_encoder_top_k: int = 20
    
    # 交叉编码器配置
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"
    cross_encoder_enabled: bool = True
    cross_encoder_top_k: int = 10
    
    # 分数融合配置
    fusion_method: str = "weighted"  # weighted, rrf, max
    bi_encoder_weight: float = 0.3
    cross_encoder_weight: float = 0.7
    original_weight: float = 0.2
```

### 融合方法说明

1. **weighted**: 加权平均融合
   ```
   final_score = original_weight * original_score + 
                 bi_encoder_weight * bi_encoder_score + 
                 cross_encoder_weight * cross_encoder_score
   ```

2. **rrf**: Reciprocal Rank Fusion
   ```
   rrf_score = Σ(1 / (k + rank_i))
   ```

3. **max**: 取最大分数
   ```
   final_score = max(original_score, bi_encoder_score, cross_encoder_score)
   ```

## 使用方法

### 1. 基本使用

```python
from reranker import HybridReranker, rerank_documents
from langchain_core.documents import Document

# 创建文档
documents = [
    Document(page_content="机器学习是AI的重要分支...", metadata={"title": "ML基础"}),
    Document(page_content="深度学习使用神经网络...", metadata={"title": "DL概述"}),
]

# 方法1: 使用混合重排序器
reranker = HybridReranker()
results = reranker.rerank("什么是机器学习", documents, top_k=5)

# 方法2: 使用便捷函数
results = rerank_documents("什么是机器学习", documents, "hybrid", top_k=5)
```

### 2. 集成到检索器

```python
from retriever import HybridRetrieverManager

# 创建启用重排序的检索器
retriever = HybridRetrieverManager(
    vector_store=vector_store,
    enable_reranking=True
)

# 执行检索（自动应用重排序）
results = retriever.retrieve(
    query="机器学习算法",
    top_k=10,
    use_reranking=True
)
```

### 3. 单独使用不同重排序器

```python
from reranker import BiEncoderReranker, CrossEncoderReranker

# 双编码器重排序
bi_encoder = BiEncoderReranker()
bi_results = bi_encoder.rerank(query, documents, top_k=10)

# 交叉编码器重排序
cross_encoder = CrossEncoderReranker()
cross_results = cross_encoder.rerank(query, documents, top_k=5)
```

## 模型要求

### 推荐模型

1. **双编码器模型**:
   - `BAAI/bge-reranker-base`: 中文优化的双编码器模型
   - `BAAI/bge-reranker-large`: 更大的双编码器模型，精度更高

2. **交叉编码器模型**:
   - `BAAI/bge-reranker-v2-m3`: 多语言交叉编码器模型
   - `BAAI/bge-reranker-v2-gemma`: 基于Gemma的交叉编码器

### 模型下载

```bash
# 确保网络连接正常，模型会自动从Hugging Face下载
# 首次使用时需要下载模型文件，请耐心等待

# 可以预先下载模型
python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('BAAI/bge-reranker-base')
CrossEncoder('BAAI/bge-reranker-v2-m3')
"
```

## 性能优化

### 1. 批处理优化

```python
# 调整批处理大小
defaultConfig.reranker.batch_size = 32  # 根据GPU内存调整
```

### 2. 设备配置

```python
# 设置计算设备
defaultConfig.reranker.device = "mps"  # macOS GPU
defaultConfig.reranker.device = "cuda"  # NVIDIA GPU
defaultConfig.reranker.device = "cpu"   # CPU
```

### 3. 候选数量调优

```python
# 平衡精度和效率
defaultConfig.reranker.bi_encoder_top_k = 20   # 双编码器筛选数量
defaultConfig.reranker.cross_encoder_top_k = 10  # 最终返回数量
```

## 测试和验证

### 运行测试

```bash
# 运行重排序器测试
python test_reranker.py

# 运行示例演示
python example_reranker.py
```

### 性能基准

```python
# 性能测试示例
from test_reranker import TestReranker

test = TestReranker()
test.setup_class()
test.test_performance_benchmark()
```

## 故障排除

### 常见问题

1. **模型下载失败**
   - 检查网络连接
   - 确认Hugging Face可访问
   - 尝试使用镜像源

2. **内存不足**
   - 减小batch_size
   - 使用CPU而非GPU
   - 减少候选文档数量

3. **重排序器初始化失败**
   - 检查模型路径是否正确
   - 确认依赖包已安装
   - 查看错误日志详情

### 调试技巧

```python
# 启用详细日志
import logging
logging.getLogger("HybridReranker").setLevel(logging.DEBUG)

# 检查重排序器状态
reranker = HybridReranker()
info = reranker.get_reranker_info()
print(info)
```

## 最佳实践

1. **模型选择**: 根据数据特点选择合适的模型
2. **参数调优**: 根据实际效果调整融合权重
3. **性能监控**: 定期评估重排序效果
4. **资源管理**: 合理配置计算资源
5. **错误处理**: 实现优雅的降级机制

## 扩展开发

### 自定义重排序器

```python
from reranker import RerankResult

class CustomReranker:
    def rerank(self, query: str, documents: List[Document], top_k: int) -> List[RerankResult]:
        # 实现自定义重排序逻辑
        pass
```

### 新增融合方法

```python
def custom_fusion(original_score, bi_score, cross_score):
    # 实现自定义分数融合逻辑
    return fused_score
```

## 参考资料

- [BGE模型介绍](https://github.com/FlagOpen/FlagEmbedding)
- [Sentence Transformers文档](https://www.sbert.net/)
- [重排序技术综述](https://arxiv.org/abs/2103.14469)
