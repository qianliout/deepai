# BERT2 - 从零实现的BERT框架

这是一个从零实现的BERT框架，专为学习目的设计。代码结构清晰，注释详细，包含完整的预训练、微调和推理功能。

## 🏗️ 架构设计

### 模块组织
```
bert2/
├── config.py          # 配置模块 - 统一参数管理
├── transformer.py     # Transformer基础组件
├── model.py           # BERT模型实现
├── data_loader.py     # 数据加载器
├── trainer.py         # 预训练训练器
├── fine_tuning.py     # 微调模块
├── inference.py       # 推理模块
├── main.py           # 主运行脚本
├── requirements.txt   # 依赖包
└── README.md         # 说明文档
```

### 核心特性
- ✅ **完整的BERT实现**：包含MLM和NSP预训练任务
- ✅ **详细的shape注释**：每个tensor都有详细的形状说明
- ✅ **清晰的数据流转**：重点解释mask创建和使用逻辑
- ✅ **模块化设计**：各模块职责清晰，易于理解和扩展
- ✅ **配置驱动**：所有参数在config.py中统一管理
- ✅ **完善的日志**：详细的训练和推理日志
- ✅ **支持微调**：从预训练模型加载权重进行分类任务微调

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 快速测试
```bash
# 运行快速测试（小规模配置，适合学习）
python main.py quick
```

### 3. 完整训练
```bash
# 预训练
python main.py pretrain

# 微调
python main.py finetune

# 或者运行完整流程
python main.py full
```

### 4. 推理测试
```bash
# 预训练模型推理（掩码预测）
python main.py inference --model_type pretraining

# 分类模型推理
python main.py inference --model_type classification
```

## ⚙️ 配置说明

所有配置都在 `config.py` 中，无需手动传参：

### 模型配置
```python
# BERT模型架构
vocab_size: int = 30522          # 词汇表大小
hidden_size: int = 768           # 隐藏层维度
num_hidden_layers: int = 12      # Transformer层数
num_attention_heads: int = 12    # 注意力头数
max_position_embeddings: int = 512  # 最大位置嵌入
```

### 训练配置
```python
# 训练参数
batch_size: int = 16             # 批次大小
learning_rate: float = 1e-4      # 学习率
num_epochs: int = 3              # 训练轮数
max_samples: int = 1000          # 最大样本数（用于快速测试）
```

### 数据配置
```python
# 数据处理
tokenizer_name: str = "bert-base-uncased"  # tokenizer
max_length: int = 128            # 最大序列长度
mlm_probability: float = 0.15    # MLM掩码概率
```

## 📊 数据流转详解

### 1. MLM掩码逻辑
```python
# 原始文本: "I love cats"
# Tokenized: [CLS] I love cats [SEP]
# 掩码后:   [CLS] I [MASK] cats [SEP]
# 标签:     [-100, -100, love, -100, -100]

# 掩码策略：
# - 15%的token被选中进行掩码
# - 80%替换为[MASK]
# - 10%替换为随机token  
# - 10%保持不变
```

### 2. 注意力掩码
```python
# 输入: [1, 1, 1, 0, 0]  (1=真实token, 0=padding)
# 扩展: (batch_size, n_heads, seq_len, seq_len)
# 转换: 1->0(可注意), 0->-10000(不可注意)
# 在softmax中-10000变成接近0的概率
```

### 3. 数据形状变化
```python
# 输入处理流程：
input_ids: (batch_size, seq_len)
-> embeddings: (batch_size, seq_len, hidden_size)
-> encoder: (batch_size, seq_len, hidden_size)
-> pooler: (batch_size, hidden_size)
-> 任务头: (batch_size, vocab_size/num_labels)
```

## 🔧 核心组件说明

### Transformer组件 (`transformer.py`)
- **MultiHeadSelfAttention**: 多头自注意力机制
- **FeedForward**: 位置前馈网络
- **LayerNorm**: 层归一化
- **AddNorm**: 残差连接 + 层归一化
- **TransformerEncoderLayer**: 完整的编码器层

### BERT模型 (`model.py`)
- **BertEmbeddings**: 词嵌入 + 位置嵌入 + 类型嵌入
- **BertEncoder**: 多层Transformer编码器
- **BertPooler**: 序列池化层
- **BertForPreTraining**: 预训练模型（MLM + NSP）
- **BertForSequenceClassification**: 分类模型

### 数据处理 (`data_loader.py`)
- **BertDataCollator**: 动态MLM掩码
- **BertPretrainingDataset**: 预训练数据集（支持NSP）
- **BertClassificationDataset**: 分类数据集

## 📈 训练监控

### 预训练指标
- **总损失**: MLM损失 + NSP损失
- **MLM损失**: 掩码语言模型损失
- **NSP损失**: 下一句预测损失
- **学习率**: 动态学习率变化

### 微调指标
- **训练损失**: 分类损失
- **验证准确率**: 分类准确率
- **F1分数**: 加权F1分数
- **精确率/召回率**: 详细分类指标

## 🎯 使用示例

### 掩码预测
```python
from inference import BertInference

inference = BertInference("./bert2_output/best_model", "pretraining")
results = inference.predict_masked_tokens("The capital of France is [MASK].")
# 输出: Paris, Lyon, Marseille...
```

### 文本分类
```python
inference = BertInference("./bert2_output_finetune/best_model", "classification")
result = inference.classify_text("This movie is amazing!")
# 输出: {"predicted_class": 1, "confidence": 0.95}
```

### 文本相似度
```python
similarity = inference.compute_text_similarity("I love cats", "I adore felines")
# 输出: 0.85
```

## 🔍 学习要点

### 1. 注意力机制
- 理解Q、K、V的计算过程
- 掌握缩放点积注意力公式
- 学习多头注意力的并行计算

### 2. 掩码机制
- MLM掩码的随机策略
- 注意力掩码的创建和使用
- padding掩码的处理

### 3. 预训练任务
- MLM：学习双向语言表示
- NSP：学习句子间关系

### 4. 微调策略
- 权重初始化和加载
- 学习率调整
- 任务特定头部设计

## 📝 注意事项

1. **内存使用**: 默认配置需要较大内存，可以调整batch_size和max_samples
2. **训练时间**: 完整训练需要较长时间，建议先运行quick测试
3. **数据集**: 使用WikiText和IMDB数据集，首次运行会自动下载
4. **设备支持**: 自动检测CUDA/MPS/CPU，优先使用GPU

## 🤝 扩展建议

1. **添加更多任务**: 实现问答、命名实体识别等任务
2. **优化训练**: 添加混合精度训练、梯度累积等
3. **模型压缩**: 实现知识蒸馏、剪枝等技术
4. **可视化**: 添加注意力权重可视化功能

## 📚 参考资料

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

---

这个框架专为学习设计，代码结构清晰，注释详细。通过阅读和运行代码，你可以深入理解BERT的工作原理和实现细节。


我准备这样安排transformer模型存放的目录
预训练过程中的模型保存目录 saved_model/transformer/pretrain/checkpoint
预训练最佳模型保存目录 saved_model/transformer/pretrain/best
预训练完成后最终模型保存目录 saved_model/transformer/pretrain/final

微调过程中的模型保存目录 saved_model/bert/finetuning/checkpoint
微调过程中最佳模型保存目录 saved_model/bert/finetuning/best
微调完成后最终模型保存目录 saved_model/bert/finetuning/final

所有目录会在首次使用时自动创建，请按上述目录安排改正代码，要求这些目录在config.py中统一管理


