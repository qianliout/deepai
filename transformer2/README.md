# Transformer2 - 从零实现的Transformer框架

这是一个从零实现的Transformer框架，用于学习和理解Transformer架构。代码结构清晰，注释详细，包含完整的数据流转说明和tensor shape注释。

## ✨ 特性

- 🏗️ **完整的Transformer架构**: 编码器-解码器结构
- 🔍 **多头注意力机制**: 详细的注意力计算过程
- 📍 **位置编码**: 正弦余弦位置编码
- 🔗 **残差连接**: 每个子层都有残差连接和层归一化
- 📊 **标签平滑损失**: 提高模型泛化能力
- 📈 **学习率调度**: Transformer论文中的预热调度策略
- 🎯 **多种解码方式**: 贪心解码和Beam Search
- 📝 **详细注释**: 每个tensor都有shape注释，完整的数据流转说明
- ⚙️ **配置驱动**: 所有参数通过config.py配置，无需手动传参
- 🚀 **简化运行**: 参考bert2实现，一键运行训练和推理

## 📁 项目结构

```
transformer2/
├── config.py          # 配置文件（所有超参数）
├── transformer.py     # Transformer核心组件
├── model.py           # 完整模型定义
├── data_loader.py     # 数据加载和预处理
├── trainer.py         # 训练器
├── inference.py       # 推理模块
├── main.py            # 主入口文件
├── requirements.txt   # 依赖包
└── README.md          # 说明文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速测试

```bash
# 使用小规模配置进行快速测试
python main.py quick
```

### 3. 训练模型

```bash
# 使用默认配置训练
python main.py train
```

### 4. 推理测试

```bash
# 交互式推理
python main.py inference

# 批量推理
python main.py inference --batch
```

### 5. 查看演示

```bash
# 查看框架特性和使用说明
python main.py demo
```

## ⚙️ 配置说明

所有超参数都在 `config.py` 中定义，主要包括：

### 模型配置 (TransformerConfig)
- `vocab_size_src/tgt`: 源/目标语言词汇表大小
- `d_model`: 模型维度 (默认512)
- `n_heads`: 注意力头数 (默认8)
- `n_layers`: 编码器/解码器层数 (默认6)
- `d_ff`: 前馈网络维度 (默认2048)
- `max_seq_len`: 最大序列长度 (默认512)
- `dropout`: Dropout概率 (默认0.1)

### 训练配置 (TrainingConfig)
- `batch_size`: 批次大小 (默认32)
- `learning_rate`: 学习率 (默认1e-4)
- `num_epochs`: 训练轮数 (默认10)
- `warmup_steps`: 预热步数 (默认4000)
- `dataset_name`: 数据集名称 (默认"Helsinki-NLP/opus_books")
- `language_pair`: 语言对 (默认"en-it")

### 数据配置 (DataConfig)
- `max_length`: 序列最大长度 (默认128)
- `min_freq`: 词汇最小频率 (默认2)
- `tokenizer_type`: 分词器类型 (默认"simple")

## 📊 数据流转说明

### 训练过程数据流转

```
原始文本对 (src, tgt)
    ↓ 分词编码
token序列 [seq_len]
    ↓ 添加特殊token + 填充
ID序列 [max_length]
    ↓ 批次化
批次数据 [batch_size, max_length]
    ↓ 词嵌入 + 位置编码
嵌入表示 [batch_size, seq_len, d_model]
    ↓ 编码器 (N层)
编码器输出 [batch_size, src_seq_len, d_model]
    ↓ 解码器 (N层)
解码器输出 [batch_size, tgt_seq_len, d_model]
    ↓ 输出投影
logits [batch_size, tgt_seq_len, vocab_size]
    ↓ 损失计算
loss [1]
```

### 推理过程数据流转

```
源文本
    ↓ 分词编码
源序列 [1, src_seq_len]
    ↓ 编码器
编码器输出 [1, src_seq_len, d_model]
    ↓ 解码器逐步生成
目标序列 [1, tgt_seq_len] (逐步增长)
    ↓ 解码
翻译结果文本
```

## 🔧 核心组件说明

### 1. 多头注意力 (MultiHeadAttention)
- 实现scaled dot-product attention
- 支持自注意力和交叉注意力
- 详细的shape变换注释

### 2. 位置编码 (PositionalEncoding)
- 正弦余弦位置编码
- 为序列添加位置信息

### 3. 前馈网络 (FeedForward)
- 两层全连接网络
- ReLU激活函数

### 4. 编码器层 (EncoderLayer)
- 多头自注意力 + 前馈网络
- 残差连接 + 层归一化

### 5. 解码器层 (DecoderLayer)
- 掩码自注意力 + 交叉注意力 + 前馈网络
- 三个子层，每个都有残差连接

## 📈 训练特性

### 损失函数
- **标签平滑损失**: 减少过拟合，提高泛化能力
- **忽略padding**: 不计算padding token的损失

### 优化器
- **Adam优化器**: 自适应学习率
- **梯度裁剪**: 防止梯度爆炸
- **权重衰减**: L2正则化

### 学习率调度
- **预热调度**: Transformer论文中的调度策略
- **动态调整**: 根据步数自动调整学习率

## 🎯 推理特性

### 解码方式
- **贪心解码**: 每步选择概率最大的token
- **Beam Search**: 保持多个候选序列，选择全局最优

### 温度采样
- **温度参数**: 控制生成的随机性
- **概率采样**: 支持随机采样生成

## 🛠️ 自定义配置

### 修改模型大小
```python
# 在config.py中修改
TRANSFORMER_CONFIG.d_model = 256      # 小模型
TRANSFORMER_CONFIG.n_heads = 4
TRANSFORMER_CONFIG.n_layers = 3
```

### 修改训练参数
```python
# 在config.py中修改
TRAINING_CONFIG.batch_size = 16       # 减小批次大小
TRAINING_CONFIG.learning_rate = 5e-5  # 调整学习率
TRAINING_CONFIG.num_epochs = 20       # 增加训练轮数
```

### 修改数据集
```python
# 在config.py中修改
TRAINING_CONFIG.dataset_name = "wmt14"
TRAINING_CONFIG.language_pair = "en-de"
```

## 📝 学习要点

### 1. 注意力机制
- 理解Query、Key、Value的作用
- 掌握多头注意力的并行计算
- 学习掩码的使用方法

### 2. 位置编码
- 理解为什么需要位置信息
- 掌握正弦余弦编码的原理

### 3. 残差连接
- 理解残差连接的作用
- 掌握层归一化的位置

### 4. 训练技巧
- 学习标签平滑的作用
- 理解学习率预热的重要性
- 掌握梯度裁剪的使用

## 🔍 调试技巧

### 1. 查看tensor shape
代码中每个关键位置都有shape注释，便于理解数据流转

### 2. 日志信息
详细的日志记录，包括：
- 模型参数数量
- 训练进度和指标
- 验证结果

### 3. 快速测试
使用 `python main.py quick` 进行小规模快速测试

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化解释
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 带注释的实现

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License
