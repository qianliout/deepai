# T5框架实现

基于PyTorch的T5（Text-to-Text Transfer Transformer）框架实现，专为学习T5架构设计。

## 项目特点

- 🏗️ **清晰的代码结构**：模块化设计，易于理解和扩展
- 📊 **详细的数据流注释**：每个函数都包含详细的shape和数据流转说明
- 🔧 **统一配置管理**：使用pydantic进行配置验证和管理
- 📝 **完善的日志系统**：详细的训练和推理日志
- 🚀 **简单的命令接口**：一键训练、推理和测试
- 💻 **Mac M1支持**：针对Apple Silicon优化

## 项目结构

```
t5/
├── config.py          # 统一配置管理
├── transformer.py     # Transformer核心组件
├── model.py           # T5模型实现
├── data_loader.py     # 数据加载和预处理
├── trainer.py         # 训练器
├── inference.py       # 推理模块
├── main.py           # 统一入口
├── requirements.txt   # 依赖包
└── README.md         # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速测试

```bash
python main.py quick
```

这将使用小规模配置进行快速训练和测试，验证环境是否正确配置。

### 3. 完整训练

```bash
python main.py train
```

### 4. 推理测试

```bash
python main.py inference
```

### 5. 演示模式

```bash
python main.py demo
```

## 详细使用说明

### 配置管理

所有配置都在 `config.py` 中统一管理：

- **T5Config**: 模型架构参数
- **TrainingConfig**: 训练相关参数
- **DataConfig**: 数据处理参数
- **LoggingConfig**: 日志配置

### 支持的任务

1. **问答 (Question Answering)**
2. **文本摘要 (Text Summarization)**
3. **机器翻译 (Machine Translation)**
4. **自由文本生成 (Text Generation)**

### 数据流转

#### 训练数据流：
```
原始文本 → Tokenizer → T5DataSample → DataLoader → Model → Loss
```

#### 推理数据流：
```
输入文本 → 预处理 → 编码器 → 解码器 → 后处理 → 输出文本
```

### 模型架构

T5模型采用编码器-解码器架构：

1. **编码器**: 多层Transformer编码器，处理输入序列
2. **解码器**: 多层Transformer解码器，生成输出序列
3. **相对位置编码**: T5特有的相对位置偏置机制
4. **共享嵌入**: 编码器和解码器共享词嵌入层

### 关键特性

#### 1. 相对位置编码
T5使用相对位置编码而不是绝对位置编码，能够更好地处理不同长度的序列。

#### 2. 文本到文本格式
所有任务都转换为文本到文本的格式，统一了不同任务的处理方式。

#### 3. 预训练目标
支持多种预训练目标，包括掩码语言模型和去噪自编码器。

## 目录结构

### 模型保存目录
- 预训练模型: `/Users/liuqianli/work/python/deepai/saved_model/t5/pretrain/`
- 微调模型: `/Users/liuqianli/work/python/deepai/saved_model/t5/finetuning/`

### 日志目录
- 训练日志: `/Users/liuqianli/work/python/deepai/logs/t5/`

### 缓存目录
- 数据集缓存: `/Users/liuqianli/.cache/huggingface/datasets/`

## 性能优化

### Mac M1优化
- 自动检测并使用MPS设备
- 优化的内存使用策略
- 适配Apple Silicon的数据类型

### 训练优化
- 梯度累积
- 学习率预热
- 梯度裁剪
- 混合精度训练支持

## 扩展指南

### 添加新任务
1. 在 `data_loader.py` 中添加新的预处理函数
2. 在 `inference.py` 中添加任务特定的推理方法
3. 在 `main.py` 中添加交互式接口

### 修改模型架构
1. 在 `config.py` 中添加新的配置参数
2. 在 `model.py` 中修改模型结构
3. 确保所有shape注释保持准确

### 自定义数据集
1. 继承 `T5Dataset` 类
2. 实现自定义的预处理逻辑
3. 在 `create_data_loader` 中添加支持

## 常见问题

### Q: 如何调整模型大小？
A: 修改 `config.py` 中的 `T5Config` 参数，如 `d_model`、`num_layers` 等。

### Q: 如何使用自己的数据集？
A: 在 `data_loader.py` 中添加自定义的预处理函数，并在配置中指定数据集名称。

### Q: 如何调整训练参数？
A: 修改 `config.py` 中的 `TrainingConfig` 参数。

### Q: 内存不足怎么办？
A: 减小 `batch_size`、`max_samples` 或模型大小参数。

## 学习资源

### T5论文
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

### 相关概念
- Transformer架构
- 注意力机制
- 相对位置编码
- 文本到文本转换

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

MIT License
