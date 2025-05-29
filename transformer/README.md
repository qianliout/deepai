# Transformer从零实现 - 英语到意大利语翻译

这是一个从零开始实现的Transformer模型，用于英语到意大利语的机器翻译任务。

## 特性

- 🔥 **从零实现**: 完全自实现Transformer架构，包括多头注意力、位置编码、前馈网络等
- 📊 **Pydantic配置**: 使用Pydantic定义所有数据结构和超参数
- 🍎 **Apple Silicon支持**: 原生支持Mac M1/M2 GPU训练
- 📝 **自定义分词器**: 自实现分词功能，支持词典保存和加载
- 📚 **HuggingFace数据集**: 使用Helsinki-NLP/opus_books数据集
- 🎯 **基础矩阵运算**: 使用最基本的矩阵运算，便于学习理解
- 📋 **完整日志**: 详细的训练日志和进度跟踪
- 💾 **模型保存**: 支持模型检查点保存和加载

## 项目结构

```
transformer/auge/
├── config.py          # 配置文件（Pydantic数据结构）
├── utils.py           # 工具函数（日志、文件操作等）
├── tokenizer.py       # 自实现分词器
├── data_loader.py     # 数据加载和处理
├── model.py           # Transformer模型实现
├── trainer.py         # 训练器
├── main.py            # 主入口文件
├── inference.py       # 推理脚本
├── requirements.txt   # 依赖包
└── README.md          # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

```bash
# 使用默认配置训练
python main.py --mode train

# 使用自定义配置文件训练
python main.py --mode train --config my_config.json
```

### 2. 测试模型

```bash
# 测试训练好的模型
python main.py --mode test --model_path ./saved_models/best_model_epoch_10.pt
```

### 3. 交互式翻译

```bash
# 交互式翻译
python inference.py --model_path ./saved_models/best_model_epoch_10.pt --mode interactive

# 单句翻译
python inference.py --model_path ./saved_models/best_model_epoch_10.pt --mode single --text "Hello, how are you?"

# 批量翻译
python inference.py --model_path ./saved_models/best_model_epoch_10.pt --mode batch --input_file input.txt --output_file output.txt
```

## 配置说明

### 模型配置
- `d_model`: 模型隐藏层维度 (默认: 512)
- `d_ff`: 前馈网络维度 (默认: 2048)
- `n_heads`: 多头注意力头数 (默认: 8)
- `n_layers`: 编码器/解码器层数 (默认: 6)
- `max_seq_len`: 最大序列长度 (默认: 128)
- `dropout`: Dropout概率 (默认: 0.1)

### 训练配置
- `train_size`: 训练数据大小 (默认: 10000)
- `val_size`: 验证数据大小 (默认: 2000)
- `batch_size`: 批次大小 (默认: 32)
- `learning_rate`: 学习率 (默认: 1e-4)
- `num_epochs`: 训练轮数 (默认: 10)
- `device`: 训练设备 (自动检测: mps/cuda/cpu)

### 数据配置
- `dataset_name`: 数据集名称 (默认: "Helsinki-NLP/opus_books")
- `language_pair`: 语言对 (默认: "en-it")
- `min_freq`: 词汇最小频率 (默认: 2)
- `max_vocab_size`: 最大词汇表大小 (默认: 10000)

## 模型架构

### Transformer组件
1. **位置编码**: 为序列添加位置信息
2. **多头注意力**: 实现缩放点积注意力机制
3. **前馈网络**: 两层全连接网络
4. **层归一化**: 自实现层归一化
5. **编码器层**: 自注意力 + 前馈网络 + 残差连接
6. **解码器层**: 自注意力 + 交叉注意力 + 前馈网络 + 残差连接
7. **编码器**: 多层编码器层堆叠
8. **解码器**: 多层解码器层堆叠

### 关键特性
- 使用基础矩阵运算实现注意力机制
- 支持填充掩码和前瞻掩码
- 标签平滑损失函数
- 学习率预热调度
- 梯度裁剪

## 训练过程

1. **数据下载**: 自动下载HuggingFace数据集
2. **数据预处理**: 文本清理、分词、构建词典
3. **模型初始化**: 创建Transformer模型
4. **训练循环**: 前向传播、损失计算、反向传播
5. **验证**: 定期在验证集上评估
6. **模型保存**: 保存最佳模型和检查点

## 推理功能

- **贪心解码**: 每步选择概率最大的token
- **交互式翻译**: 实时输入翻译
- **批量翻译**: 处理文件中的多个句子
- **模型加载**: 支持从检查点恢复

## 日志和监控

- 训练进度实时显示
- 损失和困惑度跟踪
- 学习率变化监控
- 模型参数统计
- 详细的错误日志

## 注意事项

1. **内存需求**: 根据配置调整批次大小
2. **训练时间**: Mac M1上大约需要几小时
3. **数据集**: 首次运行会下载数据集
4. **词典**: 训练后会保存词典文件
5. **设备**: 自动检测并使用最佳设备

## 示例输出

```
英语: Hello, how are you?
意大利语: Ciao, come stai?

英语: I love programming.
意大利语: Amo la programmazione.

英语: The weather is nice today.
意大利语: Il tempo è bello oggi.
```

## 故障排除

### 常见问题
1. **内存不足**: 减小batch_size或模型维度
2. **设备错误**: 检查PyTorch MPS支持
3. **数据下载失败**: 检查网络连接
4. **词典文件缺失**: 重新运行训练生成词典

### 性能优化
1. 使用更大的批次大小（如果内存允许）
2. 调整学习率和预热步数
3. 增加模型层数和维度
4. 使用更多训练数据

## 扩展功能

可以基于此实现添加：
- Beam Search解码
- BLEU评分计算
- 注意力可视化
- 更多语言对支持
- 预训练模型加载

## 学习资源

这个实现专注于教学，每个组件都有详细注释，适合：
- 理解Transformer架构
- 学习注意力机制
- 掌握深度学习训练流程
- 实践PyTorch编程

## 许可证

MIT License
