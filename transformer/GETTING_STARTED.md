# 快速开始指南

## 🚀 一键运行

### 方法1: 使用一键脚本（推荐）
```bash
cd transformer/auge
python run.py
```

### 方法2: 直接运行主程序
```bash
cd transformer/auge
python main.py --mode train
```

## 📋 运行前检查

### 1. 验证环境设置
```bash
python test_setup.py
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

## 🎯 快速训练流程

### 步骤1: 训练模型
```bash
# 使用默认配置训练
python main.py --mode train

# 或使用自定义配置
python main.py --mode train --config example_config.json
```

### 步骤2: 测试模型
```bash
# 自动找到最佳模型进行测试
python main.py --mode test --model_path ./saved_models/best_model_epoch_X.pt
```

### 步骤3: 交互式翻译
```bash
python inference.py --model_path ./saved_models/best_model_epoch_X.pt --mode interactive
```

## 📊 训练配置说明

### 快速训练（测试用）
- 训练数据: 1000条
- 验证数据: 200条  
- 模型维度: 128
- 训练轮数: 3轮
- 预计时间: 10-20分钟

### 标准训练（推荐）
- 训练数据: 10000条
- 验证数据: 2000条
- 模型维度: 512
- 训练轮数: 10轮
- 预计时间: 2-4小时

### 高质量训练
- 训练数据: 50000条
- 验证数据: 10000条
- 模型维度: 768
- 训练轮数: 20轮
- 预计时间: 8-12小时

## 🔧 自定义配置

### 创建配置文件
```python
from config import Config

# 修改配置
config = Config()
config.model.d_model = 256
config.training.batch_size = 16
config.training.num_epochs = 5

# 保存配置
config.save_config("my_config.json")
```

### 使用配置文件
```bash
python main.py --mode train --config my_config.json
```

## 📱 使用示例

### 1. 一键训练和测试
```bash
python run.py --action train
```

### 2. 快速翻译
```bash
python run.py --action translate --text "Hello, how are you?"
```

### 3. 交互式翻译
```bash
python run.py --action interactive
```

## 🐛 常见问题

### Q: 内存不足怎么办？
A: 减小batch_size或模型维度
```json
{
  "training": {
    "batch_size": 8
  },
  "model": {
    "d_model": 256
  }
}
```

### Q: 训练速度太慢？
A: 检查是否使用GPU
```python
import torch
print(torch.backends.mps.is_available())  # Mac M1/M2
print(torch.cuda.is_available())          # NVIDIA GPU
```

### Q: 数据下载失败？
A: 检查网络连接，或使用代理
```bash
export HF_ENDPOINT=https://hf-mirror.com
python main.py --mode train
```

### Q: 翻译质量不好？
A: 增加训练数据和训练轮数
```json
{
  "training": {
    "train_size": 50000,
    "num_epochs": 20
  }
}
```

## 📈 监控训练

### 查看日志
```bash
tail -f logs/training_*.log
```

### 训练指标
- Loss: 损失值（越小越好）
- Perplexity: 困惑度（越小越好）
- Learning Rate: 学习率变化

### 保存的文件
- `saved_models/`: 模型检查点
- `vocab/`: 词典文件
- `logs/`: 训练日志
- `data_cache/`: 数据缓存

## 🎉 成功标志

训练成功后，你应该看到：
1. ✅ 模型文件保存在 `saved_models/`
2. ✅ 词典文件保存在 `vocab/`
3. ✅ 验证损失逐渐下降
4. ✅ 简单翻译测试有合理输出

## 🔄 下一步

1. **提升翻译质量**: 增加训练数据和轮数
2. **添加新功能**: 实现Beam Search解码
3. **评估模型**: 计算BLEU分数
4. **可视化**: 添加注意力权重可视化
5. **扩展语言**: 支持更多语言对

## 📞 获取帮助

如果遇到问题：
1. 运行 `python test_setup.py` 检查环境
2. 查看 `logs/` 目录下的详细日志
3. 检查 `README.md` 中的详细说明
4. 确保所有依赖正确安装

祝你训练愉快！🚀
