# RAG2项目故障排除指南

## 🚨 常见问题及解决方案

### 1. ImportError: attempted relative import beyond top-level package

**问题描述**: 运行脚本时出现相对导入错误

**解决方案**:
```bash
# ❌ 不要直接运行这些命令
python test_basic_setup.py
python test_complete_system.py
python api/main.py

# ✅ 使用正确的启动方式
python test_simple.py          # 简单测试
python start.py test           # 完整测试
python start.py api            # 启动API
python run_api.py              # 直接启动API
```

### 2. ModuleNotFoundError: No module named 'xxx'

**问题描述**: 缺少Python依赖包

**解决方案**:
```bash
# 1. 确保在正确的conda环境中
conda activate aideep2

# 2. 安装基础依赖
pip install -r requirements_basic.txt

# 3. 如果需要完整功能，安装所有依赖
pip install -r requirements.txt

# 4. 如果某个包安装失败，单独安装
pip install package_name
```

**常见缺失包及安装命令**:
```bash
# 数据库驱动
pip install asyncpg aiomysql redis

# 机器学习
pip install torch transformers sentence-transformers

# 文档处理
pip install langchain langchain-community

# 其他工具
pip install loguru python-dotenv pyyaml
```

### 3. Docker服务连接失败

**问题描述**: 无法连接到PostgreSQL、MySQL、Redis等服务

**解决方案**:
```bash
# 1. 检查Docker服务状态
docker ps

# 2. 启动所有服务
docker-compose up -d

# 3. 检查特定服务日志
docker logs rag2_postgres
docker logs rag2_mysql
docker logs rag2_redis

# 4. 重启服务
docker-compose restart

# 5. 如果端口冲突，修改docker-compose.yml中的端口映射
```

### 4. 模型加载失败

**问题描述**: 无法加载HuggingFace模型或Ollama模型

**解决方案**:

**HuggingFace模型**:
```bash
# 1. 检查网络连接
ping huggingface.co

# 2. 设置镜像源（如果在中国）
export HF_ENDPOINT=https://hf-mirror.com

# 3. 手动下载模型
python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-base-zh-v1.5')"

# 4. 如果内存不足，使用CPU设备
export MODEL_DEVICE=cpu
```

**Ollama模型**:
```bash
# 1. 检查Ollama服务
curl http://localhost:11434/api/version

# 2. 手动拉取模型
docker exec rag2_ollama ollama pull qwen2.5:7b

# 3. 检查模型列表
docker exec rag2_ollama ollama list
```

### 5. Mac M1相关问题

**问题描述**: 在Mac M1上运行出现性能或兼容性问题

**解决方案**:
```bash
# 1. 确保使用MPS设备
export MODEL_DEVICE=mps

# 2. 如果MPS不可用，降级到CPU
export MODEL_DEVICE=cpu

# 3. 安装Mac M1优化的PyTorch
pip install torch torchvision torchaudio

# 4. 检查MPS可用性
python -c "import torch; print(torch.backends.mps.is_available())"
```

### 6. API服务启动失败

**问题描述**: FastAPI服务无法启动

**解决方案**:
```bash
# 1. 检查端口占用
lsof -i :8000

# 2. 使用不同端口
export API_PORT=8001
python start.py api

# 3. 检查依赖
pip install fastapi uvicorn

# 4. 使用调试模式
python start.py api --debug
```

### 7. 内存不足

**问题描述**: 运行时内存不足

**解决方案**:
```bash
# 1. 使用开发环境配置（小模型）
export RAG_ENV=development

# 2. 减少批处理大小
export BATCH_SIZE=8

# 3. 使用CPU而不是GPU
export MODEL_DEVICE=cpu

# 4. 关闭其他应用程序释放内存
```

### 8. 权限问题

**问题描述**: 文件或目录权限不足

**解决方案**:
```bash
# 1. 检查项目目录权限
ls -la

# 2. 修复权限
chmod -R 755 rag2/
chmod +x *.py

# 3. 确保数据目录可写
mkdir -p data/logs data/documents data/temp
chmod -R 777 data/
```

## 🔧 调试技巧

### 1. 启用详细日志
```bash
export LOG_LEVEL=DEBUG
python start.py test
```

### 2. 检查环境变量
```bash
python -c "import os; print({k:v for k,v in os.environ.items() if 'RAG' in k or 'MODEL' in k})"
```

### 3. 测试单个组件
```bash
# 测试配置
python -c "from config.config import get_config; print(get_config())"

# 测试日志
python -c "from utils.logger import get_logger; get_logger('test').info('test')"

# 测试数据库连接
python -c "import asyncio; from storage.redis_manager import RedisManager; asyncio.run(RedisManager().health_check())"
```

### 4. 逐步安装依赖
```bash
# 1. 基础依赖
pip install fastapi uvicorn loguru

# 2. 数据库
pip install asyncpg aiomysql redis

# 3. 机器学习
pip install torch numpy

# 4. 文本处理
pip install transformers sentence-transformers

# 5. 其他
pip install langchain python-dotenv pyyaml
```

## 📞 获取帮助

### 1. 运行诊断脚本
```bash
python test_simple.py          # 基础诊断
python check_project_status.py # 项目状态检查
python start.py test           # 完整测试
```

### 2. 查看日志
```bash
# 应用日志
tail -f data/logs/rag2.log

# API访问日志
tail -f data/logs/api_access.log

# Docker日志
docker-compose logs -f
```

### 3. 检查系统资源
```bash
# 内存使用
free -h

# 磁盘空间
df -h

# CPU使用
top
```

### 4. 环境信息收集
```bash
# Python环境
python --version
pip list | grep -E "(torch|transformers|fastapi|langchain)"

# 系统信息
uname -a

# Docker信息
docker --version
docker-compose --version
```

## 🆘 紧急恢复

如果项目完全无法运行，按以下步骤重新设置：

```bash
# 1. 停止所有服务
docker-compose down

# 2. 清理Python缓存
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# 3. 重新安装依赖
pip uninstall -y -r requirements.txt
pip install -r requirements_basic.txt

# 4. 重新启动Docker服务
docker-compose up -d

# 5. 运行简单测试
python test_simple.py

# 6. 逐步恢复功能
python start.py test
```

记住：**先运行简单测试，确保基础环境正常，再逐步添加复杂功能！**
