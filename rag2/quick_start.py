#!/usr/bin/env python3
"""
RAG2项目快速启动脚本
一键启动和测试RAG2系统
"""

import asyncio
import sys
import os
import time
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import get_config
from utils.logger import get_logger

logger = get_logger("quick_start")

class QuickStarter:
    """快速启动器"""
    
    def __init__(self):
        self.config = get_config()
        self.project_root = Path(__file__).parent
    
    def check_prerequisites(self):
        """检查前置条件"""
        logger.info("🔍 检查前置条件...")
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version < (3, 9):
            logger.error(f"Python版本过低: {python_version.major}.{python_version.minor}, 需要3.9+")
            return False
        
        logger.info(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # 检查Docker
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Docker: {result.stdout.strip()}")
            else:
                logger.error("❌ Docker未安装或不可用")
                return False
        except FileNotFoundError:
            logger.error("❌ Docker未安装")
            return False
        
        # 检查Docker Compose
        try:
            result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Docker Compose: {result.stdout.strip()}")
            else:
                logger.error("❌ Docker Compose未安装或不可用")
                return False
        except FileNotFoundError:
            logger.error("❌ Docker Compose未安装")
            return False
        
        # 检查conda环境
        conda_env = os.environ.get("CONDA_DEFAULT_ENV")
        if conda_env:
            logger.info(f"✅ Conda环境: {conda_env}")
        else:
            logger.warning("⚠️  未检测到Conda环境")
        
        return True
    
    def setup_environment(self):
        """设置环境"""
        logger.info("⚙️  设置环境...")
        
        # 创建.env文件
        env_file = self.project_root / ".env"
        if not env_file.exists():
            env_content = f"""# RAG2环境配置
RAG_ENV=development

# 模型设备 (Mac M1优化)
MODEL_DEVICE=mps

# API配置
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true

# DeepSeek API (可选，用于生产环境)
# DEEPSEEK_API_KEY=your-api-key-here

# 数据库配置 (使用Docker默认值)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
MYSQL_HOST=localhost
MYSQL_PORT=3306
REDIS_HOST=localhost
REDIS_PORT=6379
ES_HOST=localhost
ES_PORT=9200
NEO4J_URI=bolt://localhost:7687
"""
            with open(env_file, 'w') as f:
                f.write(env_content)
            logger.info("✅ 创建.env配置文件")
        else:
            logger.info("✅ .env配置文件已存在")
        
        # 创建数据目录
        data_dirs = [
            "data/logs",
            "data/documents/aiops_knowledge",
            "data/documents/technical_docs",
            "data/mock/structured"
        ]
        
        for dir_path in data_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ 数据目录创建完成")
    
    def start_databases(self):
        """启动数据库服务"""
        logger.info("🚀 启动数据库服务...")
        
        try:
            # 启动核心数据库
            result = subprocess.run([
                "docker-compose", "up", "-d", 
                "postgres", "mysql", "redis"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ 核心数据库启动成功")
            else:
                logger.error(f"❌ 数据库启动失败: {result.stderr}")
                return False
            
            # 等待数据库就绪
            logger.info("⏳ 等待数据库就绪...")
            time.sleep(10)
            
            # 启动Ollama (开发环境LLM)
            logger.info("🚀 启动Ollama服务...")
            result = subprocess.run([
                "docker-compose", "up", "-d", "ollama"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Ollama服务启动成功")
            else:
                logger.warning(f"⚠️  Ollama启动失败: {result.stderr}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 启动数据库服务失败: {str(e)}")
            return False
    
    def install_dependencies(self):
        """安装Python依赖"""
        logger.info("📦 安装Python依赖...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ 依赖安装成功")
                return True
            else:
                logger.error(f"❌ 依赖安装失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 依赖安装异常: {str(e)}")
            return False
    
    async def run_tests(self):
        """运行测试"""
        logger.info("🧪 运行系统测试...")
        
        try:
            # 运行基础测试
            from test_basic_setup import main as basic_test
            basic_result = await basic_test()
            
            if basic_result == 0:
                logger.info("✅ 基础测试通过")
            else:
                logger.warning("⚠️  基础测试部分失败")
            
            # 运行完整测试
            from test_complete_system import main as complete_test
            complete_result = await complete_test()
            
            if complete_result == 0:
                logger.info("✅ 完整测试通过")
                return True
            else:
                logger.warning("⚠️  完整测试部分失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ 测试运行失败: {str(e)}")
            return False
    
    def create_sample_documents(self):
        """创建示例文档"""
        logger.info("📄 创建示例文档...")
        
        # AIOps安全文档
        security_doc = self.project_root / "data/documents/aiops_knowledge/security_guide.txt"
        if not security_doc.exists():
            content = """AIOps安全最佳实践指南

1. 容器安全
容器安全是现代AIOps环境中的关键组成部分。

1.1 镜像安全
- 使用官方或可信的基础镜像
- 定期扫描镜像漏洞，建议使用Trivy、Clair或Snyk等工具
- 实施镜像签名验证机制

1.2 运行时安全
- 以非root用户运行容器
- 使用只读文件系统
- 限制容器的系统调用权限

2. 漏洞管理
建立自动化漏洞扫描流程，集成CI/CD管道中的安全检查。

2.1 漏洞扫描
- 定期扫描容器镜像
- 监控CVE数据库更新
- 建立漏洞修复优先级

2.2 应急响应
- 制定漏洞应急响应流程
- 建立快速修复机制
- 定期演练应急预案

3. 访问控制
实施严格的访问控制策略，确保只有授权用户能够访问关键资源。

CVE-2024-1234是一个影响nginx:latest镜像的高危漏洞。
建议立即更新到nginx:1.25.3以修复此漏洞。
"""
            with open(security_doc, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info("✅ 创建安全指南文档")
        
        # 运维手册
        ops_doc = self.project_root / "data/documents/aiops_knowledge/ops_manual.txt"
        if not ops_doc.exists():
            content = """AIOps运维操作手册

1. 日常监控
建立全面的监控体系，及时发现和处理问题。

1.1 基础监控
- CPU、内存、磁盘使用率监控
- 网络流量和连接数监控
- 应用程序性能监控

1.2 业务监控
- 关键业务指标监控
- 用户体验监控
- 服务可用性监控

2. 故障处理
建立标准化的故障处理流程。

2.1 故障发现
- 自动告警机制
- 主动巡检
- 用户反馈

2.2 故障处理
- 快速定位问题
- 制定解决方案
- 实施修复措施
- 验证修复效果

3. 容量规划
根据业务增长预测，合理规划资源容量。

主机192.168.1.100运行的nginx容器存在内存泄漏问题。
建议重启容器并升级到最新版本。
"""
            with open(ops_doc, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info("✅ 创建运维手册文档")
    
    def show_next_steps(self):
        """显示后续步骤"""
        logger.info("\n" + "=" * 60)
        logger.info("🎉 RAG2系统启动完成！")
        logger.info("=" * 60)
        
        logger.info("\n📋 后续步骤:")
        logger.info("1. 启动API服务:")
        logger.info("   python -m api.main")
        logger.info("   或")
        logger.info("   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
        
        logger.info("\n2. 访问API文档:")
        logger.info("   http://localhost:8000/docs")
        
        logger.info("\n3. 测试API:")
        logger.info("   curl http://localhost:8000/health")
        
        logger.info("\n4. 处理文档:")
        logger.info("   python -c \"import asyncio; from core.document_processor import process_directory; asyncio.run(process_directory('data/documents'))\"")
        
        logger.info("\n5. 测试查询:")
        logger.info("   curl -X POST http://localhost:8000/api/v1/query/ask \\")
        logger.info("     -H 'Content-Type: application/json' \\")
        logger.info("     -d '{\"query\": \"什么是容器安全？\", \"user_id\": \"test_user\"}'")
        
        logger.info("\n📚 更多信息请查看 README.md")
    
    async def run(self):
        """运行快速启动流程"""
        logger.info("🚀 RAG2项目快速启动")
        logger.info("=" * 50)
        
        # 1. 检查前置条件
        if not self.check_prerequisites():
            logger.error("❌ 前置条件检查失败，请解决后重试")
            return 1
        
        # 2. 设置环境
        self.setup_environment()
        
        # 3. 启动数据库
        if not self.start_databases():
            logger.error("❌ 数据库启动失败")
            return 1
        
        # 4. 安装依赖
        if not self.install_dependencies():
            logger.error("❌ 依赖安装失败")
            return 1
        
        # 5. 创建示例文档
        self.create_sample_documents()
        
        # 6. 运行测试
        test_success = await self.run_tests()
        
        # 7. 显示后续步骤
        self.show_next_steps()
        
        if test_success:
            logger.info("\n✅ 系统启动成功，所有测试通过！")
            return 0
        else:
            logger.warning("\n⚠️  系统启动完成，但部分测试失败")
            return 1

async def main():
    """主函数"""
    starter = QuickStarter()
    return await starter.run()

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("启动被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"启动过程中发生错误: {str(e)}")
        sys.exit(1)
