#!/usr/bin/env python3
"""
RAG2 API服务启动脚本
解决相对导入问题的启动入口
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ.setdefault('PYTHONPATH', str(project_root))

def main():
    """主函数"""
    try:
        import uvicorn
        from api.main import app
        from config.config import get_config
        
        config = get_config()
        
        print("🚀 启动RAG2 API服务...")
        print(f"📍 地址: http://{config.api.host}:{config.api.port}")
        print(f"📚 API文档: http://{config.api.host}:{config.api.port}/docs")
        print(f"🔧 环境: {config.environment}")
        
        uvicorn.run(
            "api.main:app",
            host=config.api.host,
            port=config.api.port,
            reload=config.api.reload,
            workers=config.api.workers if not config.debug else 1,
            log_level=config.api.log_level.lower(),
            access_log=False  # 使用自定义日志中间件
        )
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
