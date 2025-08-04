#!/usr/bin/env python3
"""
RAG2简化API启动脚本
只启动基础API功能，避免复杂依赖
"""

import sys
import os
from pathlib import Path

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def create_simple_api():
    """创建简化的API应用"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    app = FastAPI(
        title="RAG2 Simple API",
        description="RAG2项目简化API服务",
        version="1.0.0"
    )
    
    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        """根路径"""
        return {
            "message": "RAG2 Simple API",
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health_check():
        """健康检查"""
        try:
            # 基础健康检查
            from config.config import get_config
            config = get_config()
            
            return {
                "status": "healthy",
                "environment": config.environment,
                "timestamp": "2025-08-04T14:30:00Z"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "2025-08-04T14:30:00Z"
            }
    
    @app.get("/info")
    async def get_info():
        """获取系统信息"""
        try:
            from config.config import get_config
            config = get_config()
            
            return {
                "environment": config.environment,
                "debug": config.debug,
                "python_version": sys.version,
                "project_root": str(PROJECT_ROOT)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/test/echo")
    async def test_echo(data: dict):
        """测试回显接口"""
        return {
            "echo": data,
            "message": "API正常工作"
        }
    
    return app

def main():
    """主函数"""
    print("🚀 启动RAG2简化API服务...")
    
    try:
        # 检查基础依赖
        import fastapi
        import uvicorn
        print("✅ FastAPI和Uvicorn可用")
        
        # 创建应用
        app = create_simple_api()
        
        # 启动服务
        print("📍 地址: http://localhost:8000")
        print("📚 API文档: http://localhost:8000/docs")
        print("🔧 这是简化版API，只包含基础功能")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,  # 禁用reload避免警告
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请安装: pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
