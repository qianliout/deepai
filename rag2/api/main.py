"""
RAG2 FastAPI应用主入口
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ..config.config import get_config
    from ..utils.logger import get_logger, get_api_logger, log_api_access
    from ..core.rag_pipeline import get_rag_pipeline
    from ..storage.redis_manager import RedisManager
    from ..storage.mysql_manager import MySQLManager
    from .routes.query import router as query_router
    from .routes.document import router as document_router
    from .routes.admin import router as admin_router
except ImportError:
    # 绝对导入（当作为脚本运行时）
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from config.config import get_config
    from utils.logger import get_logger, get_api_logger, log_api_access
    from core.rag_pipeline import get_rag_pipeline
    from storage.redis_manager import RedisManager
    from storage.mysql_manager import MySQLManager
    from api.routes.query import router as query_router
    from api.routes.document import router as document_router
    from api.routes.admin import router as admin_router

logger = get_logger("api_main")
api_logger = get_api_logger()

# 全局变量
rag_pipeline = None
redis_manager = None
mysql_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("🚀 RAG2 API服务启动中...")
    
    global rag_pipeline, redis_manager, mysql_manager
    
    try:
        # 初始化组件
        rag_pipeline = await get_rag_pipeline()
        redis_manager = RedisManager()
        await redis_manager.initialize()
        mysql_manager = MySQLManager()
        await mysql_manager.initialize()
        
        logger.info("✅ 所有组件初始化完成")
        
        # 健康检查
        health_status = await rag_pipeline.health_check()
        logger.info(f"📊 组件健康状态: {health_status}")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ 应用启动失败: {str(e)}")
        raise
    
    finally:
        # 关闭时清理
        logger.info("🔄 RAG2 API服务关闭中...")
        
        if redis_manager:
            await redis_manager.close()
        if mysql_manager:
            await mysql_manager.close()
        
        logger.info("✅ 资源清理完成")

# 创建FastAPI应用
config = get_config()

app = FastAPI(
    title="RAG2 AIOps Assistant API",
    description="基于RAG的智能运维助手API服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(query_router, prefix="/api/v1/query", tags=["查询"])
app.include_router(document_router, prefix="/api/v1/document", tags=["文档管理"])
app.include_router(admin_router, prefix="/api/v1/admin", tags=["系统管理"])

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RAG2 AIOps Assistant API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        if rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG管道未初始化")
        
        health_status = await rag_pipeline.health_check()
        
        # 计算整体健康状态
        all_healthy = True
        for component, status in health_status.items():
            if isinstance(status, dict):
                # 检索器状态
                if not any(status.values()):
                    all_healthy = False
            elif not status:
                all_healthy = False
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": health_status,
            "timestamp": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=503, detail=f"健康检查失败: {str(e)}")

@app.get("/info")
async def get_info():
    """获取系统信息"""
    try:
        return {
            "environment": config.environment,
            "debug": config.debug,
            "models": config.models,
            "rag_config": {
                "retrieval_top_k": config.rag.retrieval_top_k,
                "rerank_top_k": config.rag.rerank_top_k,
                "max_context_length": config.rag.max_context_length,
                "chunk_size": config.rag.chunk_size
            }
        }
    except Exception as e:
        logger.error(f"获取系统信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统信息失败: {str(e)}")

# 依赖注入
async def get_rag_pipeline_dependency():
    """获取RAG管道依赖"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG管道未初始化")
    return rag_pipeline

async def get_redis_manager_dependency():
    """获取Redis管理器依赖"""
    if redis_manager is None:
        raise HTTPException(status_code=503, detail="Redis管理器未初始化")
    return redis_manager

async def get_mysql_manager_dependency():
    """获取MySQL管理器依赖"""
    if mysql_manager is None:
        raise HTTPException(status_code=503, detail="MySQL管理器未初始化")
    return mysql_manager

# 中间件
@app.middleware("http")
async def log_requests(request, call_next):
    """请求日志中间件"""
    import time
    
    start_time = time.time()
    
    # 处理请求
    response = await call_next(request)
    
    # 记录访问日志
    process_time = time.time() - start_time
    
    log_api_access(
        method=request.method,
        path=str(request.url.path),
        status_code=response.status_code,
        duration=process_time,
        user_id=request.headers.get("X-User-ID")
    )
    
    # 添加响应头
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# 异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    
    return {
        "error": "内部服务器错误",
        "message": str(exc) if config.debug else "服务暂时不可用，请稍后重试",
        "path": str(request.url.path),
        "method": request.method
    }

def run_server():
    """运行服务器"""
    uvicorn.run(
        "api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers if not config.debug else 1,
        log_level=config.api.log_level,
        access_log=False  # 使用自定义日志中间件
    )

if __name__ == "__main__":
    run_server()
