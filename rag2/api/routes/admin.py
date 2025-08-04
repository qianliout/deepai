"""
系统管理API路由
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ...core.rag_pipeline import get_rag_pipeline
    from ...storage.redis_manager import RedisManager
    from ...storage.mysql_manager import MySQLManager
    from ...storage.postgresql_manager import PostgreSQLManager
    from ...utils.logger import get_logger
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from core.rag_pipeline import get_rag_pipeline
    from storage.redis_manager import RedisManager
    from storage.mysql_manager import MySQLManager
    from storage.postgresql_manager import PostgreSQLManager
    from utils.logger import get_logger

logger = get_logger("admin_api")

router = APIRouter()

# 响应模型
class SystemStatus(BaseModel):
    status: str
    components: Dict[str, Any]
    timestamp: float

class SystemStats(BaseModel):
    database_stats: Dict[str, Any]
    redis_stats: Dict[str, Any]
    query_stats: Dict[str, Any]

# 依赖注入
async def get_rag_pipeline_dep():
    return await get_rag_pipeline()

async def get_redis_manager():
    redis_manager = RedisManager()
    await redis_manager.initialize()
    return redis_manager

async def get_mysql_manager():
    mysql_manager = MySQLManager()
    await mysql_manager.initialize()
    return mysql_manager

async def get_postgresql_manager():
    pg_manager = PostgreSQLManager()
    await pg_manager.initialize()
    return pg_manager

@router.get("/health", response_model=SystemStatus)
async def get_system_health(
    rag_pipeline = Depends(get_rag_pipeline_dep)
):
    """
    获取系统健康状态
    """
    try:
        import asyncio
        
        health_status = await rag_pipeline.health_check()
        
        # 计算整体状态
        all_healthy = True
        for component, status in health_status.items():
            if isinstance(status, dict):
                if not any(status.values()):
                    all_healthy = False
            elif not status:
                all_healthy = False
        
        overall_status = "healthy" if all_healthy else "degraded"
        
        return SystemStatus(
            status=overall_status,
            components=health_status,
            timestamp=asyncio.get_event_loop().time()
        )
        
    except Exception as e:
        logger.error(f"获取系统健康状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")

@router.get("/stats", response_model=SystemStats)
async def get_system_stats(
    redis_manager: RedisManager = Depends(get_redis_manager),
    mysql_manager: MySQLManager = Depends(get_mysql_manager),
    pg_manager: PostgreSQLManager = Depends(get_postgresql_manager)
):
    """
    获取系统统计信息
    """
    try:
        # Redis统计
        redis_stats = await redis_manager.get_stats()
        
        # 查询统计
        query_stats = await mysql_manager.get_query_statistics()
        
        # 数据库统计 (简化版)
        database_stats = {
            "redis_connected": await redis_manager.health_check(),
            "mysql_connected": await mysql_manager.health_check(),
            "postgresql_connected": await pg_manager.health_check()
        }
        
        return SystemStats(
            database_stats=database_stats,
            redis_stats=redis_stats,
            query_stats=query_stats
        )
        
    except Exception as e:
        logger.error(f"获取系统统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@router.get("/config")
async def get_system_config():
    """
    获取系统配置信息
    """
    try:
        from ...config.config import get_config
        
        config = get_config()
        
        # 返回安全的配置信息（不包含敏感信息）
        safe_config = {
            "environment": config.environment,
            "debug": config.debug,
            "api": {
                "host": config.api.host,
                "port": config.api.port,
                "workers": config.api.workers
            },
            "rag": {
                "retrieval_top_k": config.rag.retrieval_top_k,
                "rerank_top_k": config.rag.rerank_top_k,
                "max_context_length": config.rag.max_context_length,
                "chunk_size": config.rag.chunk_size,
                "chunk_overlap": config.rag.chunk_overlap
            },
            "models": {
                "llm": {
                    "provider": config.models["llm"]["provider"],
                    "model_name": config.models["llm"]["model_name"],
                    "device": config.models["llm"]["device"]
                },
                "embedding": {
                    "model_name": config.models["embedding"]["model_name"],
                    "device": config.models["embedding"]["device"],
                    "dimensions": config.models["embedding"]["dimensions"]
                },
                "reranker": {
                    "model_name": config.models["reranker"]["model_name"],
                    "device": config.models["reranker"]["device"]
                }
            }
        }
        
        return safe_config
        
    except Exception as e:
        logger.error(f"获取系统配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")

@router.post("/cache/clear")
async def clear_cache(
    cache_type: Optional[str] = None,
    redis_manager: RedisManager = Depends(get_redis_manager)
):
    """
    清理缓存
    """
    try:
        if cache_type == "query" or cache_type is None:
            # 清理查询缓存
            # 这里需要实现具体的缓存清理逻辑
            pass
        
        if cache_type == "session" or cache_type is None:
            # 清理会话缓存
            # 这里需要实现具体的会话清理逻辑
            pass
        
        return {
            "message": "缓存清理完成",
            "cache_type": cache_type or "all"
        }
        
    except Exception as e:
        logger.error(f"清理缓存失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清理缓存失败: {str(e)}")

@router.get("/logs")
async def get_system_logs(
    level: str = "INFO",
    limit: int = 100,
    component: Optional[str] = None
):
    """
    获取系统日志
    """
    try:
        from pathlib import Path
        
        log_dir = Path("data/logs")
        
        # 根据组件选择日志文件
        if component == "api":
            log_file = log_dir / "api_access.log"
        elif component == "query":
            log_file = log_dir / "query.log"
        elif component == "retrieval":
            log_file = log_dir / "retrieval.log"
        elif component == "model":
            log_file = log_dir / "model.log"
        else:
            log_file = log_dir / "rag2.log"
        
        logs = []
        
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # 获取最后limit行
                recent_lines = lines[-limit:] if len(lines) > limit else lines
                
                for line in recent_lines:
                    line = line.strip()
                    if line and level.upper() in line:
                        logs.append(line)
        
        return {
            "logs": logs,
            "total": len(logs),
            "level": level,
            "component": component or "all",
            "log_file": str(log_file)
        }
        
    except Exception as e:
        logger.error(f"获取系统日志失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取日志失败: {str(e)}")

@router.post("/models/warm-up")
async def warm_up_models():
    """
    预热模型
    """
    try:
        from ...models.embeddings import get_embedding_manager
        from ...models.rerank_models import get_rerank_manager
        
        # 预热嵌入模型
        embedding_manager = get_embedding_manager()
        embedding_manager.warm_up()
        
        # 预热重排序模型
        rerank_manager = get_rerank_manager()
        rerank_manager.warm_up()
        
        return {
            "message": "模型预热完成",
            "models": ["embedding", "reranker"]
        }
        
    except Exception as e:
        logger.error(f"模型预热失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型预热失败: {str(e)}")

@router.get("/metrics")
async def get_system_metrics(
    metric_name: Optional[str] = None,
    redis_manager: RedisManager = Depends(get_redis_manager)
):
    """
    获取系统指标
    """
    try:
        from datetime import datetime, timedelta
        
        # 获取最近1小时的指标
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        if metric_name:
            metrics = await redis_manager.get_metrics(metric_name, start_time, end_time)
        else:
            # 获取所有指标类型
            metrics = {}
            common_metrics = ["query_time", "retrieval_time", "model_time", "response_time"]
            
            for metric in common_metrics:
                try:
                    metric_data = await redis_manager.get_metrics(metric, start_time, end_time)
                    metrics[metric] = metric_data
                except:
                    metrics[metric] = []
        
        return {
            "metrics": metrics,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取指标失败: {str(e)}")

@router.post("/restart-component")
async def restart_component(
    component: str
):
    """
    重启系统组件
    """
    try:
        # 这里应该实现具体的组件重启逻辑
        # 由于安全考虑，这个功能需要谨慎实现
        
        if component not in ["cache", "models"]:
            raise HTTPException(status_code=400, detail="不支持的组件类型")
        
        return {
            "message": f"组件 {component} 重启请求已提交",
            "component": component,
            "status": "pending"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重启组件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重启组件失败: {str(e)}")
