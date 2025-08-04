"""
查询相关API路由
"""

import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ...core.rag_pipeline import RAGPipeline
    from ...storage.redis_manager import RedisManager
    from ...storage.mysql_manager import MySQLManager
    from ...utils.logger import get_logger
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from core.rag_pipeline import RAGPipeline
    from storage.redis_manager import RedisManager
    from storage.mysql_manager import MySQLManager
    from utils.logger import get_logger

logger = get_logger("query_api")

router = APIRouter()

# 请求模型
class QueryRequest(BaseModel):
    query: str = Field(..., description="用户查询", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="会话ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    stream: bool = Field(False, description="是否流式返回")
    top_k: Optional[int] = Field(10, description="检索文档数量", ge=1, le=50)
    similarity_threshold: Optional[float] = Field(0.7, description="相似度阈值", ge=0.0, le=1.0)

class SessionRequest(BaseModel):
    user_id: str = Field(..., description="用户ID")
    session_name: Optional[str] = Field(None, description="会话名称")

# 响应模型
class QueryResponse(BaseModel):
    query: str
    response: str
    session_id: Optional[str]
    retrieved_documents: list
    context_used: bool
    processing_time_ms: float

class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    session_name: str
    created_at: str

class ConversationMessage(BaseModel):
    role: str
    content: str
    timestamp: str

# 依赖注入
async def get_rag_pipeline():
    from ...api.main import rag_pipeline
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG管道未初始化")
    return rag_pipeline

async def get_redis_manager():
    from ...api.main import redis_manager
    if redis_manager is None:
        raise HTTPException(status_code=503, detail="Redis管理器未初始化")
    return redis_manager

async def get_mysql_manager():
    from ...api.main import mysql_manager
    if mysql_manager is None:
        raise HTTPException(status_code=503, detail="MySQL管理器未初始化")
    return mysql_manager

@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    提问接口
    """
    try:
        # 如果没有提供session_id，创建新会话
        session_id = request.session_id
        if not session_id and request.user_id:
            session_id = str(uuid.uuid4())
        
        # 执行RAG查询
        result = await rag_pipeline.query(
            query=request.query,
            session_id=session_id,
            user_id=request.user_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"查询处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")

@router.post("/ask/stream")
async def ask_question_stream(
    request: QueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    流式提问接口
    """
    try:
        # 如果没有提供session_id，创建新会话
        session_id = request.session_id
        if not session_id and request.user_id:
            session_id = str(uuid.uuid4())
        
        async def generate_response():
            try:
                async for chunk in rag_pipeline.stream_query(
                    query=request.query,
                    session_id=session_id,
                    user_id=request.user_id,
                    top_k=request.top_k,
                    similarity_threshold=request.similarity_threshold
                ):
                    yield f"data: {chunk}\n\n"
                
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"流式查询失败: {str(e)}")
                yield f"data: ERROR: {str(e)}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-ID": session_id or ""
            }
        )
        
    except Exception as e:
        logger.error(f"流式查询处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"流式查询处理失败: {str(e)}")

@router.post("/session", response_model=SessionResponse)
async def create_session(
    request: SessionRequest,
    redis_manager: RedisManager = Depends(get_redis_manager),
    mysql_manager: MySQLManager = Depends(get_mysql_manager)
):
    """
    创建新会话
    """
    try:
        # 在Redis中创建会话
        session_id = await redis_manager.create_session(
            user_id=request.user_id,
            session_name=request.session_name
        )
        
        # 在MySQL中创建会话记录
        await mysql_manager.create_session(
            user_id=request.user_id,
            session_name=request.session_name or f"Session_{session_id[:8]}"
        )
        
        # 获取会话信息
        session_info = await redis_manager.get_session(session_id)
        
        return SessionResponse(
            session_id=session_id,
            user_id=request.user_id,
            session_name=session_info.get("session_name", ""),
            created_at=session_info.get("created_at", "")
        )
        
    except Exception as e:
        logger.error(f"创建会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")

@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    redis_manager: RedisManager = Depends(get_redis_manager)
):
    """
    获取会话信息
    """
    try:
        session_info = await redis_manager.get_session(session_id)
        
        if not session_info:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        return session_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取会话失败: {str(e)}")

@router.get("/session/{session_id}/history")
async def get_conversation_history(
    session_id: str,
    limit: int = 20,
    redis_manager: RedisManager = Depends(get_redis_manager)
):
    """
    获取对话历史
    """
    try:
        messages = await redis_manager.get_conversation_context(session_id, limit)
        
        conversation = []
        for msg in messages:
            conversation.append(ConversationMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"]
            ))
        
        return {
            "session_id": session_id,
            "messages": conversation,
            "total": len(conversation)
        }
        
    except Exception as e:
        logger.error(f"获取对话历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取对话历史失败: {str(e)}")

@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    redis_manager: RedisManager = Depends(get_redis_manager)
):
    """
    删除会话
    """
    try:
        await redis_manager.delete_session(session_id)
        
        return {"message": "会话删除成功", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"删除会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")

@router.get("/user/{user_id}/sessions")
async def get_user_sessions(
    user_id: str,
    mysql_manager: MySQLManager = Depends(get_mysql_manager)
):
    """
    获取用户的所有会话
    """
    try:
        sessions = await mysql_manager.get_user_sessions(user_id)
        
        return {
            "user_id": user_id,
            "sessions": sessions,
            "total": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"获取用户会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取用户会话失败: {str(e)}")

@router.post("/feedback")
async def submit_feedback(
    session_id: Optional[str] = None,
    query_log_id: Optional[str] = None,
    feedback_type: str = "rating",
    rating: Optional[int] = None,
    comment: Optional[str] = None,
    mysql_manager: MySQLManager = Depends(get_mysql_manager)
):
    """
    提交用户反馈
    """
    try:
        feedback_id = await mysql_manager.add_feedback(
            session_id=session_id,
            query_log_id=query_log_id,
            feedback_type=feedback_type,
            rating=rating,
            comment=comment
        )
        
        return {
            "message": "反馈提交成功",
            "feedback_id": feedback_id
        }
        
    except Exception as e:
        logger.error(f"提交反馈失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")
