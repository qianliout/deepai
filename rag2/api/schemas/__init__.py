"""
API数据模型定义
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# 基础响应模型
class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = True
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseResponse):
    """错误响应模型"""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# 查询相关模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="用户查询", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="会话ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    stream: bool = Field(False, description="是否流式返回")
    top_k: Optional[int] = Field(10, description="检索文档数量", ge=1, le=50)
    similarity_threshold: Optional[float] = Field(0.7, description="相似度阈值", ge=0.0, le=1.0)

class QueryResponse(BaseResponse):
    """查询响应模型"""
    query: str
    response: str
    session_id: Optional[str]
    retrieved_documents: List[Dict[str, Any]]
    context_used: bool
    processing_time_ms: float