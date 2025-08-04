"""
文档管理API路由
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ...core.document_processor import DocumentProcessor
    from ...retrieval.semantic_retriever import SemanticRetriever
    from ...utils.logger import get_logger
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from core.document_processor import DocumentProcessor
    from retrieval.semantic_retriever import SemanticRetriever
    from utils.logger import get_logger

logger = get_logger("document_api")

router = APIRouter()

# 请求模型
class DocumentUploadResponse(BaseModel):
    document_id: str
    title: str
    chunks_count: int
    processing_time_ms: float
    status: str

class DocumentListResponse(BaseModel):
    documents: List[dict]
    total: int

# 依赖注入
async def get_document_processor():
    return DocumentProcessor()

async def get_semantic_retriever():
    return SemanticRetriever()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    processor: DocumentProcessor = Depends(get_document_processor),
    retriever: SemanticRetriever = Depends(get_semantic_retriever)
):
    """
    上传并处理文档
    """
    try:
        import tempfile
        import time
        
        start_time = time.time()
        
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 处理文档
            result = await processor.process_document(tmp_file_path)
            
            # 添加到检索索引
            await retriever.add_documents([result])
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            return DocumentUploadResponse(
                document_id=result["document"]["id"],
                title=result["document"]["title"],
                chunks_count=result["total_chunks"],
                processing_time_ms=processing_time,
                status="success"
            )
            
        finally:
            # 清理临时文件
            import os
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"文档上传处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

@router.post("/process-directory")
async def process_directory(
    directory_path: str,
    processor: DocumentProcessor = Depends(get_document_processor),
    retriever: SemanticRetriever = Depends(get_semantic_retriever)
):
    """
    处理目录中的所有文档
    """
    try:
        import time
        from pathlib import Path
        
        start_time = time.time()
        
        # 检查目录是否存在
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise HTTPException(status_code=404, detail="目录不存在")
        
        # 加载目录中的文档
        documents = processor.load_documents_from_directory(directory_path)
        
        if not documents:
            return {
                "message": "目录中没有找到支持的文档",
                "processed_count": 0,
                "total_time_ms": 0
            }
        
        # 批量处理文档
        file_paths = [doc["source"] for doc in documents]
        results = await processor.process_documents_batch(file_paths)
        
        # 添加到检索索引
        if results:
            await retriever.add_documents(results)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        return {
            "message": "目录处理完成",
            "processed_count": len(results),
            "total_chunks": sum(r["total_chunks"] for r in results),
            "total_time_ms": processing_time,
            "documents": [
                {
                    "id": r["document"]["id"],
                    "title": r["document"]["title"],
                    "chunks_count": r["total_chunks"]
                }
                for r in results
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"目录处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"目录处理失败: {str(e)}")

@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    retriever: SemanticRetriever = Depends(get_semantic_retriever)
):
    """
    获取文档列表
    """
    try:
        # 获取文档统计信息
        stats = await retriever.get_document_stats()
        
        # 这里简化处理，实际应该从数据库查询文档列表
        # 由于当前的检索器没有提供列表接口，我们返回统计信息
        
        return DocumentListResponse(
            documents=[],  # 实际应该查询文档列表
            total=stats.get("total_documents", 0)
        )
        
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

@router.get("/stats")
async def get_document_stats(
    retriever: SemanticRetriever = Depends(get_semantic_retriever)
):
    """
    获取文档统计信息
    """
    try:
        stats = await retriever.get_document_stats()
        return stats
        
    except Exception as e:
        logger.error(f"获取文档统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档统计失败: {str(e)}")

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    retriever: SemanticRetriever = Depends(get_semantic_retriever)
):
    """
    删除文档
    """
    try:
        success = await retriever.delete_document(document_id)
        
        if success:
            return {"message": "文档删除成功", "document_id": document_id}
        else:
            raise HTTPException(status_code=404, detail="文档不存在或删除失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")

@router.post("/search")
async def search_documents(
    query: str,
    top_k: int = 10,
    similarity_threshold: float = 0.7,
    retriever: SemanticRetriever = Depends(get_semantic_retriever)
):
    """
    搜索文档
    """
    try:
        results = await retriever.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        return {
            "query": query,
            "results": results,
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"文档搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档搜索失败: {str(e)}")

@router.get("/supported-formats")
async def get_supported_formats(
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    获取支持的文件格式
    """
    try:
        formats = processor.get_supported_formats()
        return {
            "supported_formats": formats,
            "description": "支持的文档格式列表"
        }
        
    except Exception as e:
        logger.error(f"获取支持格式失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取支持格式失败: {str(e)}")
