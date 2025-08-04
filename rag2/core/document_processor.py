"""
文档处理器
负责文档加载、分块和预处理
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio

# 文档处理相关
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
import jieba

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ..config.config import get_config
    from ..utils.logger import get_logger, log_performance, LogContext
    from ..models.embeddings import get_embedding_manager, SemanticChunker
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from config.config import get_config
    from utils.logger import get_logger, log_performance, LogContext
    from models.embeddings import get_embedding_manager, SemanticChunker

logger = get_logger("document_processor")

class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self):
        self.config = get_config()
        self.embedding_manager = get_embedding_manager()
        
        # 初始化分词器
        jieba.initialize()
        
        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.rag.chunk_size,
            chunk_overlap=self.config.rag.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        # 语义分块器
        self.semantic_chunker = SemanticChunker(
            embedding_manager=self.embedding_manager,
            similarity_threshold=0.8
        )
        
        logger.info("文档处理器初始化完成")
    
    @log_performance()
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """加载单个文档"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 根据文件类型选择加载器
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
                content = "\n".join([doc.page_content for doc in documents])
            elif file_path.suffix.lower() in ['.txt', '.md']:
                loader = TextLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
                content = documents[0].page_content
            else:
                # 尝试作为文本文件读取
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # 提取元数据
            metadata = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "file_type": file_path.suffix.lower(),
                "created_time": file_path.stat().st_ctime,
                "modified_time": file_path.stat().st_mtime
            }
            
            document = {
                "id": str(uuid.uuid4()),
                "title": file_path.stem,
                "content": content,
                "source": str(file_path),
                "document_type": file_path.suffix.lower().replace('.', ''),
                "metadata": metadata
            }
            
            logger.info(f"文档加载成功: {file_path.name}, 长度: {len(content)}")
            return document
            
        except Exception as e:
            logger.error(f"文档加载失败: {file_path}, 错误: {str(e)}")
            raise
    
    @log_performance()
    def load_documents_from_directory(self, directory_path: str, 
                                    file_extensions: List[str] = None) -> List[Dict[str, Any]]:
        """从目录加载多个文档"""
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.pdf']
        
        directory_path = Path(directory_path)
        documents = []
        
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        # 递归查找文件
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                try:
                    document = self.load_document(str(file_path))
                    documents.append(document)
                except Exception as e:
                    logger.warning(f"跳过文件 {file_path}: {str(e)}")
        
        logger.info(f"从目录 {directory_path} 加载了 {len(documents)} 个文档")
        return documents
    
    @log_performance()
    def chunk_document(self, document: Dict[str, Any], 
                      use_semantic_chunking: bool = True) -> List[Dict[str, Any]]:
        """对文档进行分块"""
        try:
            content = document["content"]
            
            if use_semantic_chunking:
                # 使用语义分块
                chunks_info = self.semantic_chunker.chunk_text(
                    content, 
                    max_chunk_size=self.config.rag.chunk_size,
                    overlap_size=self.config.rag.chunk_overlap
                )
            else:
                # 使用传统分块
                text_chunks = self.text_splitter.split_text(content)
                chunks_info = []
                for i, chunk_content in enumerate(text_chunks):
                    chunks_info.append({
                        "content": chunk_content,
                        "start_index": i * self.config.rag.chunk_size,
                        "end_index": (i + 1) * self.config.rag.chunk_size,
                        "sentence_count": chunk_content.count('。') + chunk_content.count('！') + chunk_content.count('？') + 1
                    })
            
            # 创建分块对象
            chunks = []
            for i, chunk_info in enumerate(chunks_info):
                chunk = {
                    "id": str(uuid.uuid4()),
                    "document_id": document["id"],
                    "chunk_index": i,
                    "content": chunk_info["content"],
                    "token_count": len(chunk_info["content"]) // 4,  # 粗略估算
                    "chunk_metadata": {
                        "start_index": chunk_info.get("start_index", 0),
                        "end_index": chunk_info.get("end_index", len(chunk_info["content"])),
                        "sentence_count": chunk_info.get("sentence_count", 1),
                        "chunking_method": "semantic" if use_semantic_chunking else "recursive"
                    }
                }
                chunks.append(chunk)
            
            logger.info(f"文档分块完成: {document['title']}, 分块数: {len(chunks)}")
            return chunks
            
        except Exception as e:
            logger.error(f"文档分块失败: {document.get('title', 'Unknown')}, 错误: {str(e)}")
            raise
    
    @log_performance()
    async def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为分块生成嵌入向量"""
        try:
            # 提取文本内容
            texts = [chunk["content"] for chunk in chunks]
            
            # 批量生成嵌入
            embeddings = await self.embedding_manager.encode_texts_async(texts)
            
            # 将嵌入添加到分块中
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            
            logger.info(f"嵌入生成完成: {len(chunks)} 个分块")
            return chunks
            
        except Exception as e:
            logger.error(f"嵌入生成失败: {str(e)}")
            raise
    
    @log_performance()
    async def process_document(self, file_path: str, 
                             use_semantic_chunking: bool = True) -> Dict[str, Any]:
        """完整处理单个文档"""
        with LogContext("文档处理", file_path=file_path):
            # 1. 加载文档
            document = self.load_document(file_path)
            
            # 2. 分块
            chunks = self.chunk_document(document, use_semantic_chunking)
            
            # 3. 生成嵌入
            chunks_with_embeddings = await self.generate_embeddings(chunks)
            
            # 4. 组装结果
            result = {
                "document": document,
                "chunks": chunks_with_embeddings,
                "total_chunks": len(chunks_with_embeddings),
                "total_tokens": sum(chunk["token_count"] for chunk in chunks_with_embeddings)
            }
            
            logger.info(f"文档处理完成: {document['title']}")
            return result
    
    @log_performance()
    async def process_documents_batch(self, file_paths: List[str], 
                                    use_semantic_chunking: bool = True,
                                    batch_size: int = 5) -> List[Dict[str, Any]]:
        """批量处理多个文档"""
        results = []
        
        # 分批处理以避免内存过载
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]
            
            # 并发处理当前批次
            batch_tasks = [
                self.process_document(path, use_semantic_chunking) 
                for path in batch_paths
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理结果和异常
            for path, result in zip(batch_paths, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"文档处理失败: {path}, 错误: {str(result)}")
                else:
                    results.append(result)
        
        logger.info(f"批量文档处理完成: 成功 {len(results)} / 总计 {len(file_paths)}")
        return results
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式"""
        return ['.txt', '.md', '.pdf']
    
    def validate_document(self, document: Dict[str, Any]) -> bool:
        """验证文档格式"""
        required_fields = ['id', 'title', 'content', 'source']
        return all(field in document for field in required_fields)

# 便捷函数
async def process_single_document(file_path: str) -> Dict[str, Any]:
    """处理单个文档的便捷函数"""
    processor = DocumentProcessor()
    return await processor.process_document(file_path)

async def process_directory(directory_path: str) -> List[Dict[str, Any]]:
    """处理目录中所有文档的便捷函数"""
    processor = DocumentProcessor()
    documents = processor.load_documents_from_directory(directory_path)
    file_paths = [doc["source"] for doc in documents]
    return await processor.process_documents_batch(file_paths)
