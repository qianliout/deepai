"""
嵌入模型管理器
统一管理BAAI/BGE嵌入模型
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ..config.config import get_config, get_model_config
    from ..utils.logger import get_logger, log_performance, log_model_call
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from config.config import get_config, get_model_config
    from utils.logger import get_logger, log_performance, log_model_call

logger = get_logger("embeddings")

class EmbeddingManager:
    """嵌入模型管理器"""
    
    def __init__(self):
        self.config = get_config()
        self.model_config = get_model_config()
        
        self.embedding_model = None
        self.model_name = None
        self.device = None
        self.dimensions = None
        
        # 初始化模型
        self._init_model()
    
    def _init_model(self):
        """初始化嵌入模型"""
        embedding_config = self.model_config["embedding"]
        
        self.model_name = embedding_config["model_name"]
        self.device = embedding_config.get("device", "cpu")
        self.dimensions = embedding_config.get("dimensions", 768)
        
        try:
            # 检查设备可用性
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA不可用，切换到CPU")
                self.device = "cpu"
            
            # 加载模型
            self.embedding_model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # 设置模型为评估模式
            self.embedding_model.eval()
            
            logger.info(f"嵌入模型加载成功: {self.model_name} (设备: {self.device})")
            
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {str(e)}")
            raise
    
    @log_performance()
    def encode_text(self, text: str, normalize: bool = True) -> List[float]:
        """编码单个文本"""
        try:
            start_time = time.time()
            
            # 编码文本
            embedding = self.embedding_model.encode(
                text,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            
            # 记录模型调用
            end_time = time.time()
            duration = end_time - start_time
            
            log_model_call(
                model_name=self.model_name,
                input_tokens=len(text) // 4,  # 粗略估算
                output_tokens=len(embedding),
                duration=duration
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"文本编码失败: {str(e)}")
            raise
    
    @log_performance()
    def encode_texts(self, texts: List[str], normalize: bool = True, 
                    batch_size: int = 32) -> List[List[float]]:
        """批量编码文本"""
        try:
            start_time = time.time()
            
            # 批量编码
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100
            )
            
            # 记录模型调用
            end_time = time.time()
            duration = end_time - start_time
            
            total_input_tokens = sum(len(text) // 4 for text in texts)
            total_output_tokens = len(embeddings) * len(embeddings[0]) if len(embeddings) > 0 else 0
            
            log_model_call(
                model_name=self.model_name,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                duration=duration
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"批量文本编码失败: {str(e)}")
            raise
    
    async def encode_text_async(self, text: str, normalize: bool = True) -> List[float]:
        """异步编码单个文本"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode_text, text, normalize)
    
    async def encode_texts_async(self, texts: List[str], normalize: bool = True, 
                               batch_size: int = 32) -> List[List[float]]:
        """异步批量编码文本"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode_texts, texts, normalize, batch_size)
    
    def compute_similarity(self, embedding1: List[float], 
                          embedding2: List[float]) -> float:
        """计算两个嵌入向量的余弦相似度"""
        try:
            # 转换为numpy数组
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {str(e)}")
            return 0.0
    
    def compute_similarities(self, query_embedding: List[float], 
                           candidate_embeddings: List[List[float]]) -> List[float]:
        """计算查询向量与候选向量列表的相似度"""
        try:
            query_vec = np.array(query_embedding)
            candidate_vecs = np.array(candidate_embeddings)
            
            # 批量计算余弦相似度
            dot_products = np.dot(candidate_vecs, query_vec)
            query_norm = np.linalg.norm(query_vec)
            candidate_norms = np.linalg.norm(candidate_vecs, axis=1)
            
            # 避免除零
            valid_indices = (candidate_norms != 0) & (query_norm != 0)
            similarities = np.zeros(len(candidate_embeddings))
            
            if query_norm != 0:
                similarities[valid_indices] = dot_products[valid_indices] / (
                    candidate_norms[valid_indices] * query_norm
                )
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"批量相似度计算失败: {str(e)}")
            return [0.0] * len(candidate_embeddings)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimensions": self.dimensions,
            "max_seq_length": getattr(self.embedding_model, 'max_seq_length', 512),
            "normalize_embeddings": True
        }
    
    def warm_up(self):
        """模型预热"""
        try:
            logger.info("开始嵌入模型预热...")
            
            # 使用一些示例文本进行预热
            warm_up_texts = [
                "这是一个测试文本",
                "CVE-2024-1234是一个高危漏洞",
                "主机192.168.1.100存在安全风险",
                "容器镜像nginx:latest需要更新"
            ]
            
            start_time = time.time()
            self.encode_texts(warm_up_texts)
            end_time = time.time()
            
            logger.info(f"嵌入模型预热完成，耗时: {end_time - start_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"嵌入模型预热失败: {str(e)}")
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 测试编码一个简单文本
            test_text = "健康检查测试"
            embedding = self.encode_text(test_text)
            
            # 检查输出维度
            if len(embedding) != self.dimensions:
                logger.error(f"嵌入维度不匹配: 期望{self.dimensions}, 实际{len(embedding)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"嵌入模型健康检查失败: {str(e)}")
            return False

class SemanticChunker:
    """基于语义的文本分块器"""
    
    def __init__(self, embedding_manager: EmbeddingManager, 
                 similarity_threshold: float = 0.8):
        self.embedding_manager = embedding_manager
        self.similarity_threshold = similarity_threshold
    
    def chunk_text(self, text: str, max_chunk_size: int = 512, 
                  overlap_size: int = 50) -> List[Dict[str, Any]]:
        """基于语义相似度的智能分块"""
        try:
            # 首先按句子分割
            sentences = self._split_sentences(text)
            
            if len(sentences) <= 1:
                return [{
                    "content": text,
                    "start_index": 0,
                    "end_index": len(text),
                    "sentence_count": len(sentences)
                }]
            
            # 计算句子嵌入
            sentence_embeddings = self.embedding_manager.encode_texts(sentences)
            
            # 基于语义相似度进行分块
            chunks = self._semantic_chunking(
                sentences, sentence_embeddings, max_chunk_size, overlap_size
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"语义分块失败: {str(e)}")
            # 回退到简单分块
            return self._simple_chunking(text, max_chunk_size, overlap_size)
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        import re
        
        # 中英文句子分割
        sentence_endings = r'[.!?。！？]+'
        sentences = re.split(sentence_endings, text)
        
        # 清理空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _semantic_chunking(self, sentences: List[str], 
                          embeddings: List[List[float]],
                          max_chunk_size: int, overlap_size: int) -> List[Dict[str, Any]]:
        """基于语义相似度的分块"""
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # 检查是否需要开始新块
            if (current_chunk_size + sentence_length > max_chunk_size and 
                len(current_chunk) > 0):
                
                # 保存当前块
                chunk_content = ' '.join(current_chunk)
                chunks.append({
                    "content": chunk_content,
                    "start_index": len(' '.join(chunks)) if chunks else 0,
                    "end_index": len(' '.join(chunks)) + len(chunk_content) if chunks else len(chunk_content),
                    "sentence_count": len(current_chunk)
                })
                
                # 开始新块，保留重叠
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, overlap_size
                )
                current_chunk = overlap_sentences + [sentence]
                current_chunk_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_chunk_size += sentence_length
        
        # 添加最后一个块
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append({
                "content": chunk_content,
                "start_index": len(' '.join([c["content"] for c in chunks])) if chunks else 0,
                "end_index": len(' '.join([c["content"] for c in chunks])) + len(chunk_content) if chunks else len(chunk_content),
                "sentence_count": len(current_chunk)
            })
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str], 
                             overlap_size: int) -> List[str]:
        """获取重叠句子"""
        if overlap_size <= 0:
            return []
        
        total_length = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            if total_length + len(sentence) <= overlap_size:
                overlap_sentences.insert(0, sentence)
                total_length += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def _simple_chunking(self, text: str, max_chunk_size: int, 
                        overlap_size: int) -> List[Dict[str, Any]]:
        """简单的字符级分块（回退方案）"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_chunk_size, len(text))
            
            # 尝试在单词边界分割
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({
                    "content": chunk_content,
                    "start_index": start,
                    "end_index": end,
                    "sentence_count": chunk_content.count('.') + chunk_content.count('。') + 1
                })
            
            start = max(start + 1, end - overlap_size)
        
        return chunks

# 全局嵌入管理器实例
_embedding_manager = None

def get_embedding_manager() -> EmbeddingManager:
    """获取嵌入管理器实例"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager
