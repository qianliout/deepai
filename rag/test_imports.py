#!/usr/bin/env python3
"""
测试导入脚本

验证所有模块是否可以正常导入
"""

def test_imports():
    """测试所有模块导入"""
    print("🧪 开始测试模块导入...")
    
    try:
        print("  ✓ 导入 config...")
        from config import config
        
        print("  ✓ 导入 logger...")
        from logger import get_logger, setup_logger
        
        print("  ✓ 导入 document_loader...")
        from document_loader import DocumentLoader
        
        print("  ✓ 导入 embeddings...")
        from embeddings import EmbeddingManager
        
        print("  ✓ 导入 vector_store...")
        from vector_store import VectorStoreManager
        
        print("  ✓ 导入 llm...")
        from llm import LLMManager
        
        print("  ✓ 导入 text_splitter...")
        from text_splitter import TextSplitterManager
        
        print("  ✓ 导入 retriever...")
        from retriever import RetrieverManager
        
        print("  ✓ 导入 rag_chain...")
        from rag_chain import RAGChain
        
        print("✅ 所有模块导入成功！")
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

if __name__ == "__main__":
    test_imports()
