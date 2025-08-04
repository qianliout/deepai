#!/usr/bin/env python3
"""
RAG2项目完整系统测试
测试从文档处理到RAG查询的完整流程
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import get_config
from utils.logger import get_logger, LogContext
from core.document_processor import DocumentProcessor
from core.rag_pipeline import get_rag_pipeline
from retrieval.semantic_retriever import SemanticRetriever
from retrieval.base_retriever import get_retriever_manager
from storage.postgresql_manager import PostgreSQLManager
from storage.redis_manager import RedisManager
from storage.mysql_manager import MySQLManager

logger = get_logger("test_complete_system")

class SystemTester:
    """系统测试器"""
    
    def __init__(self):
        self.config = get_config()
        self.test_results = {}
    
    async def test_database_connections(self):
        """测试数据库连接"""
        logger.info("=== 测试数据库连接 ===")
        
        results = {}
        
        # PostgreSQL
        try:
            pg_manager = PostgreSQLManager()
            await pg_manager.initialize()
            results["PostgreSQL"] = await pg_manager.health_check()
            await pg_manager.close()
        except Exception as e:
            logger.error(f"PostgreSQL测试失败: {str(e)}")
            results["PostgreSQL"] = False
        
        # Redis
        try:
            redis_manager = RedisManager()
            await redis_manager.initialize()
            results["Redis"] = await redis_manager.health_check()
            await redis_manager.close()
        except Exception as e:
            logger.error(f"Redis测试失败: {str(e)}")
            results["Redis"] = False
        
        # MySQL
        try:
            mysql_manager = MySQLManager()
            await mysql_manager.initialize()
            results["MySQL"] = await mysql_manager.health_check()
            await mysql_manager.close()
        except Exception as e:
            logger.error(f"MySQL测试失败: {str(e)}")
            results["MySQL"] = False
        
        self.test_results["databases"] = results
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"数据库连接测试: {success_count}/{len(results)} 成功")
        
        return success_count > 0
    
    async def test_model_loading(self):
        """测试模型加载"""
        logger.info("=== 测试模型加载 ===")
        
        results = {}
        
        # 嵌入模型
        try:
            from models.embeddings import get_embedding_manager
            embedding_manager = get_embedding_manager()
            
            # 测试编码
            test_text = "这是一个测试文本"
            embedding = embedding_manager.encode_text(test_text)
            
            results["embedding"] = len(embedding) > 0
            logger.info(f"嵌入模型测试: {'✅ 成功' if results['embedding'] else '❌ 失败'}")
            
        except Exception as e:
            logger.error(f"嵌入模型测试失败: {str(e)}")
            results["embedding"] = False
        
        # 重排序模型
        try:
            from models.rerank_models import get_rerank_manager
            rerank_manager = get_rerank_manager()
            
            # 测试重排序
            test_query = "测试查询"
            test_docs = ["相关文档", "不相关文档"]
            rerank_results = rerank_manager.rerank_documents(test_query, test_docs)
            
            results["reranker"] = len(rerank_results) > 0
            logger.info(f"重排序模型测试: {'✅ 成功' if results['reranker'] else '❌ 失败'}")
            
        except Exception as e:
            logger.error(f"重排序模型测试失败: {str(e)}")
            results["reranker"] = False
        
        # LLM客户端
        try:
            from models.llm_client import get_llm_client
            llm_client = await get_llm_client()
            
            results["llm"] = await llm_client.health_check()
            logger.info(f"LLM客户端测试: {'✅ 成功' if results['llm'] else '❌ 失败'}")
            
        except Exception as e:
            logger.error(f"LLM客户端测试失败: {str(e)}")
            results["llm"] = False
        
        self.test_results["models"] = results
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"模型加载测试: {success_count}/{len(results)} 成功")
        
        return success_count > 0
    
    async def test_document_processing(self):
        """测试文档处理"""
        logger.info("=== 测试文档处理 ===")
        
        try:
            # 创建测试文档
            test_doc_path = Path("data/test_document.txt")
            test_doc_path.parent.mkdir(parents=True, exist_ok=True)
            
            test_content = """
AIOps安全最佳实践

1. 容器安全
容器安全是现代AIOps环境中的关键组成部分。以下是确保容器安全的基本原则：

1.1 镜像安全
- 使用官方或可信的基础镜像
- 定期扫描镜像漏洞，建议使用Trivy、Clair或Snyk等工具
- 实施镜像签名验证机制

1.2 运行时安全
- 以非root用户运行容器
- 使用只读文件系统
- 限制容器的系统调用权限

2. 漏洞管理
建立自动化漏洞扫描流程，集成CI/CD管道中的安全检查。

CVE-2024-1234是一个高危漏洞，影响nginx:latest镜像。
建议立即更新到最新版本以修复此漏洞。
"""
            
            with open(test_doc_path, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            # 测试文档处理
            processor = DocumentProcessor()
            result = await processor.process_document(str(test_doc_path))
            
            # 验证结果
            document = result["document"]
            chunks = result["chunks"]
            
            success = (
                len(document["content"]) > 0 and
                len(chunks) > 0 and
                all("embedding" in chunk for chunk in chunks)
            )
            
            logger.info(f"文档处理测试: {'✅ 成功' if success else '❌ 失败'}")
            logger.info(f"  - 文档长度: {len(document['content'])}")
            logger.info(f"  - 分块数量: {len(chunks)}")
            logger.info(f"  - 嵌入维度: {len(chunks[0]['embedding']) if chunks else 0}")
            
            # 清理测试文件
            test_doc_path.unlink()
            
            self.test_results["document_processing"] = success
            return success
            
        except Exception as e:
            logger.error(f"文档处理测试失败: {str(e)}")
            self.test_results["document_processing"] = False
            return False
    
    async def test_retrieval_system(self):
        """测试检索系统"""
        logger.info("=== 测试检索系统 ===")
        
        try:
            # 创建语义检索器
            semantic_retriever = SemanticRetriever()
            
            # 注册检索器
            retriever_manager = get_retriever_manager()
            retriever_manager.register_retriever(semantic_retriever, is_default=True)
            
            # 准备测试数据
            test_documents = [{
                "document": {
                    "id": "test-doc-1",
                    "title": "AIOps安全指南",
                    "content": "这是一个关于AIOps安全的测试文档。包含容器安全、漏洞管理等内容。",
                    "source": "test",
                    "document_type": "txt",
                    "metadata": {"test": True}
                },
                "chunks": [{
                    "id": "test-chunk-1",
                    "document_id": "test-doc-1",
                    "chunk_index": 0,
                    "content": "AIOps安全是现代运维的重要组成部分，包括容器安全、漏洞管理、访问控制等方面。",
                    "token_count": 25,
                    "chunk_metadata": {"test": True}
                }]
            }]
            
            # 添加文档到检索器
            add_success = await semantic_retriever.add_documents(test_documents)
            
            if add_success:
                # 测试检索
                await asyncio.sleep(1)  # 等待索引更新
                
                results = await semantic_retriever.retrieve("AIOps安全", top_k=5)
                
                success = len(results) > 0
                logger.info(f"检索系统测试: {'✅ 成功' if success else '❌ 失败'}")
                logger.info(f"  - 检索结果数: {len(results)}")
                
                if results:
                    logger.info(f"  - 最高相似度: {results[0].get('similarity', 0):.4f}")
            else:
                success = False
                logger.error("文档添加失败")
            
            self.test_results["retrieval"] = success
            return success
            
        except Exception as e:
            logger.error(f"检索系统测试失败: {str(e)}")
            self.test_results["retrieval"] = False
            return False
    
    async def test_rag_pipeline(self):
        """测试RAG管道"""
        logger.info("=== 测试RAG管道 ===")
        
        try:
            rag_pipeline = await get_rag_pipeline()
            
            # 测试查询
            test_queries = [
                "什么是AIOps安全？",
                "如何确保容器安全？",
                "CVE-2024-1234漏洞的影响是什么？"
            ]
            
            success_count = 0
            
            for query in test_queries:
                try:
                    result = await rag_pipeline.query(query)
                    
                    if result and result.get("response"):
                        success_count += 1
                        logger.info(f"查询成功: {query[:20]}... -> {result['response'][:50]}...")
                    else:
                        logger.warning(f"查询无响应: {query}")
                        
                except Exception as e:
                    logger.error(f"查询失败: {query}, 错误: {str(e)}")
            
            success = success_count > 0
            logger.info(f"RAG管道测试: {'✅ 成功' if success else '❌ 失败'}")
            logger.info(f"  - 成功查询: {success_count}/{len(test_queries)}")
            
            self.test_results["rag_pipeline"] = success
            return success
            
        except Exception as e:
            logger.error(f"RAG管道测试失败: {str(e)}")
            self.test_results["rag_pipeline"] = False
            return False
    
    async def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始RAG2完整系统测试")
        
        start_time = time.time()
        
        # 运行测试
        tests = [
            ("数据库连接", self.test_database_connections()),
            ("模型加载", self.test_model_loading()),
            ("文档处理", self.test_document_processing()),
            ("检索系统", self.test_retrieval_system()),
            ("RAG管道", self.test_rag_pipeline())
        ]
        
        results = []
        for test_name, test_coro in tests:
            logger.info(f"\n--- 开始测试: {test_name} ---")
            try:
                result = await test_coro
                results.append((test_name, result))
            except Exception as e:
                logger.error(f"测试 {test_name} 异常: {str(e)}")
                results.append((test_name, False))
        
        # 汇总结果
        end_time = time.time()
        total_time = end_time - start_time
        
        success_count = sum(1 for _, success in results if success)
        total_count = len(results)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 测试结果汇总")
        logger.info("=" * 60)
        
        for test_name, success in results:
            status = "✅ 通过" if success else "❌ 失败"
            logger.info(f"{test_name:20} : {status}")
        
        logger.info("-" * 60)
        logger.info(f"总体结果: {success_count}/{total_count} 测试通过")
        logger.info(f"测试耗时: {total_time:.2f} 秒")
        
        if success_count == total_count:
            logger.info("🎉 所有测试通过！系统运行正常")
            return 0
        else:
            logger.warning(f"⚠️  有 {total_count - success_count} 项测试失败")
            return 1

async def main():
    """主函数"""
    tester = SystemTester()
    return await tester.run_all_tests()

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试过程中发生未预期的错误: {str(e)}")
        sys.exit(1)
