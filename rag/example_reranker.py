"""
重排序器使用示例

该脚本展示如何在RAG系统中使用双编码器和交叉编码器进行重排序。
包含完整的使用流程和配置示例。

使用场景：
1. 提升检索结果的相关性
2. 在大规模候选集中精确排序
3. 结合多种排序信号
4. 优化用户查询体验
"""

import os
import sys
from typing import List
from langchain_core.documents import Document

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import defaultConfig, load_env_config, print_config
from logger import get_logger, setup_logging
from reranker import BiEncoderReranker, CrossEncoderReranker, HybridReranker, rerank_documents


def create_sample_documents() -> List[Document]:
    """创建示例文档集合"""
    documents = [
        Document(
            page_content="机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习和改进。机器学习算法通过分析大量数据来识别模式，并使用这些模式来做出预测或决策。",
            metadata={"title": "机器学习基础", "source": "AI教程", "doc_id": "ml_001"}
        ),
        Document(
            page_content="深度学习是机器学习的一个子集，它使用人工神经网络来模拟人脑的学习过程。深度学习在图像识别、自然语言处理和语音识别等领域取得了突破性进展。",
            metadata={"title": "深度学习概述", "source": "AI教程", "doc_id": "dl_001"}
        ),
        Document(
            page_content="自然语言处理（NLP）是人工智能的一个分支，专注于让计算机理解、解释和生成人类语言。NLP技术被广泛应用于搜索引擎、聊天机器人和机器翻译等领域。",
            metadata={"title": "自然语言处理", "source": "AI教程", "doc_id": "nlp_001"}
        ),
        Document(
            page_content="计算机视觉是人工智能的一个重要领域，致力于让计算机能够识别和理解图像和视频中的内容。它在自动驾驶、医疗诊断和安防监控等方面有广泛应用。",
            metadata={"title": "计算机视觉", "source": "AI教程", "doc_id": "cv_001"}
        ),
        Document(
            page_content="强化学习是机器学习的一个分支，通过与环境的交互来学习最优的行为策略。智能体通过试错和奖励机制来改进其决策能力，在游戏AI和机器人控制中表现出色。",
            metadata={"title": "强化学习", "source": "AI教程", "doc_id": "rl_001"}
        ),
        Document(
            page_content="数据挖掘是从大量数据中发现有用信息和知识的过程。它结合了统计学、机器学习和数据库技术，帮助企业从数据中获得商业洞察。",
            metadata={"title": "数据挖掘", "source": "数据科学", "doc_id": "dm_001"}
        ),
        Document(
            page_content="人工智能伦理是研究AI系统开发和应用中的道德问题的领域。随着AI技术的快速发展，确保AI系统的公平性、透明性和可解释性变得越来越重要。",
            metadata={"title": "AI伦理", "source": "AI哲学", "doc_id": "ethics_001"}
        ),
        Document(
            page_content="云计算是一种通过互联网提供计算资源和服务的模式。它为机器学习和人工智能应用提供了强大的计算能力和存储空间，降低了技术门槛。",
            metadata={"title": "云计算与AI", "source": "技术架构", "doc_id": "cloud_001"}
        )
    ]
    return documents


def demonstrate_bi_encoder_reranking():
    """演示双编码器重排序"""
    logger = get_logger("BiEncoderDemo")
    logger.info("=" * 60)
    logger.info("双编码器重排序演示")
    logger.info("=" * 60)
    
    try:
        # 创建双编码器重排序器
        bi_encoder = BiEncoderReranker()
        
        # 准备测试数据
        documents = create_sample_documents()
        query = "机器学习算法的原理是什么"
        
        logger.info(f"查询: {query}")
        logger.info(f"候选文档数量: {len(documents)}")
        
        # 执行重排序
        results = bi_encoder.rerank(query, documents, top_k=5)
        
        logger.info(f"\n双编码器重排序结果 (Top {len(results)}):")
        logger.info("-" * 80)
        
        for i, result in enumerate(results):
            logger.info(
                f"排名 {i+1}: {result.document.metadata['title']}\n"
                f"  分数: {result.bi_encoder_score:.4f}\n"
                f"  内容: {result.document.page_content[:100]}...\n"
            )
        
    except Exception as e:
        logger.error(f"双编码器重排序演示失败: {e}")


def demonstrate_cross_encoder_reranking():
    """演示交叉编码器重排序"""
    logger = get_logger("CrossEncoderDemo")
    logger.info("=" * 60)
    logger.info("交叉编码器重排序演示")
    logger.info("=" * 60)
    
    try:
        # 创建交叉编码器重排序器
        cross_encoder = CrossEncoderReranker()
        
        # 准备测试数据
        documents = create_sample_documents()
        query = "深度学习在哪些领域有应用"
        
        logger.info(f"查询: {query}")
        logger.info(f"候选文档数量: {len(documents)}")
        
        # 执行重排序
        results = cross_encoder.rerank(query, documents, top_k=5)
        
        logger.info(f"\n交叉编码器重排序结果 (Top {len(results)}):")
        logger.info("-" * 80)
        
        for i, result in enumerate(results):
            logger.info(
                f"排名 {i+1}: {result.document.metadata['title']}\n"
                f"  分数: {result.cross_encoder_score:.4f}\n"
                f"  内容: {result.document.page_content[:100]}...\n"
            )
        
    except Exception as e:
        logger.error(f"交叉编码器重排序演示失败: {e}")


def demonstrate_hybrid_reranking():
    """演示混合重排序"""
    logger = get_logger("HybridDemo")
    logger.info("=" * 60)
    logger.info("混合重排序演示")
    logger.info("=" * 60)
    
    try:
        # 创建混合重排序器
        hybrid_reranker = HybridReranker()
        
        # 准备测试数据
        documents = create_sample_documents()
        query = "人工智能技术的发展趋势"
        
        logger.info(f"查询: {query}")
        logger.info(f"候选文档数量: {len(documents)}")
        
        # 执行重排序
        results = hybrid_reranker.rerank(query, documents, top_k=5)
        
        logger.info(f"\n混合重排序结果 (Top {len(results)}):")
        logger.info("-" * 80)
        
        for i, result in enumerate(results):
            logger.info(
                f"排名 {i+1}: {result.document.metadata['title']}\n"
                f"  最终分数: {result.final_score:.4f}\n"
                f"  双编码器分数: {result.bi_encoder_score:.4f if result.bi_encoder_score else 'N/A'}\n"
                f"  交叉编码器分数: {result.cross_encoder_score:.4f if result.cross_encoder_score else 'N/A'}\n"
                f"  重排序方法: {result.rerank_method}\n"
                f"  内容: {result.document.page_content[:100]}...\n"
            )
        
        # 显示重排序器信息
        reranker_info = hybrid_reranker.get_reranker_info()
        logger.info("\n重排序器配置信息:")
        logger.info("-" * 40)
        for key, value in reranker_info.items():
            logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"混合重排序演示失败: {e}")


def demonstrate_convenience_function():
    """演示便捷函数使用"""
    logger = get_logger("ConvenienceDemo")
    logger.info("=" * 60)
    logger.info("便捷函数使用演示")
    logger.info("=" * 60)
    
    try:
        # 准备测试数据
        documents = create_sample_documents()
        query = "什么是强化学习"
        
        logger.info(f"查询: {query}")
        logger.info(f"候选文档数量: {len(documents)}")
        
        # 使用便捷函数进行重排序
        results = rerank_documents(
            query=query,
            documents=documents,
            reranker_type="hybrid",
            top_k=3
        )
        
        logger.info(f"\n便捷函数重排序结果 (Top {len(results)}):")
        logger.info("-" * 80)
        
        for i, result in enumerate(results):
            logger.info(
                f"排名 {i+1}: {result.document.metadata['title']}\n"
                f"  分数: {result.final_score:.4f}\n"
                f"  方法: {result.rerank_method}\n"
            )
        
    except Exception as e:
        logger.error(f"便捷函数演示失败: {e}")


def compare_reranking_methods():
    """比较不同重排序方法的效果"""
    logger = get_logger("ComparisonDemo")
    logger.info("=" * 60)
    logger.info("重排序方法效果比较")
    logger.info("=" * 60)
    
    try:
        # 准备测试数据
        documents = create_sample_documents()
        query = "机器学习和深度学习的区别"
        
        logger.info(f"查询: {query}")
        logger.info(f"候选文档数量: {len(documents)}")
        
        # 测试不同的重排序方法
        methods = ["bi_encoder", "cross_encoder", "hybrid"]
        
        for method in methods:
            try:
                logger.info(f"\n{method.upper()}重排序结果:")
                logger.info("-" * 40)
                
                results = rerank_documents(
                    query=query,
                    documents=documents,
                    reranker_type=method,
                    top_k=3
                )
                
                for i, result in enumerate(results):
                    score_info = ""
                    if result.bi_encoder_score:
                        score_info += f"双编码器: {result.bi_encoder_score:.4f} "
                    if result.cross_encoder_score:
                        score_info += f"交叉编码器: {result.cross_encoder_score:.4f} "
                    
                    logger.info(
                        f"  {i+1}. {result.document.metadata['title']} "
                        f"(最终分数: {result.final_score:.4f}, {score_info})"
                    )
                
            except Exception as e:
                logger.warning(f"{method}重排序失败: {e}")
        
    except Exception as e:
        logger.error(f"重排序方法比较失败: {e}")


def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logger = get_logger("RerankerExample")
    
    # 加载配置
    load_env_config()
    
    logger.info("开始重排序器使用示例")
    
    # 打印配置信息
    logger.info("\n当前重排序器配置:")
    logger.info(f"  双编码器模型: {defaultConfig.reranker.bi_encoder_model}")
    logger.info(f"  交叉编码器模型: {defaultConfig.reranker.cross_encoder_model}")
    logger.info(f"  融合方法: {defaultConfig.reranker.fusion_method}")
    logger.info(f"  设备: {defaultConfig.reranker.device}")
    
    # 运行各种演示
    try:
        demonstrate_bi_encoder_reranking()
    except Exception as e:
        logger.error(f"双编码器演示失败: {e}")
    
    try:
        demonstrate_cross_encoder_reranking()
    except Exception as e:
        logger.error(f"交叉编码器演示失败: {e}")
    
    try:
        demonstrate_hybrid_reranking()
    except Exception as e:
        logger.error(f"混合重排序演示失败: {e}")
    
    try:
        demonstrate_convenience_function()
    except Exception as e:
        logger.error(f"便捷函数演示失败: {e}")
    
    try:
        compare_reranking_methods()
    except Exception as e:
        logger.error(f"方法比较演示失败: {e}")
    
    logger.info("\n重排序器使用示例完成")


if __name__ == "__main__":
    main()
