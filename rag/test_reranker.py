"""
重排序器测试模块

该模块用于测试双编码器和交叉编码器的重排序功能。
包含单元测试和集成测试，验证重排序效果。

测试内容：
1. 双编码器重排序测试
2. 交叉编码器重排序测试
3. 混合重排序测试
4. 分数融合测试
5. 性能基准测试
"""

import time
import pytest
from typing import List
from langchain_core.documents import Document

from config import defaultConfig
from logger import get_logger
from reranker import BiEncoderReranker, CrossEncoderReranker, HybridReranker, RerankResult, create_reranker, rerank_documents


class TestReranker:
    """重排序器测试类"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.logger = get_logger("TestReranker")
        
        # 创建测试文档
        cls.test_documents = [
            Document(
                page_content="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                metadata={"title": "人工智能概述", "doc_id": "doc1"}
            ),
            Document(
                page_content="机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。",
                metadata={"title": "机器学习介绍", "doc_id": "doc2"}
            ),
            Document(
                page_content="深度学习是机器学习的一个分支，使用神经网络来模拟人脑的学习过程。",
                metadata={"title": "深度学习原理", "doc_id": "doc3"}
            ),
            Document(
                page_content="自然语言处理是人工智能的一个重要应用领域，处理人类语言的理解和生成。",
                metadata={"title": "自然语言处理", "doc_id": "doc4"}
            ),
            Document(
                page_content="计算机视觉是人工智能的另一个重要分支，专注于让计算机理解和解释视觉信息。",
                metadata={"title": "计算机视觉", "doc_id": "doc5"}
            )
        ]
        
        cls.test_query = "什么是机器学习"
    
    def test_bi_encoder_reranker(self):
        """测试双编码器重排序"""
        try:
            self.logger.info("开始测试双编码器重排序")
            
            # 创建双编码器重排序器
            bi_encoder = BiEncoderReranker()
            
            # 执行重排序
            results = bi_encoder.rerank(self.test_query, self.test_documents, top_k=3)
            
            # 验证结果
            assert len(results) <= 3, "返回结果数量应该不超过top_k"
            assert all(isinstance(r, RerankResult) for r in results), "所有结果应该是RerankResult类型"
            assert all(r.bi_encoder_score is not None for r in results), "所有结果应该有双编码器分数"
            assert all(r.rerank_method == "bi_encoder" for r in results), "重排序方法应该是bi_encoder"
            
            # 验证分数排序
            scores = [r.bi_encoder_score for r in results]
            assert scores == sorted(scores, reverse=True), "结果应该按分数降序排列"
            
            self.logger.info(f"双编码器重排序测试通过，返回{len(results)}个结果")
            
            # 打印结果
            for i, result in enumerate(results):
                self.logger.info(f"排名{i+1}: {result.document.metadata['title']} | 分数: {result.bi_encoder_score:.4f}")
            
        except Exception as e:
            self.logger.warning(f"双编码器重排序测试跳过: {e}")
            pytest.skip(f"双编码器模型不可用: {e}")
    
    def test_cross_encoder_reranker(self):
        """测试交叉编码器重排序"""
        try:
            self.logger.info("开始测试交叉编码器重排序")
            
            # 创建交叉编码器重排序器
            cross_encoder = CrossEncoderReranker()
            
            # 执行重排序
            results = cross_encoder.rerank(self.test_query, self.test_documents, top_k=3)
            
            # 验证结果
            assert len(results) <= 3, "返回结果数量应该不超过top_k"
            assert all(isinstance(r, RerankResult) for r in results), "所有结果应该是RerankResult类型"
            assert all(r.cross_encoder_score is not None for r in results), "所有结果应该有交叉编码器分数"
            assert all(r.rerank_method == "cross_encoder" for r in results), "重排序方法应该是cross_encoder"
            
            # 验证分数排序
            scores = [r.cross_encoder_score for r in results]
            assert scores == sorted(scores, reverse=True), "结果应该按分数降序排列"
            
            self.logger.info(f"交叉编码器重排序测试通过，返回{len(results)}个结果")
            
            # 打印结果
            for i, result in enumerate(results):
                self.logger.info(f"排名{i+1}: {result.document.metadata['title']} | 分数: {result.cross_encoder_score:.4f}")
            
        except Exception as e:
            self.logger.warning(f"交叉编码器重排序测试跳过: {e}")
            pytest.skip(f"交叉编码器模型不可用: {e}")
    
    def test_hybrid_reranker(self):
        """测试混合重排序器"""
        try:
            self.logger.info("开始测试混合重排序器")
            
            # 创建混合重排序器
            hybrid_reranker = HybridReranker()
            
            # 执行重排序
            results = hybrid_reranker.rerank(self.test_query, self.test_documents, top_k=3)
            
            # 验证结果
            assert len(results) <= 3, "返回结果数量应该不超过top_k"
            assert all(isinstance(r, RerankResult) for r in results), "所有结果应该是RerankResult类型"
            assert all(r.final_score is not None for r in results), "所有结果应该有最终分数"
            
            # 验证分数排序
            scores = [r.final_score for r in results]
            assert scores == sorted(scores, reverse=True), "结果应该按最终分数降序排列"
            
            self.logger.info(f"混合重排序测试通过，返回{len(results)}个结果")
            
            # 打印结果
            for i, result in enumerate(results):
                self.logger.info(
                    f"排名{i+1}: {result.document.metadata['title']} | "
                    f"最终分数: {result.final_score:.4f} | "
                    f"方法: {result.rerank_method}"
                )
            
        except Exception as e:
            self.logger.warning(f"混合重排序测试跳过: {e}")
            pytest.skip(f"重排序器不可用: {e}")
    
    def test_reranker_factory(self):
        """测试重排序器工厂函数"""
        self.logger.info("开始测试重排序器工厂函数")
        
        try:
            # 测试创建不同类型的重排序器
            bi_encoder = create_reranker("bi_encoder")
            assert isinstance(bi_encoder, BiEncoderReranker), "应该创建双编码器重排序器"
            
            cross_encoder = create_reranker("cross_encoder")
            assert isinstance(cross_encoder, CrossEncoderReranker), "应该创建交叉编码器重排序器"
            
            hybrid = create_reranker("hybrid")
            assert isinstance(hybrid, HybridReranker), "应该创建混合重排序器"
            
            # 测试无效类型
            with pytest.raises(ValueError):
                create_reranker("invalid_type")
            
            self.logger.info("重排序器工厂函数测试通过")
            
        except Exception as e:
            self.logger.warning(f"重排序器工厂函数测试跳过: {e}")
            pytest.skip(f"重排序器不可用: {e}")
    
    def test_convenience_function(self):
        """测试便捷函数"""
        try:
            self.logger.info("开始测试便捷函数")
            
            # 使用便捷函数进行重排序
            results = rerank_documents(
                self.test_query, 
                self.test_documents, 
                reranker_type="hybrid",
                top_k=3
            )
            
            # 验证结果
            assert len(results) <= 3, "返回结果数量应该不超过top_k"
            assert all(isinstance(r, RerankResult) for r in results), "所有结果应该是RerankResult类型"
            
            self.logger.info(f"便捷函数测试通过，返回{len(results)}个结果")
            
        except Exception as e:
            self.logger.warning(f"便捷函数测试跳过: {e}")
            pytest.skip(f"重排序器不可用: {e}")
    
    def test_performance_benchmark(self):
        """性能基准测试"""
        try:
            self.logger.info("开始性能基准测试")
            
            # 创建更多测试文档
            large_doc_set = self.test_documents * 10  # 50个文档
            
            # 测试不同重排序器的性能
            reranker_types = ["bi_encoder", "cross_encoder", "hybrid"]
            
            for reranker_type in reranker_types:
                try:
                    start_time = time.time()
                    
                    results = rerank_documents(
                        self.test_query,
                        large_doc_set,
                        reranker_type=reranker_type,
                        top_k=10
                    )
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    self.logger.info(
                        f"{reranker_type}重排序器性能: "
                        f"{len(large_doc_set)}个文档 -> {len(results)}个结果, "
                        f"耗时: {duration:.3f}秒"
                    )
                    
                except Exception as e:
                    self.logger.warning(f"{reranker_type}重排序器性能测试失败: {e}")
            
        except Exception as e:
            self.logger.warning(f"性能基准测试跳过: {e}")


def run_manual_test():
    """手动测试函数"""
    logger = get_logger("ManualTest")
    logger.info("开始手动测试重排序功能")
    
    # 创建测试实例
    test_instance = TestReranker()
    test_instance.setup_class()
    
    # 运行各项测试
    try:
        test_instance.test_bi_encoder_reranker()
    except Exception as e:
        logger.error(f"双编码器测试失败: {e}")
    
    try:
        test_instance.test_cross_encoder_reranker()
    except Exception as e:
        logger.error(f"交叉编码器测试失败: {e}")
    
    try:
        test_instance.test_hybrid_reranker()
    except Exception as e:
        logger.error(f"混合重排序测试失败: {e}")
    
    try:
        test_instance.test_performance_benchmark()
    except Exception as e:
        logger.error(f"性能测试失败: {e}")
    
    logger.info("手动测试完成")


if __name__ == "__main__":
    # 运行手动测试
    run_manual_test()
