"""
推理模块 - BERT模型推理和测试
支持掩码语言模型预测、文本相似度计算、嵌入提取等功能
重点关注推理过程的数据流转，包含详细的shape注释
"""

import torch
import torch.nn.functional as F
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

from config import BERT_CONFIG, DATA_CONFIG, get_device, setup_logging
from model import BertForPreTraining, BertForSequenceClassification

logger = logging.getLogger("BERT")


class BertInference:
    """
    BERT推理器

    支持多种推理任务：
    1. 掩码语言模型预测
    2. 文本相似度计算
    3. 文本嵌入提取
    4. 分类预测
    """

    def __init__(self, model_path: str, model_type: str = "pretraining"):
        """
        初始化推理器

        Args:
            model_path: 模型路径
            model_type: 模型类型 ("pretraining" 或 "classification")
        """
        # 设置日志
        setup_logging()

        # 设备配置
        self.device = get_device()
        logger.info(f"使用设备: {self.device}")

        # 模型配置
        self.model_path = Path(model_path)
        self.model_type = model_type

        # 加载配置
        self.config = self._load_config()

        # 加载tokenizer
        self.tokenizer = self._load_tokenizer()

        # 加载模型
        self.model = self._load_model()

        logger.info("推理器初始化完成")

    def _load_config(self) -> Dict[str, Any]:
        """加载模型配置"""
        config_file = self.model_path / "config.json"

        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
            logger.info(f"加载模型配置: {config_file}")
        else:
            logger.warning(f"配置文件不存在: {config_file}，使用默认配置")
            config = BERT_CONFIG.model_dump()

        return config

    def _load_tokenizer(self) -> AutoTokenizer:
        """加载tokenizer"""
        logger.info(f"加载tokenizer: {DATA_CONFIG.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(DATA_CONFIG.tokenizer_name)

        # 确保有必要的特殊token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "[MASK]"})

        return tokenizer

    def _load_model(self):
        """加载模型"""
        logger.info(f"加载{self.model_type}模型...")

        # 根据模型类型创建模型
        if self.model_type == "pretraining":
            model = BertForPreTraining()
        elif self.model_type == "classification":
            num_labels = self.config.get("num_labels", 2)
            model = BertForSequenceClassification(num_labels=num_labels)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        # 加载权重
        model_file = self.model_path / "pytorch_model.bin"
        if model_file.exists():
            state_dict = torch.load(model_file, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info(f"成功加载模型权重: {model_file}")
        else:
            logger.warning(f"模型权重文件不存在: {model_file}，使用随机初始化权重")

        model.to(self.device)
        model.eval()

        return model

    def predict_masked_tokens(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        预测掩码位置的词

        Args:
            text: 包含[MASK]的文本
            top_k: 返回前k个预测结果

        Returns:
            预测结果列表

        数据流转：
        text -> tokenization -> (batch_size=1, seq_len) -> model -> prediction_logits: (1, seq_len, vocab_size)
        -> 找到[MASK]位置 -> softmax -> top_k预测
        """
        if self.model_type != "pretraining":
            raise ValueError("掩码预测需要预训练模型")

        logger.info(f"预测掩码文本: {text}")

        # 检查是否包含[MASK]
        if "[MASK]" not in text:
            logger.warning("文本中没有[MASK] token")
            return []

        # tokenize文本
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=DATA_CONFIG.max_length,
            padding=True,
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction_logits = outputs["prediction_logits"]  # (1, seq_len, vocab_size)

        # 找到[MASK]位置
        input_ids = inputs["input_ids"].squeeze(
            0
        )  # (seq_len,) - 移除batch维度，得到序列长度维的张量

        # 计算[MASK]标记的位置：
        # 1. (input_ids == self.tokenizer.mask_token_id)
        #    - 比较input_ids中的每个token_id是否等于mask_token_id
        #    - 返回一个布尔张量，大小为(seq_len,)，值为True的位置即为[MASK]标记位置
        #
        # 2. .nonzero(as_tuple=True)
        #    - nonzero()找出张量中非零(True)元素的索引
        #    - as_tuple=True让返回值是一个元组，每个维度的索引分别存储
        #    - 对于1维张量，返回一个只包含一个张量的元组
        #
        # 3. [0]
        #    - 取元组中的第一个元素，即得到包含所有[MASK]位置索引的张量
        #
        # 最终mask_positions的形状为(num_masks,)，其中num_masks是文本中[MASK]标记的数量
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(
            as_tuple=True
        )[0]

        results = []
        for mask_pos in mask_positions:
            # 获取该位置的预测logits
            mask_logits = prediction_logits[0, mask_pos, :]  # (vocab_size,)

            # 计算概率
            probs = F.softmax(mask_logits, dim=-1)  # (vocab_size,)

            # 获取top_k预测
            # torch.topk返回两个张量：
            # 1. top_probs: 形状为(top_k,)
            #    - 包含概率分布中最大的k个概率值
            #    - 这些值是经过softmax后的概率，范围在[0,1]之间
            #    - 按降序排列，即第一个值是最大概率
            # 2. top_indices: 形状为(top_k,)
            #    - 包含这k个最大概率值对应的词表索引
            #    - 这些索引可以用来从词表中查找对应的实际单词
            #    - 与top_probs一一对应，例如top_indices[0]是概率为top_probs[0]的词的索引
            #
            # 例如，如果top_k=3，可能的结果：
            # top_probs:   [0.82, 0.12, 0.06] - 三个最高概率
            # top_indices: [2045,  182,  359] - 对应的词表索引
            top_probs, top_indices = torch.topk(probs, top_k)

            # 转换为词汇
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                token = self.tokenizer.decode([idx.item()])
                predictions.append(
                    {"token": token, "probability": prob.item(), "token_id": idx.item()}
                )

            results.append({"position": mask_pos.item(), "predictions": predictions})

        logger.info(f"预测完成，找到 {len(results)} 个[MASK]位置")

        return results

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            相似度分数 (0-1)

        数据流转：
        text1, text2 -> tokenization -> (batch_size=2, seq_len)
        -> model -> last_hidden_state: (2, seq_len, hidden_size)
        -> 池化 -> embeddings: (2, hidden_size)
        -> 余弦相似度 -> similarity_score
        """
        logger.info(f"计算文本相似度")
        logger.info(f"文本1: {text1}")
        logger.info(f"文本2: {text2}")

        # tokenize两个文本
        inputs1 = self.tokenizer(
            text1,
            return_tensors="pt",
            truncation=True,
            max_length=DATA_CONFIG.max_length,
            padding=True,
        )

        inputs2 = self.tokenizer(
            text2,
            return_tensors="pt",
            truncation=True,
            max_length=DATA_CONFIG.max_length,
            padding=True,
        )

        # 移动到设备
        inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}

        # 获取文本嵌入
        with torch.no_grad():
            # 获取第一个文本的嵌入
            outputs1 = self.model.bert(**inputs1)  # 使用基础BERT模型
            embedding1 = outputs1["pooler_output"]  # (1, hidden_size)

            # 获取第二个文本的嵌入
            outputs2 = self.model.bert(**inputs2)
            embedding2 = outputs2["pooler_output"]  # (1, hidden_size)

        # 计算余弦相似度
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1)  # (1,)
        similarity_score = similarity.item()

        logger.info(f"相似度分数: {similarity_score:.4f}")

        return similarity_score

    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        提取文本嵌入

        Args:
            texts: 文本列表

        Returns:
            嵌入矩阵 (num_texts, hidden_size)

        数据流转：
        texts -> tokenization -> (batch_size, seq_len)
        -> model -> pooler_output: (batch_size, hidden_size)
        -> numpy array
        """
        logger.info(f"提取 {len(texts)} 个文本的嵌入")

        embeddings = []

        for text in texts:
            # tokenize文本
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=DATA_CONFIG.max_length,
                padding=True,
            )

            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 获取嵌入
            with torch.no_grad():
                outputs = self.model.bert(**inputs)
                embedding = outputs["pooler_output"]  # (1, hidden_size)
                embeddings.append(embedding.cpu().numpy())

        # 合并所有嵌入
        embeddings = np.vstack(embeddings)  # (num_texts, hidden_size)

        logger.info(f"嵌入提取完成，形状: {embeddings.shape}")

        return embeddings

    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        文本分类预测

        Args:
            text: 要分类的文本

        Returns:
            分类结果

        数据流转：
        text -> tokenization -> (batch_size=1, seq_len)
        -> model -> logits: (1, num_labels)
        -> softmax -> probabilities
        """
        if self.model_type != "classification":
            raise ValueError("文本分类需要分类模型")

        logger.info(f"分类文本: {text}")

        # tokenize文本
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=DATA_CONFIG.max_length,
            padding=True,
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]  # (1, num_labels)

        # 计算概率
        probs = F.softmax(logits, dim=-1)  # (1, num_labels)

        # 获取预测结果
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probs[0, predicted_class].item()

        # 获取所有类别的概率
        all_probs = probs.squeeze(0).cpu().numpy()

        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probs.tolist(),
        }

        logger.info(f"预测类别: {predicted_class}, 置信度: {confidence:.4f}")

        return result


def demo_inference():
    """推理演示"""
    print("=== BERT推理演示 ===")

    # 这里需要实际的模型路径
    model_path = "./bert2_output/best_model"

    if not Path(model_path).exists():
        print(f"模型路径不存在: {model_path}")
        print("请先运行预训练或提供正确的模型路径")
        return

    # 创建推理器
    inference = BertInference(model_path, model_type="pretraining")

    # 1. 掩码预测演示
    print("\n1. 掩码语言模型预测:")
    mask_text = "The capital of France is [MASK]."
    results = inference.predict_masked_tokens(mask_text, top_k=3)

    for result in results:
        print(f"位置 {result['position']}:")
        for pred in result["predictions"]:
            print(f"  {pred['token']}: {pred['probability']:.4f}")

    # 2. 文本相似度演示
    print("\n2. 文本相似度计算:")
    text1 = "I love cats."
    text2 = "I adore felines."
    similarity = inference.compute_text_similarity(text1, text2)
    print(f"'{text1}' 和 '{text2}' 的相似度: {similarity:.4f}")

    # 3. 嵌入提取演示
    print("\n3. 文本嵌入提取:")
    texts = ["Hello world", "Machine learning", "Natural language processing"]
    embeddings = inference.extract_embeddings(texts)
    print(f"提取了 {len(texts)} 个文本的嵌入，形状: {embeddings.shape}")


def main():
    """主函数"""
    import sys

    if len(sys.argv) < 2:
        print("用法: python inference.py <模型路径> [模型类型]")
        print("模型类型: pretraining (默认) 或 classification")
        demo_inference()
        return

    model_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "pretraining"

    # 创建推理器
    inference = BertInference(model_path, model_type)

    # 交互式推理
    print(f"BERT推理器已启动 (模型类型: {model_type})")
    print("输入 'quit' 退出")

    while True:
        try:
            if model_type == "pretraining":
                text = input("\n请输入包含[MASK]的文本: ")
                if text.lower() == "quit":
                    break

                results = inference.predict_masked_tokens(text, top_k=3)
                for result in results:
                    print(f"位置 {result['position']}:")
                    for pred in result["predictions"]:
                        print(f"  {pred['token']}: {pred['probability']:.4f}")

            elif model_type == "classification":
                text = input("\n请输入要分类的文本: ")
                if text.lower() == "quit":
                    break

                result = inference.classify_text(text)
                print(f"预测类别: {result['predicted_class']}")
                print(f"置信度: {result['confidence']:.4f}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")

    print("推理器已退出")


if __name__ == "__main__":
    main()
