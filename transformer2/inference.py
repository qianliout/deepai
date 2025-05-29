"""
推理模块 - 重构版本
负责模型推理和文本翻译
详细的数据流转注释和shape说明
"""

import torch
import torch.nn.functional as F
import logging
import os
from typing import List, Optional

from config import TRANSFORMER_CONFIG, TRAINING_CONFIG
from model import Transformer
from data_loader import SimpleTokenizer
from transformer import create_padding_mask, create_look_ahead_mask

logger = logging.getLogger("Transformer2")


class TransformerInference:
    """Transformer推理器

    负责加载训练好的模型并进行文本翻译

    数据流转：
    输入文本 -> 分词编码 -> 编码器 -> 解码器(逐步生成) -> 解码输出文本
    """

    def __init__(self, model_path: str, vocab_dir: Optional[str] = None):
        """
        初始化推理器

        Args:
            model_path: 模型文件路径
            vocab_dir: 词汇表目录路径
        """
        self.device = TRAINING_CONFIG.device
        self.model_path = model_path

        # 加载模型
        self.model = self._load_model()

        # 加载分词器
        self.tokenizer = self._load_tokenizer(vocab_dir)

        logger.info("推理器初始化完成")

    def _load_model(self) -> Transformer:
        """
        加载训练好的模型

        Returns:
            加载的Transformer模型
        """
        logger.info(f"正在加载模型: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 更新配置(如果检查点中包含配置)
        if "config" in checkpoint:
            config_dict = checkpoint["config"]["transformer"]
            for key, value in config_dict.items():
                if hasattr(TRANSFORMER_CONFIG, key):
                    setattr(TRANSFORMER_CONFIG, key, value)

        # 创建模型
        model = Transformer()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        logger.info("模型加载完成")
        return model

    def _load_tokenizer(self, vocab_dir: Optional[str] = None) -> SimpleTokenizer:
        """
        加载分词器

        Args:
            vocab_dir: 词汇表目录路径

        Returns:
            加载的分词器
        """
        if vocab_dir is None:
            vocab_dir = TRAINING_CONFIG.vocab_save_dir

        logger.info(f"正在加载分词器: {vocab_dir}")

        tokenizer = SimpleTokenizer()

        if os.path.exists(vocab_dir):
            tokenizer.load_vocab(vocab_dir)
        else:
            logger.warning(f"词汇表目录不存在: {vocab_dir}")
            raise FileNotFoundError(f"词汇表目录不存在: {vocab_dir}")

        logger.info("分词器加载完成")
        return tokenizer

    def translate(
        self,
        src_text: str,
        max_length: int = 100,
        beam_size: int = 1,
        temperature: float = 1.0,
    ) -> str:
        """
        翻译文本

        Args:
            src_text: 源文本
            max_length: 最大生成长度
            beam_size: beam search大小 (1表示贪心搜索)
            temperature: 温度参数，控制生成的随机性

        Returns:
            翻译后的文本

        数据流转：
        源文本 -> 分词编码 -> [batch_size=1, src_seq_len] ->
        编码器 -> [1, src_seq_len, d_model] ->
        解码器逐步生成 -> [1, tgt_seq_len] -> 解码为文本
        """
        logger.info(f"开始翻译: {src_text}")

        with torch.no_grad():
            if beam_size == 1:
                result = self._greedy_decode(src_text, max_length, temperature)
            else:
                result = self._beam_search_decode(src_text, max_length, beam_size, temperature)

        logger.info(f"翻译结果: {result}")
        return result

    def _greedy_decode(self, src_text: str, max_length: int, temperature: float) -> str:
        """
        贪心解码

        Args:
            src_text: 源文本
            max_length: 最大生成长度
            temperature: 温度参数

        Returns:
            翻译结果
        """
        # 1. 编码源序列
        # 使用训练时的最大序列长度，但不填充到最大长度，避免在推理时超长
        max_seq_len = TRANSFORMER_CONFIG.max_seq_len
        src_ids = self.tokenizer.encode(src_text, "src", max_seq_len, pad_to_max=False)
        src_tensor = torch.tensor([src_ids], device=self.device)  # [1, actual_src_len]

        # 2. 编码器前向传播
        encoder_output = self.model.encode(src_tensor, None)  # [1, src_seq_len, d_model]

        # 3. 初始化解码器输入
        bos_id = self.tokenizer.vocab_tgt[self.tokenizer.bos_token]
        eos_id = self.tokenizer.vocab_tgt[self.tokenizer.eos_token]

        # 解码器输入从BOS token开始
        tgt_ids = [bos_id]

        # 4. 逐步生成
        for step in range(max_length):
            # 检查序列长度是否会超过模型限制
            if len(tgt_ids) >= TRANSFORMER_CONFIG.max_seq_len:
                logger.warning(f"达到最大序列长度限制 {TRANSFORMER_CONFIG.max_seq_len}，停止生成")
                break

            # 当前目标序列
            tgt_tensor = torch.tensor([tgt_ids], device=self.device)  # [1, current_len]

            # 创建目标序列掩码
            tgt_len = tgt_tensor.size(1)
            tgt_mask = create_look_ahead_mask(tgt_len, self.device)  # [current_len, current_len]

            # 解码器前向传播
            logits = self.model.decode_step(tgt_tensor, encoder_output, tgt_mask, None)  # [1, current_len, vocab_size]

            # 获取最后一个位置的logits
            next_token_logits = logits[0, -1, :] / temperature  # [vocab_size]

            # 选择下一个token
            if temperature == 1.0:
                next_token_id = torch.argmax(next_token_logits).item()
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()

            # 添加到序列
            tgt_ids.append(next_token_id)

            # 如果生成了EOS token，停止生成
            if next_token_id == eos_id:
                break

            logger.debug(f"生成步骤 {step+1}: token_id={next_token_id}")

        # 5. 解码为文本
        result = self.tokenizer.decode(tgt_ids, "tgt")
        return result

    def _beam_search_decode(self, src_text: str, max_length: int, beam_size: int, temperature: float) -> str:
        """
        Beam search解码

        Args:
            src_text: 源文本
            max_length: 最大生成长度
            beam_size: beam大小
            temperature: 温度参数

        Returns:
            翻译结果
        """
        # 1. 编码源序列
        # 使用训练时的最大序列长度，但不填充到最大长度，避免在推理时超长
        max_seq_len = TRANSFORMER_CONFIG.max_seq_len
        src_ids = self.tokenizer.encode(src_text, "src", max_seq_len, pad_to_max=False)
        src_tensor = torch.tensor([src_ids], device=self.device)  # [1, actual_src_len]

        # 2. 编码器前向传播
        encoder_output = self.model.encode(src_tensor, None)  # [1, src_seq_len, d_model]

        # 3. 初始化beam
        bos_id = self.tokenizer.vocab_tgt[self.tokenizer.bos_token]
        eos_id = self.tokenizer.vocab_tgt[self.tokenizer.eos_token]

        # beam中的候选序列: (序列, 累积分数)
        beams = [([bos_id], 0.0)]
        completed_sequences = []

        # 4. Beam search
        for step in range(max_length):
            candidates = []

            for seq, score in beams:
                # 如果序列已经结束，直接添加到候选中
                if seq[-1] == eos_id:
                    candidates.append((seq, score))
                    continue

                # 当前目标序列
                tgt_tensor = torch.tensor([seq], device=self.device)  # [1, current_len]

                # 创建目标序列掩码
                tgt_len = tgt_tensor.size(1)
                tgt_mask = create_look_ahead_mask(tgt_len, self.device)

                # 解码器前向传播
                logits = self.model.decode_step(tgt_tensor, encoder_output, tgt_mask, None)  # [1, current_len, vocab_size]

                # 获取最后一个位置的logits
                next_token_logits = logits[0, -1, :] / temperature  # [vocab_size]
                log_probs = F.log_softmax(next_token_logits, dim=-1)

                # 选择top-k个候选
                top_k_probs, top_k_ids = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    new_seq = seq + [top_k_ids[i].item()]
                    new_score = score + top_k_probs[i].item()
                    candidates.append((new_seq, new_score))

            # 选择最佳的beam_size个候选
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]

            # 检查是否有完成的序列
            for seq, score in beams:
                if seq[-1] == eos_id and (seq, score) not in completed_sequences:
                    completed_sequences.append((seq, score))

            # 如果有足够的完成序列，可以提前停止
            if len(completed_sequences) >= beam_size:
                break

        # 5. 选择最佳序列
        if completed_sequences:
            best_seq, _ = max(completed_sequences, key=lambda x: x[1])
        else:
            best_seq, _ = max(beams, key=lambda x: x[1])

        # 6. 解码为文本
        result = self.tokenizer.decode(best_seq, "tgt")
        return result

    def translate_batch(self, src_texts: List[str], **kwargs) -> List[str]:
        """
        批量翻译

        Args:
            src_texts: 源文本列表
            **kwargs: 传递给translate方法的参数

        Returns:
            翻译结果列表
        """
        results = []
        for src_text in src_texts:
            result = self.translate(src_text, **kwargs)
            results.append(result)
        return results


def interactive_translation(model_path: str, vocab_dir: Optional[str] = None):
    """
    交互式翻译

    Args:
        model_path: 模型文件路径
        vocab_dir: 词汇表目录路径
    """
    print("🚀 启动交互式翻译器...")

    try:
        # 初始化推理器
        inference = TransformerInference(model_path, vocab_dir)

        print("✅ 推理器初始化完成")
        print("💡 输入 'quit' 退出程序")
        print("💡 输入 'help' 查看帮助")
        print("-" * 50)

        while True:
            try:
                # 获取用户输入
                src_text = input("\n请输入要翻译的文本: ").strip()

                if src_text.lower() == "quit":
                    print("👋 再见!")
                    break
                elif src_text.lower() == "help":
                    print("帮助信息:")
                    print("  - 直接输入文本进行翻译")
                    print("  - 输入 'quit' 退出程序")
                    print("  - 输入 'help' 查看帮助")
                    continue
                elif not src_text:
                    print("❌ 请输入有效的文本")
                    continue

                # 翻译
                print("🔄 正在翻译...")
                result = inference.translate(src_text)
                print(f"✅ 翻译结果: {result}")

            except KeyboardInterrupt:
                print("\n👋 用户中断，退出程序")
                break
            except Exception as e:
                print(f"❌ 翻译出错: {e}")
                logger.error(f"翻译出错: {e}")

    except Exception as e:
        print(f"❌ 推理器初始化失败: {e}")
        logger.error(f"推理器初始化失败: {e}")


if __name__ == "__main__":
    # 测试推理器 - 使用配置中的模型路径
    import os
    model_path = os.path.join(TRAINING_CONFIG.model_save_dir, "best_model.pt")
    print(f"使用模型路径: {model_path}")
    interactive_translation(model_path)
