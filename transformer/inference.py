"""
推理脚本 - 用于翻译测试
"""
import torch
import os
import argparse
import logging
from typing import Optional
from config import Config, default_config
from model import Transformer
from tokenizer import SimpleTokenizer
from utils import get_device


class TranslationInference:
    """翻译推理类"""

    def __init__(self, model_path: str, config: Config):
        """
        初始化推理器

        Args:
            model_path: 模型路径
            config: 配置对象
        """
        self.config = config
        self.device = get_device()
        self.logger = logging.getLogger('transformer.inference')

        # 加载分词器
        self.tokenizer = SimpleTokenizer(config.model)
        self.tokenizer.load_vocabs(config.training.vocab_save_path)

        # 更新配置中的词汇表大小为实际大小
        config.model.vocab_size_en = len(self.tokenizer.vocab_en)
        config.model.vocab_size_it = len(self.tokenizer.vocab_it)

        # 创建和加载模型
        self.model = Transformer(config.model).to(self.device)
        self.load_model(model_path)

        self.logger.info(f"推理器初始化完成，使用设备: {self.device}")

    def load_model(self, model_path: str):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.logger.info(f"模型已加载: {model_path}")
        if 'epoch' in checkpoint:
            self.logger.info(f"训练轮数: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            self.logger.info(f"损失: {checkpoint['loss']:.4f}")

    def translate(self, text: str, max_length: Optional[int] = None, beam_size: int = 1) -> str:
        """
        翻译文本

        Args:
            text: 输入英语文本
            max_length: 最大生成长度
            beam_size: beam search大小（暂时只支持贪心解码）

        Returns:
            翻译后的意大利语文本
        """
        if max_length is None:
            max_length = self.config.model.max_seq_len

        with torch.no_grad():
            # 编码输入
            en_ids = self.tokenizer.encode(text, 'en', max_length)
            # 这里写成[en_ids] 的意思就是batch_size=1
            src = torch.tensor([en_ids], device=self.device)

            # 编码
            encoder_output = self.model.encode(src)

            # 贪心解码
            if beam_size == 1:
                return self._greedy_decode(encoder_output, max_length)
            else:
                # TODO: 实现beam search
                return self._greedy_decode(encoder_output, max_length)

    def _greedy_decode(self, encoder_output: torch.Tensor, max_length: int) -> str:
        """
        贪心解码

        Args:
            encoder_output: 编码器输出
            max_length: 最大长度

        Returns:
            解码后的文本
        """
        # 初始化解码器输入（只有BOS token）
        tgt = torch.tensor([[self.tokenizer.bos_id]], device=self.device)

        for _ in range(max_length - 1):
            # 解码一步
            output = self.model.decode_step(tgt, encoder_output)

            # 获取下一个token（贪心选择概率最大的）
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

            # 添加到序列
            tgt = torch.cat([tgt, next_token], dim=1)

            # 如果生成了EOS token，停止
            if next_token.item() == self.tokenizer.eos_id:
                break

        # 解码为文本
        # tgt.shape = [1, seq_len],所以取tgt[0]
        it_ids = tgt[0].cpu().tolist()
        it_text = self.tokenizer.decode(it_ids, 'it')

        return it_text

    def interactive_translate(self):
        """交互式翻译"""
        print("\n" + "="*50)
        print("交互式英语到意大利语翻译")
        print("输入 'quit' 或 'exit' 退出")
        print("="*50)

        while True:
            try:
                # 获取用户输入
                en_text = input("\n请输入英语句子: ").strip()

                # 检查退出条件
                if en_text.lower() in ['quit', 'exit', 'q']:
                    print("再见!")
                    break

                if not en_text:
                    print("请输入有效的句子")
                    continue

                # 翻译
                print("正在翻译...")
                it_text = self.translate(en_text)

                print(f"英语: {en_text}")
                print(f"意大利语: {it_text}")

            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                print(f"翻译出错: {e}")

    def batch_translate(self, sentences: list) -> list:
        """
        批量翻译

        Args:
            sentences: 英语句子列表

        Returns:
            意大利语翻译列表
        """
        translations = []

        self.logger.info(f"开始批量翻译 {len(sentences)} 个句子...")

        for i, sentence in enumerate(sentences):
            try:
                translation = self.translate(sentence)
                translations.append(translation)
                self.logger.info(f"[{i+1}/{len(sentences)}] {sentence} -> {translation}")
            except Exception as e:
                self.logger.error(f"[{i+1}/{len(sentences)}] 翻译失败: {e}")
                translations.append("")

        return translations


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Transformer翻译推理")
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['interactive', 'single', 'batch'],
                       default='interactive', help='推理模式')
    parser.add_argument('--text', type=str, help='单句翻译的输入文本')
    parser.add_argument('--input_file', type=str, help='批量翻译的输入文件')
    parser.add_argument('--output_file', type=str, help='批量翻译的输出文件')

    args = parser.parse_args()

    # 加载配置
    if args.config and os.path.exists(args.config):
        config = Config.load_config(args.config)
    else:
        config = default_config

    try:
        # 创建推理器
        inference = TranslationInference(args.model_path, config)

        if args.mode == 'interactive':
            # 交互式模式
            inference.interactive_translate()

        elif args.mode == 'single':
            # 单句翻译
            if not args.text:
                print("单句翻译模式需要指定 --text 参数")
                return

            translation = inference.translate(args.text)
            print(f"英语: {args.text}")
            print(f"意大利语: {translation}")

        elif args.mode == 'batch':
            # 批量翻译
            if not args.input_file:
                print("批量翻译模式需要指定 --input_file 参数")
                return

            if not os.path.exists(args.input_file):
                print(f"输入文件不存在: {args.input_file}")
                return

            # 读取输入文件
            with open(args.input_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]

            # 批量翻译
            translations = inference.batch_translate(sentences)

            # 保存结果
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for en, it in zip(sentences, translations):
                        f.write(f"{en}\t{it}\n")
                print(f"翻译结果已保存到: {args.output_file}")
            else:
                print("\n翻译结果:")
                for en, it in zip(sentences, translations):
                    print(f"{en} -> {it}")

    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
