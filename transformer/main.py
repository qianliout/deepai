"""
主入口文件 - 一键运行训练和测试
"""
import os
import sys
import torch
import argparse
import logging
from config import Config, default_config, create_directories
from trainer import Trainer
from utils import setup_logging, get_device


def setup_environment():
    """设置环境"""
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 检查设备
    device = get_device()
    print(f"检测到设备: {device}")

    if device == "mps":
        print("使用Apple Silicon GPU (MPS)")
    elif device == "cuda":
        print("使用NVIDIA GPU")
    else:
        print("使用CPU")

    return device


def create_directories(config: Config):
    """创建必要的目录"""
    directories = [
        config.training.model_save_path,
        config.training.vocab_save_path,
        config.training.log_dir,
        config.data.cache_dir
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")


def print_config(config: Config):
    """打印配置信息"""
    print("\n" + "="*50)
    print("配置信息")
    print("="*50)

    print(f"模型配置:")
    print(f"  - 模型维度: {config.model.d_model}")
    print(f"  - 注意力头数: {config.model.n_heads}")
    print(f"  - 编码器/解码器层数: {config.model.n_layers}")
    print(f"  - 前馈网络维度: {config.model.d_ff}")
    print(f"  - 最大序列长度: {config.model.max_seq_len}")
    print(f"  - Dropout: {config.model.dropout}")

    print(f"\n训练配置:")
    print(f"  - 训练数据大小: {config.training.train_size}")
    print(f"  - 验证数据大小: {config.training.val_size}")
    print(f"  - 批次大小: {config.training.batch_size}")
    print(f"  - 学习率: {config.training.learning_rate}")
    print(f"  - 训练轮数: {config.training.num_epochs}")
    print(f"  - 设备: {config.training.device}")

    print(f"\n数据配置:")
    print(f"  - 数据集: {config.data.dataset_name}")
    print(f"  - 语言对: {config.data.language_pair}")
    print(f"  - 最小词频: {config.data.min_freq}")
    print(f"  - 最大词汇表大小: {config.data.max_vocab_size}")

    print("="*50 + "\n")


def simple_translate_test(trainer: Trainer):
    """简单的翻译测试"""
    print("\n" + "="*50)
    print("简单翻译测试")
    print("="*50)

    # 获取分词器
    tokenizer = trainer.get_tokenizer()
    model = trainer.model
    device = trainer.device

    # 测试句子
    test_sentences = [
        "Hello, how are you?",
        "I love programming.",
        "The weather is nice today.",
        "Thank you very much.",
        "Good morning!"
    ]

    model.eval()
    with torch.no_grad():
        for en_text in test_sentences:
            print(f"\n英语: {en_text}")

            try:
                # 编码输入
                en_ids = tokenizer.encode(en_text, 'en', tokenizer.config.max_seq_len)
                src = torch.tensor([en_ids], device=device)

                # 编码
                encoder_output = model.encode(src)

                # 简单的贪心解码
                max_len = tokenizer.config.max_seq_len
                tgt = torch.tensor([[tokenizer.bos_id]], device=device)

                for _ in range(max_len - 1):
                    output = model.decode_step(tgt, encoder_output)
                    next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                    tgt = torch.cat([tgt, next_token], dim=1)

                    # 如果生成了EOS token，停止
                    if next_token.item() == tokenizer.eos_id:
                        break

                # 解码输出
                it_ids = tgt[0].cpu().tolist()
                it_text = tokenizer.decode(it_ids, 'it')
                print(f"意大利语: {it_text}")

            except Exception as e:
                print(f"翻译失败: {e}")

    print("="*50 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Transformer训练和测试")
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                       help='运行模式: train(训练) 或 test(测试)')
    parser.add_argument('--model_path', type=str, help='测试时使用的模型路径')

    args = parser.parse_args()

    print("Transformer从零实现 - 英语到意大利语翻译")
    print("="*60)

    # 设置环境
    device = setup_environment()

    # 加载配置
    if args.config and os.path.exists(args.config):
        config = Config.load_config(args.config)
        print(f"从文件加载配置: {args.config}")
    else:
        config = default_config
        print("使用默认配置")

    # 更新设备配置
    config.training.device = device

    # 创建目录
    create_directories()

    # 打印配置信息
    print(f"模型保存目录: {config.training.model_save_dir}")
    print(f"日志保存目录: {config.training.log_dir}")
    print(f"数据缓存目录: {config.training.cache_dir}")

    if args.mode == 'train':
        # 训练模式
        print("开始训练...")

        try:
            # 创建训练器
            trainer = Trainer(config)

            # 开始训练
            trainer.train()

            print("\n训练完成!")

            # 简单测试
            print("\n进行简单翻译测试...")
            simple_translate_test(trainer)

        except KeyboardInterrupt:
            print("\n训练被用户中断")
        except Exception as e:
            print(f"\n训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

    elif args.mode == 'test':
        # 测试模式
        if not args.model_path:
            print("测试模式需要指定模型路径 --model_path")
            return

        if not os.path.exists(args.model_path):
            print(f"模型文件不存在: {args.model_path}")
            return

        print(f"加载模型进行测试: {args.model_path}")

        try:
            # 创建训练器（用于加载模型）
            trainer = Trainer(config)
            trainer.setup_model()

            # 加载模型
            checkpoint = torch.load(args.model_path, map_location=device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])

            print("模型加载成功")

            # 进行测试
            simple_translate_test(trainer)

        except Exception as e:
            print(f"测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
