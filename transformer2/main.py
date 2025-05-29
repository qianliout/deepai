"""
主运行脚本 - Transformer2框架的统一入口
支持训练、推理等所有功能
不需要手动传参，所有参数都在config.py中配置
参考bert2的实现方式，简化运行命令
"""

import argparse
import logging
import sys
from pathlib import Path

from config import print_config, setup_logging, update_config_for_quick_test
from trainer import Trainer
from inference import TransformerInference, interactive_translation

logger = logging.getLogger("Transformer2")


def run_training():
    """运行训练"""
    print("\n🚀 开始Transformer训练")
    print("=" * 50)

    # 打印配置信息
    print_config()

    # 创建并运行训练器
    trainer = Trainer()
    history = trainer.train()

    print("\n🎉 训练完成！")
    print(f"输出目录: {trainer.output_dir}")
    print(f"最佳模型: {trainer.output_dir}/best_model.pt")

    return history


def run_inference(model_path: str = None, interactive: bool = True):
    """运行推理"""
    print("\n🔍 开始Transformer推理")
    print("=" * 50)

    # 如果没有指定模型路径，使用默认路径
    if model_path is None:
        model_path = "./transformer2_output/best_model.pt"

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"❌ 模型路径不存在: {model_path_obj}")
        print("请先运行训练或指定正确的模型路径")
        return

    print(f"使用模型: {model_path_obj}")

    if interactive:
        # 交互式推理
        interactive_translation(str(model_path_obj))
    else:
        # 批量推理示例
        try:
            inference = TransformerInference(str(model_path_obj))

            # 测试句子
            test_sentences = [
                "Hello, how are you?",
                "I love programming.",
                "The weather is nice today.",
                "Thank you very much.",
                "Good morning!",
            ]

            print("\n📝 批量翻译测试:")
            for i, src_text in enumerate(test_sentences, 1):
                print(f"\n{i}. 源文本: {src_text}")
                try:
                    result = inference.translate(src_text)
                    print(f"   翻译: {result}")
                except Exception as e:
                    print(f"   错误: {e}")

        except Exception as e:
            print(f"❌ 推理失败: {e}")


def run_quick_test():
    """快速测试 - 使用小规模配置"""
    print("\n⚡ 快速测试模式")
    print("=" * 50)

    # 切换到快速测试配置
    update_config_for_quick_test()

    print("使用小规模配置进行快速测试...")

    try:
        # 运行训练
        print("\n第一步：训练模型")
        history = run_training()

        if history:
            # 运行推理测试
            print("\n第二步：推理测试")
            run_inference(interactive=False)

        print("\n⚡ 快速测试完成！")

    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        logger.error(f"快速测试失败: {e}")


def run_demo():
    """运行演示"""
    print("\n🎭 Transformer2 演示")
    print("=" * 50)

    print("这是一个从零实现的Transformer框架演示")
    print("\n特性:")
    print("  ✅ 完整的Transformer架构 (编码器-解码器)")
    print("  ✅ 多头注意力机制")
    print("  ✅ 位置编码")
    print("  ✅ 残差连接和层归一化")
    print("  ✅ 标签平滑损失")
    print("  ✅ 学习率预热调度")
    print("  ✅ 贪心解码和Beam Search")
    print("  ✅ 详细的数据流转注释")
    print("  ✅ 配置驱动，无需手动传参")

    print("\n可用命令:")
    print("  python main.py train       # 训练模型")
    print("  python main.py inference   # 交互式推理")
    print("  python main.py quick       # 快速测试")
    print("  python main.py demo        # 查看演示")

    print("\n配置文件: config.py")
    print("所有超参数都在配置文件中定义，修改配置即可调整模型")


def main():
    """主函数"""
    # 设置日志
    setup_logging()

    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="Transformer2框架 - 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py train                    # 训练模型
  python main.py inference               # 交互式推理
  python main.py inference --model_path ./path/to/model.pt  # 指定模型推理
  python main.py quick                   # 快速测试
  python main.py demo                    # 查看演示

注意：所有参数都在config.py中配置，无需手动传参
        """,
    )

    # 添加子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 训练命令
    train_parser = subparsers.add_parser("train", help="运行训练")

    # 推理命令
    inference_parser = subparsers.add_parser("inference", help="运行推理")
    inference_parser.add_argument("--model_path", type=str, help="模型路径（默认: ./transformer2_output/best_model.pt）")
    inference_parser.add_argument("--batch", action="store_true", help="批量推理模式（默认: 交互式）")

    # 快速测试命令
    quick_parser = subparsers.add_parser("quick", help="快速测试（小规模配置）")

    # 演示命令
    demo_parser = subparsers.add_parser("demo", help="查看演示信息")

    # 解析参数
    args = parser.parse_args()

    # 如果没有提供命令，显示帮助
    if args.command is None:
        parser.print_help()
        return

    try:
        # 执行对应的命令
        if args.command == "train":
            run_training()

        elif args.command == "inference":
            interactive_mode = not args.batch
            run_inference(args.model_path, interactive_mode)

        elif args.command == "quick":
            run_quick_test()

        elif args.command == "demo":
            run_demo()

        else:
            print(f"❌ 未知命令: {args.command}")
            parser.print_help()

    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断操作")
    except Exception as e:
        logger.error(f"执行失败: {e}")
        print(f"❌ 执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
