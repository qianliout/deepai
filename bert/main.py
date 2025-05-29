"""
主运行脚本 - BERT框架的统一入口
支持预训练、微调、推理等所有功能
不需要手动传参，所有参数都在config.py中配置
"""

import argparse
import logging
import sys
from pathlib import Path

from config import print_config, setup_logging, DataConfig, TRAINING_CONFIG
from trainer import BertTrainer
from fine_tuning import BertFineTuner
from inference import BertInference

logger = logging.getLogger("BERT")


def run_pretraining():
    """运行预训练"""
    print("\n🚀 开始BERT预训练")
    print("=" * 50)

    # 打印配置信息
    print_config()

    # 创建并运行训练器
    trainer = BertTrainer()
    history = trainer.train()

    print("\n🎉 预训练完成！")
    print(f"最佳损失: {trainer.best_loss:.4f}")
    print(f"输出目录: {trainer.output_dir}")
    print(f"最佳模型: {trainer.output_dir}/best_model")

    return history


def run_fine_tuning(pretrained_model_path: str = None):
    """运行微调"""
    print("\n🔧 开始BERT微调")
    print("=" * 50)

    # 如果没有指定预训练模型路径，使用默认路径
    if pretrained_model_path is None:
        pretrained_model_path = TRAINING_CONFIG.output_dir + "/" + "best_model"

    pretrained_path = Path(pretrained_model_path)
    if not pretrained_path.exists():
        print(f"❌ 预训练模型路径不存在: {pretrained_path}")
        print("请先运行预训练或指定正确的模型路径")
        return None

    print(f"使用预训练模型: {pretrained_path}")

    # 创建并运行微调器
    fine_tuner = BertFineTuner(pretrained_model_path, num_labels=2)
    history = fine_tuner.fine_tune()

    print("\n🎉 微调完成！")
    print(f"最佳准确率: {fine_tuner.best_accuracy:.4f}")
    print(f"输出目录: {fine_tuner.output_dir}")
    print(f"最佳模型: {fine_tuner.output_dir}/best_model")

    return history


def run_inference(model_path: str = None, model_type: str = "pretraining"):
    """运行推理"""
    print("\n🔍 开始BERT推理")
    print("=" * 50)

    # 如果没有指定模型路径，使用配置中的默认路径
    if model_path is None:
        if model_type == "pretraining":
            model_path = os.path.join(TRAINING_CONFIG.model_save_dir, "best_model")
        else:
            model_path = os.path.join(TRAINING_CONFIG.fine_tuning_save_dir, "best_model")

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"❌ 模型路径不存在: {model_path_obj}")
        print("请先运行训练或指定正确的模型路径")
        return

    print(f"使用模型: {model_path_obj}")
    print(f"模型类型: {model_type}")

    # 创建推理器
    inference = BertInference(model_path, model_type)

    # 交互式推理
    print("\n开始交互式推理...")
    print("输入 'quit' 退出")

    while True:
        try:
            if model_type == "pretraining":
                print("\n--- 掩码语言模型预测 ---")
                text = input("请输入包含[MASK]的文本: ")
                if text.lower() == "quit":
                    break

                results = inference.predict_masked_tokens(text, top_k=3)
                if results:
                    for result in results:
                        print(f"\n位置 {result['position']} 的预测:")
                        for i, pred in enumerate(result["predictions"], 1):
                            print(f"  {i}. {pred['token']}: {pred['probability']:.4f}")
                else:
                    print("没有找到[MASK] token")

            elif model_type == "classification":
                print("\n--- 文本分类预测 ---")
                text = input("请输入要分类的文本: ")
                if text.lower() == "quit":
                    break

                result = inference.classify_text(text)
                print(f"\n预测结果:")
                print(f"  类别: {result['predicted_class']}")
                print(f"  置信度: {result['confidence']:.4f}")

                # 显示所有类别的概率
                print(f"  所有概率:")
                for i, prob in enumerate(result["all_probabilities"]):
                    print(f"    类别 {i}: {prob:.4f}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

    print("\n推理器已退出")


def run_full_pipeline():
    """运行完整流程：预训练 + 微调"""
    print("\n🔄 开始完整BERT流程")
    print("=" * 50)

    # 1. 预训练
    print("\n第一步：预训练")
    pretrain_history = run_pretraining()

    if pretrain_history is None:
        print("❌ 预训练失败，停止流程")
        return

    # 2. 微调
    print("\n第二步：微调")
    finetune_history = run_fine_tuning()

    if finetune_history is None:
        print("❌ 微调失败")
        return

    print("\n🎉 完整流程完成！")
    print("可以使用以下命令进行推理：")
    print("  预训练模型推理: python main.py inference --model_type pretraining")
    print("  分类模型推理: python main.py inference --model_type classification")


def run_quick_test():
    """快速测试 - 使用小规模配置"""
    print("\n⚡ 快速测试模式")
    print("=" * 50)

    # 临时修改配置为小规模
    from config import BERT_CONFIG, TRAINING_CONFIG

    # 保存原始配置
    original_config = {
        "hidden_size": BERT_CONFIG.hidden_size,
        "num_hidden_layers": BERT_CONFIG.num_hidden_layers,
        "num_attention_heads": BERT_CONFIG.num_attention_heads,
        "intermediate_size": BERT_CONFIG.intermediate_size,
        "num_epochs": TRAINING_CONFIG.num_epochs,
        "max_samples": TRAINING_CONFIG.max_samples,
        "batch_size": TRAINING_CONFIG.batch_size,
    }

    # 设置小规模配置
    BERT_CONFIG.hidden_size = 256
    BERT_CONFIG.num_hidden_layers = 4
    BERT_CONFIG.num_attention_heads = 4
    BERT_CONFIG.intermediate_size = 1024
    TRAINING_CONFIG.num_epochs = 1
    TRAINING_CONFIG.max_samples = 100
    TRAINING_CONFIG.batch_size = 8
    # 更新快速测试的保存目录
    TRAINING_CONFIG.model_save_dir = "/Users/liuqianli/work/python/deepai/saved_model/bert_quick_test"
    TRAINING_CONFIG.fine_tuning_save_dir = "/Users/liuqianli/work/python/deepai/saved_model/bert_quick_test/fine_tuning"
    TRAINING_CONFIG.log_dir = "/Users/liuqianli/work/python/deepai/logs/bert_quick_test"

    print("使用小规模配置进行快速测试...")

    try:
        # 运行预训练
        pretrain_history = run_pretraining()

        if pretrain_history:
            # 运行微调
            run_fine_tuning()

        print("\n⚡ 快速测试完成！")

    finally:
        # 恢复原始配置
        for key, value in original_config.items():
            if hasattr(BERT_CONFIG, key):
                setattr(BERT_CONFIG, key, value)
            elif hasattr(TRAINING_CONFIG, key):
                setattr(TRAINING_CONFIG, key, value)


def main():
    """主函数"""
    # 设置日志
    setup_logging()

    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="BERT框架 - 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py pretrain                    # 预训练
  python main.py finetune                    # 微调（使用默认预训练模型）
  python main.py inference                   # 推理（预训练模型）
  python main.py inference --model_type classification  # 推理（分类模型）
  python main.py full                        # 完整流程
  python main.py quick                       # 快速测试

注意：所有参数都在config.py中配置，无需手动传参
        """,
    )

    # 添加子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 预训练命令
    pretrain_parser = subparsers.add_parser("pretrain", help="运行预训练")

    # 微调命令
    finetune_parser = subparsers.add_parser("finetune", help="运行微调")
    finetune_parser.add_argument(
        "--pretrained_model_path",
        type=str,
        help="预训练模型路径（默认: 使用配置中的路径）",
    )

    # 推理命令
    inference_parser = subparsers.add_parser("inference", help="运行推理")
    inference_parser.add_argument("--model_path", type=str, help="模型路径")
    inference_parser.add_argument(
        "--model_type",
        type=str,
        choices=["pretraining", "classification"],
        default="pretraining",
        help="模型类型（默认: pretraining）",
    )

    # 完整流程命令
    full_parser = subparsers.add_parser("full", help="运行完整流程（预训练+微调）")

    # 快速测试命令
    quick_parser = subparsers.add_parser("quick", help="快速测试（小规模配置）")

    # 解析参数
    args = parser.parse_args()

    # 如果没有提供命令，显示帮助
    if args.command is None:
        parser.print_help()
        return

    try:
        # 执行对应的命令
        if args.command == "pretrain":
            run_pretraining()

        elif args.command == "finetune":
            run_fine_tuning(args.pretrained_model_path)

        elif args.command == "inference":
            run_inference(args.model_path, args.model_type)

        elif args.command == "full":
            run_full_pipeline()

        elif args.command == "quick":
            run_quick_test()

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
