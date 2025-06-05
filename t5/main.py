"""
T5框架主运行脚本 - 统一入口
支持训练、推理等所有功能
不需要手动传参，所有参数都在config.py中配置
"""

import argparse
import logging
import sys
from pathlib import Path

from config import print_config, setup_logging, TRAINING_CONFIG, T5_CONFIG
from trainer import T5Trainer
from inference import T5Inference, GenerationConfig

logger = logging.getLogger("T5")


def run_training():
    """运行训练"""
    print("\n🚀 开始T5训练")
    print("=" * 50)
    
    # 打印配置信息
    print_config()
    
    # 创建并运行训练器
    trainer = T5Trainer()
    history = trainer.train()
    
    print("\n🎉 训练完成！")
    print(f"最佳损失: {trainer.best_loss:.4f}")
    print(f"检查点目录: {trainer.checkpoints_dir}")
    print(f"最佳模型目录: {trainer.best_model_dir}")
    print(f"最终模型目录: {trainer.final_model_dir}")
    
    return history


def run_inference(model_path: str = None):
    """运行推理"""
    print("\n🔍 开始T5推理")
    print("=" * 50)
    
    # 如果没有指定模型路径，使用默认路径
    if model_path is None:
        model_path = TRAINING_CONFIG.pretrain_best_dir
    
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"❌ 模型路径不存在: {model_path_obj}")
        print("请先运行训练或指定正确的模型路径")
        return
    
    print(f"使用模型: {model_path_obj}")
    
    # 创建推理器
    inference = T5Inference(model_path)
    
    # 交互式推理
    print("\n开始交互式推理...")
    print("支持的任务:")
    print("1. 问答 (qa)")
    print("2. 摘要 (summarize)")
    print("3. 翻译 (translate)")
    print("4. 自由生成 (generate)")
    print("输入 'quit' 退出")
    
    while True:
        try:
            print("\n" + "-" * 30)
            task = input("请选择任务类型 (qa/summarize/translate/generate): ").strip().lower()
            
            if task == "quit":
                break
            
            if task == "qa":
                print("\n--- 问答任务 ---")
                question = input("请输入问题: ")
                context = input("请输入上下文: ")
                
                if question.lower() == "quit" or context.lower() == "quit":
                    break
                
                result = inference.answer_question(question, context)
                print(f"\n答案: {result}")
                
            elif task == "summarize":
                print("\n--- 摘要任务 ---")
                text = input("请输入要摘要的文本: ")
                
                if text.lower() == "quit":
                    break
                
                result = inference.summarize_text(text)
                print(f"\n摘要: {result}")
                
            elif task == "translate":
                print("\n--- 翻译任务 ---")
                text = input("请输入要翻译的文本: ")
                source_lang = input("源语言 (默认: English): ").strip() or "English"
                target_lang = input("目标语言 (默认: German): ").strip() or "German"
                
                if text.lower() == "quit":
                    break
                
                result = inference.translate_text(text, source_lang, target_lang)
                print(f"\n翻译结果: {result}")
                
            elif task == "generate":
                print("\n--- 自由生成 ---")
                text = input("请输入文本: ")
                task_prefix = input("任务前缀 (可选): ").strip()
                
                if text.lower() == "quit":
                    break
                
                if not task_prefix:
                    task_prefix = None
                
                result = inference.generate(text, task_prefix=task_prefix)
                print(f"\n生成结果: {result}")
                
            else:
                print("❌ 不支持的任务类型，请选择: qa/summarize/translate/generate")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    print("\n推理器已退出")


def run_quick_test():
    """快速测试 - 使用小规模配置"""
    print("\n⚡ 快速测试模式")
    print("=" * 50)
    
    # 临时修改配置为小规模
    from config import T5_CONFIG, TRAINING_CONFIG
    
    # 保存原始配置
    original_config = {
        "d_model": T5_CONFIG.d_model,
        "num_layers": T5_CONFIG.num_layers,
        "num_heads": T5_CONFIG.num_heads,
        "d_ff": T5_CONFIG.d_ff,
        "num_epochs": TRAINING_CONFIG.num_epochs,
        "max_samples": TRAINING_CONFIG.max_samples,
        "batch_size": TRAINING_CONFIG.batch_size,
    }
    
    # 设置小规模配置
    T5_CONFIG.d_model = 256
    T5_CONFIG.num_layers = 2
    T5_CONFIG.num_heads = 4
    T5_CONFIG.d_ff = 1024
    T5_CONFIG.d_kv = T5_CONFIG.d_model // T5_CONFIG.num_heads
    TRAINING_CONFIG.num_epochs = 1
    TRAINING_CONFIG.max_samples = 50
    TRAINING_CONFIG.batch_size = 4
    
    # 更新快速测试的保存目录
    TRAINING_CONFIG.pretrain_checkpoints_dir = "/Users/liuqianli/work/python/deepai/saved_model/t5_quick_test/pretrain/checkpoints"
    TRAINING_CONFIG.pretrain_best_dir = "/Users/liuqianli/work/python/deepai/saved_model/t5_quick_test/pretrain/best"
    TRAINING_CONFIG.pretrain_final_dir = "/Users/liuqianli/work/python/deepai/saved_model/t5_quick_test/pretrain/final"
    TRAINING_CONFIG.log_dir = "/Users/liuqianli/work/python/deepai/logs/t5_quick_test"
    
    print("使用小规模配置进行快速测试...")
    
    try:
        # 运行训练
        training_history = run_training()
        
        if training_history:
            print("\n⚡ 快速测试训练完成！")
            
            # 简单推理测试
            print("\n测试推理功能...")
            model_path = TRAINING_CONFIG.pretrain_best_dir
            if Path(model_path).exists():
                inference = T5Inference(model_path)
                test_result = inference.generate("Hello world", task_prefix="translate English to German: ")
                print(f"测试生成结果: {test_result}")
            else:
                print("❌ 模型文件不存在，跳过推理测试")
        
        print("\n⚡ 快速测试完成！")
        
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        logger.error(f"快速测试失败: {e}")
    
    finally:
        # 恢复原始配置
        for key, value in original_config.items():
            if hasattr(T5_CONFIG, key):
                setattr(T5_CONFIG, key, value)
            elif hasattr(TRAINING_CONFIG, key):
                setattr(TRAINING_CONFIG, key, value)


def run_demo():
    """运行演示"""
    print("\n🎭 T5演示模式")
    print("=" * 50)
    
    # 检查是否有训练好的模型
    model_path = TRAINING_CONFIG.pretrain_best_dir
    if not Path(model_path).exists():
        print("❌ 未找到训练好的模型")
        print("请先运行: python main.py train")
        return
    
    # 创建推理器
    inference = T5Inference(model_path)
    
    # 演示不同任务
    print("\n🔍 演示不同的T5任务:")
    
    # 1. 问答演示
    print("\n1. 问答任务演示:")
    question = "What is the capital of France?"
    context = "France is a country in Europe. Its capital city is Paris, which is known for the Eiffel Tower."
    answer = inference.answer_question(question, context)
    print(f"问题: {question}")
    print(f"上下文: {context}")
    print(f"答案: {answer}")
    
    # 2. 摘要演示
    print("\n2. 摘要任务演示:")
    text = "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents."
    summary = inference.summarize_text(text)
    print(f"原文: {text}")
    print(f"摘要: {summary}")
    
    # 3. 翻译演示
    print("\n3. 翻译任务演示:")
    text = "Hello, how are you today?"
    translation = inference.translate_text(text, "English", "German")
    print(f"英文: {text}")
    print(f"德文: {translation}")
    
    print("\n🎭 演示完成！")


def main():
    """主函数"""
    # 设置日志
    setup_logging()
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="T5框架 - 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py train                       # 训练模型
  python main.py inference                   # 推理（使用默认模型）
  python main.py inference --model_path /path/to/model  # 推理（指定模型）
  python main.py quick                       # 快速测试
  python main.py demo                        # 演示模式

注意：所有参数都在config.py中配置，无需手动传参
        """,
    )
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="运行训练")
    
    # 推理命令
    inference_parser = subparsers.add_parser("inference", help="运行推理")
    inference_parser.add_argument("--model_path", type=str, help="模型路径")
    
    # 快速测试命令
    quick_parser = subparsers.add_parser("quick", help="快速测试（小规模配置）")
    
    # 演示命令
    demo_parser = subparsers.add_parser("demo", help="演示模式")
    
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
            run_inference(args.model_path)
            
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
