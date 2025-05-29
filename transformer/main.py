"""
Transformer主入口文件 - 统一的训练和推理入口
支持训练、测试、推理等所有功能
"""
import os
import sys
import torch
import argparse

from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, create_directories, get_device, print_config
from trainer import Trainer


def setup_environment():
    """设置环境"""
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 检查设备
    device = get_device()
    print(f"检测到设备: {device}")

    if str(device) == "mps":
        print("使用Apple Silicon GPU (MPS)")
    elif str(device) == "cuda":
        print("使用NVIDIA GPU")
    else:
        print("使用CPU")

    return device


def check_dependencies():
    """检查依赖包"""
    print("检查依赖包...")

    required_packages = ["torch", "pydantic", "datasets", "numpy", "tqdm"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (缺失)")

    if missing_packages:
        print(f"\n缺失依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False

    print("所有依赖包已安装 ✓")
    return True


def simple_translate_test(trainer: Trainer):
    """简单的翻译测试"""
    test_sentences = [
        "Hello, how are you?",
        "I love programming.",
        "The weather is nice today.",
        "Thank you very much.",
        "Good morning!"
    ]

    print("\n" + "="*50)
    print("简单翻译测试")
    print("="*50)

    for sentence in test_sentences:
        try:
            translation = trainer.translate(sentence)
            print(f"英语: {sentence}")
            print(f"意大利语: {translation}")
            print("-" * 30)
        except Exception as e:
            print(f"翻译失败: {sentence} -> 错误: {e}")


def train_model():
    """训练模型"""
    print("\n开始训练...")

    # 创建训练器
    trainer = Trainer()

    # 训练
    trainer.train()

    print("训练完成!")

    # 简单测试
    simple_translate_test(trainer)


def test_model(model_path: str = None):
    """测试模型"""
    print("\n开始测试...")

    # 创建训练器
    trainer = Trainer()

    # 加载模型
    if model_path and os.path.exists(model_path):
        trainer.load_model(model_path)
        print(f"加载模型: {model_path}")
    else:
        # 查找最新的模型
        model_dir = TRAINING_CONFIG.pretrain_best_dir
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            if model_files:
                model_file = sorted(model_files)[-1]
                model_path = os.path.join(model_dir, model_file)
                trainer.load_model(model_path)
                print(f"自动加载模型: {model_path}")
            else:
                print("未找到模型文件，请先训练模型")
                return
        else:
            print("模型目录不存在，请先训练模型")
            return

    # 测试
    simple_translate_test(trainer)


def interactive_translate():
    """交互式翻译"""
    print("\n" + "=" * 60)
    print("交互式翻译")
    print("=" * 60)

    # 查找模型文件
    model_dir = TRAINING_CONFIG.pretrain_best_dir
    if not os.path.exists(model_dir):
        print("未找到模型目录，请先训练模型")
        return

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not model_files:
        print("未找到模型文件，请先训练模型")
        return

    # 选择模型
    model_file = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, model_file)
    print(f"使用模型: {model_path}")

    # 创建训练器并加载模型
    trainer = Trainer()
    trainer.load_model(model_path)

    print("输入英语句子进行翻译，输入 'quit' 退出")
    print("-" * 60)

    try:
        while True:
            text = input("英语: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if text:
                try:
                    translation = trainer.translate(text)
                    print(f"意大利语: {translation}")
                except Exception as e:
                    print(f"翻译失败: {e}")
            print("-" * 30)
    except KeyboardInterrupt:
        print("\n退出交互式翻译")


def quick_test():
    """快速测试模式 - 使用较小的参数快速验证模型流程"""
    print("\n🚀 快速测试模式")
    print("=" * 60)
    
    # 更新配置为快速测试参数
    TRAINING_CONFIG.train_size = 1000
    TRAINING_CONFIG.val_size = 200
    TRAINING_CONFIG.batch_size = 16
    TRAINING_CONFIG.num_epochs = 1
    TRAINING_CONFIG.log_interval = 50
    TRAINING_CONFIG.save_interval = 500
    
    # 更新目录为快速测试目录
    base_dir = "/Users/liuqianli/work/python/deepai/saved_model/transformer/quick_test"
    TRAINING_CONFIG.pretrain_checkpoints_dir = f"{base_dir}/pretrain/checkpoints"
    TRAINING_CONFIG.pretrain_best_dir = f"{base_dir}/pretrain/best"
    TRAINING_CONFIG.pretrain_final_dir = f"{base_dir}/pretrain/final"
    TRAINING_CONFIG.pretrain_vocab_dir = f"{base_dir}/pretrain/vocab"
    TRAINING_CONFIG.log_dir = f"{base_dir}/logs"
    
    print("快速测试配置:")
    print(f"  训练数据: {TRAINING_CONFIG.train_size}")
    print(f"  验证数据: {TRAINING_CONFIG.val_size}")
    print(f"  批次大小: {TRAINING_CONFIG.batch_size}")
    print(f"  训练轮数: {TRAINING_CONFIG.num_epochs}")
    print(f"  保存目录: {base_dir}")
    
    # 创建目录
    create_directories()
    
    # 训练模型
    train_model()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Transformer训练和推理")
    parser.add_argument("mode", nargs="?", default="train", 
                       choices=["train", "test", "interactive", "quick"], 
                       help="运行模式: train(训练), test(测试), interactive(交互式翻译), quick(快速测试)")
    parser.add_argument("--model_path", type=str, help="模型文件路径（用于测试）")

    args = parser.parse_args()

    # 设置环境
    device = setup_environment()

    # 检查依赖
    if not check_dependencies():
        return

    # 打印配置
    print_config()

    # 创建目录
    if args.mode != "quick":
        create_directories()

    # 根据模式执行
    if args.mode == "train":
        train_model()
    elif args.mode == "test":
        test_model(args.model_path)
    elif args.mode == "interactive":
        interactive_translate()
    elif args.mode == "quick":
        quick_test()


if __name__ == "__main__":
    main()
