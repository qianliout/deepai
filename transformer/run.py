#!/usr/bin/env python3
"""
一键运行脚本 - 简化训练和推理流程
"""
import os
import sys
import subprocess
import argparse


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


def train_model():
    """训练模型"""
    print("\n" + "=" * 60)
    print("开始训练Transformer模型")
    print("=" * 60)

    if not check_dependencies():
        return

    try:
        # 运行训练
        subprocess.run([sys.executable, "main.py", "--mode", "train"], check=True)
        print("\n训练完成! 🎉")

    except subprocess.CalledProcessError as e:
        print(f"\n训练失败: {e}")
    except KeyboardInterrupt:
        print("\n训练被用户中断")


def test_model():
    """测试模型"""
    print("\n" + "=" * 60)
    print("测试训练好的模型")
    print("=" * 60)

    # 使用配置中的模型目录
    from config import default_config
    model_dir = default_config.training.model_save_dir

    if not os.path.exists(model_dir):
        print(f"未找到模型目录: {model_dir}，请先训练模型")
        return

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not model_files:
        print("未找到模型文件，请先训练模型")
        return

    # 选择最新的best模型或最后一个模型
    best_models = [f for f in model_files if "best" in f]
    if best_models:
        model_file = sorted(best_models)[-1]
    else:
        model_file = sorted(model_files)[-1]

    model_path = os.path.join(model_dir, model_file)
    print(f"使用模型: {model_path}")

    try:
        subprocess.run(
            [sys.executable, "main.py", "--mode", "test", "--model_path", model_path],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"测试失败: {e}")


def interactive_translate():
    """交互式翻译"""
    print("\n" + "=" * 60)
    print("交互式翻译")
    print("=" * 60)

    # 查找模型文件
    model_dir = "./saved_models"
    if not os.path.exists(model_dir):
        print("未找到模型目录，请先训练模型")
        return

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not model_files:
        print("未找到模型文件，请先训练模型")
        return

    # 选择模型
    best_models = [f for f in model_files if "best" in f]
    if best_models:
        model_file = sorted(best_models)[-1]
    else:
        model_file = sorted(model_files)[-1]

    model_path = os.path.join(model_dir, model_file)
    print(f"使用模型: {model_path}")

    try:
        subprocess.run(
            [
                sys.executable,
                "inference.py",
                "--model_path",
                model_path,
                "--mode",
                "interactive",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"推理失败: {e}")
    except KeyboardInterrupt:
        print("\n退出交互式翻译")


def quick_translate(text):
    """快速翻译单个句子"""
    # 使用配置中的模型目录
    from config import default_config
    model_dir = default_config.training.model_save_dir

    if not os.path.exists(model_dir):
        print(f"未找到模型目录: {model_dir}，请先训练模型")
        return

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not model_files:
        print("未找到模型文件，请先训练模型")
        return

    # 选择模型
    best_models = [f for f in model_files if "best" in f]
    if best_models:
        model_file = sorted(best_models)[-1]
    else:
        model_file = sorted(model_files)[-1]

    model_path = os.path.join(model_dir, model_file)

    # 检查是否有example_config.json
    config_args = []
    if os.path.exists("example_config.json"):
        config_args = ["--config", "example_config.json"]

    try:
        cmd = (
            [sys.executable, "inference.py", "--model_path", model_path]
            + config_args
            + ["--mode", "single", "--text", text]
        )
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"翻译失败: {e}")


def show_menu():
    """显示菜单"""
    print("\n" + "=" * 60)
    print("Transformer从零实现 - 英语到意大利语翻译")
    print("=" * 60)
    print("请选择操作:")
    print("1. 训练模型")
    print("2. 测试模型")
    print("3. 交互式翻译")
    print("4. 快速翻译")
    print("5. 检查依赖")
    print("6. 退出")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Transformer一键运行脚本")
    parser.add_argument(
        "--action",
        type=str,
        choices=["train", "test", "interactive", "translate", "check"],
        help="直接执行的操作",
    )
    parser.add_argument("--text", type=str, help="要翻译的文本（用于translate操作）")

    args = parser.parse_args()

    if args.action:
        # 直接执行指定操作
        if args.action == "train":
            train_model()
        elif args.action == "test":
            test_model()
        elif args.action == "interactive":
            interactive_translate()
        elif args.action == "translate":
            if args.text:
                quick_translate(args.text)
            else:
                print("请提供要翻译的文本: --text '你的文本'")
        elif args.action == "check":
            check_dependencies()
    else:
        # 交互式菜单
        while True:
            show_menu()

            try:
                choice = input("请输入选择 (1-6): ").strip()

                if choice == "1":
                    train_model()
                elif choice == "2":
                    test_model()
                elif choice == "3":
                    interactive_translate()
                elif choice == "4":
                    text = input("请输入要翻译的英语句子: ").strip()
                    if text:
                        quick_translate(text)
                    else:
                        print("请输入有效的句子")
                elif choice == "5":
                    check_dependencies()
                elif choice == "6":
                    print("再见! 👋")
                    break
                else:
                    print("无效选择，请输入1-6")

            except KeyboardInterrupt:
                print("\n\n再见! 👋")
                break
            except Exception as e:
                print(f"发生错误: {e}")


if __name__ == "__main__":
    main()


"""
# 1. 训练模型
python main.py --mode train --config example_config.json

# 2. 单句翻译
python inference.py --model_path ./saved_models/best_model_epoch_3.pt --config example_config.json --mode single --text "Hello, how are you?"

# 3. 交互式翻译
python inference.py --model_path ./saved_models/best_model_epoch_3.pt --config example_config.json --mode interactive

# 4. 一键运行
python run.py --action translate --text "Hello world"
"""
