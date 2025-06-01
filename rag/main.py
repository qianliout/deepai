"""
RAG个人知识库主程序
参考BERT项目的简化方式，使用argparse子命令，所有参数在config.py中配置
"""

import argparse
import logging
import sys
from pathlib import Path

from config import (
    setup_logging, print_config, create_directories, load_env_config,
    defaultConfig
)

logger = logging.getLogger("RAG")


def run_build():
    """构建知识库"""
    print("\n📚 开始构建知识库")
    print("=" * 50)
    
    print_config()
    
    try:
        from document_loader import DocumentLoader
        from text_splitter import TextSplitterManager
        from embeddings import EmbeddingManager
        from vector_store import VectorStoreManager
        
        # 1. 加载文档
        print("\n1. 加载文档...")
        loader = DocumentLoader()
        documents = loader.load_directory("data/documents")
        print(f"加载了 {len(documents)} 个文档")
        
        if not documents:
            print("❌ 没有找到文档，请将文档放入 data/documents 目录")
            return
        
        # 2. 分割文档
        print("\n2. 分割文档...")
        splitter = TextSplitterManager()
        split_docs = splitter.split_documents(documents)
        print(f"分割成 {len(split_docs)} 个文档块")
        
        # 3. 创建嵌入
        print("\n3. 创建嵌入...")
        embedding_manager = EmbeddingManager()
        
        # 4. 构建向量存储
        print("\n4. 构建向量存储...")
        vector_store = VectorStoreManager(embedding_manager)
        vector_store.add_documents(split_docs)
        
        print("\n✅ 知识库构建完成！")
        print(f"文档数量: {len(documents)}")
        print(f"文档块数量: {len(split_docs)}")
        print(f"向量存储目录: {defaultConfig.vector_store.persist_directory}")
        
    except Exception as e:
        logger.error(f"构建知识库失败: {e}")
        print(f"❌ 构建失败: {e}")


def run_chat():
    """启动交互式对话"""
    print("\n💬 启动交互式对话")
    print("=" * 50)
    
    try:
        from rag_chain import RAGChain
        
        # 初始化RAG链
        print("正在初始化RAG系统...")
        rag_chain = RAGChain()
        
        print("\n🤖 RAG助手已启动！")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清空对话历史")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 您: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                elif user_input.lower() in ['clear', '清空']:
                    rag_chain.clear_history()
                    print("✅ 对话历史已清空")
                    continue
                elif not user_input:
                    continue
                
                print("\n🤖 助手: ", end="", flush=True)
                
                # 流式输出
                for chunk in rag_chain.stream_chat(user_input):
                    print(chunk, end="", flush=True)
                print()  # 换行
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
        
        print("\n👋 再见！")
        
    except Exception as e:
        logger.error(f"对话失败: {e}")
        print(f"❌ 对话失败: {e}")


def run_query(query: str):
    """单次查询"""
    print(f"\n🔍 查询: {query}")
    print("=" * 50)
    
    try:
        from rag_chain import RAGChain
        
        rag_chain = RAGChain()
        response = rag_chain.query(query)
        
        print(f"\n📝 回答:")
        print(response)
        
    except Exception as e:
        logger.error(f"查询失败: {e}")
        print(f"❌ 查询失败: {e}")


def run_clear():
    """清空知识库"""
    print("\n🗑️ 清空知识库")
    print("=" * 50)
    
    try:
        from vector_store import VectorStoreManager
        from embeddings import EmbeddingManager
        
        embedding_manager = EmbeddingManager()
        vector_store = VectorStoreManager(embedding_manager)
        vector_store.clear()
        
        print("✅ 知识库已清空")
        
    except Exception as e:
        logger.error(f"清空失败: {e}")
        print(f"❌ 清空失败: {e}")


def run_status():
    """查看系统状态"""
    print("\n📊 系统状态")
    print("=" * 50)
    
    try:
        from vector_store import VectorStoreManager
        from embeddings import EmbeddingManager
        
        embedding_manager = EmbeddingManager()
        vector_store = VectorStoreManager(embedding_manager)
        
        # 获取文档数量
        stats = vector_store.get_stats()
        doc_count = stats.get("document_count", 0)
        
        print(f"\n📚 知识库状态:")
        print(f"  文档数量: {doc_count}")
        print(f"  存储目录: {defaultConfig.vector_store.persist_directory}")
        print(f"  集合名称: {defaultConfig.vector_store.collection_name}")

        print(f"\n🤖 LLM状态:")
        print(f"  模型: {defaultConfig.llm.model_name}")
        print(f"  API密钥: {'已设置' if defaultConfig.llm.api_key else '未设置'}")

        print(f"\n📊 嵌入模型状态:")
        print(f"  模型: {defaultConfig.embedding.model_name}")
        print(f"  设备: {defaultConfig.embedding.device}")
        
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        print(f"❌ 获取状态失败: {e}")


def run_quick():
    """快速测试"""
    print("\n⚡ 快速测试")
    print("=" * 50)
    
    # 创建测试文档
    test_doc_path = Path("data/documents/test.txt")
    test_doc_path.parent.mkdir(parents=True, exist_ok=True)
    
    test_content = """
人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。

机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习和改进。

深度学习是机器学习的一个分支，使用神经网络来模拟人脑的工作方式。

自然语言处理（NLP）是人工智能的一个领域，专注于计算机与人类语言之间的交互。
"""
    
    with open(test_doc_path, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    print("✅ 创建测试文档")
    
    # 构建知识库
    print("\n1. 构建知识库...")
    run_build()
    
    # 测试查询
    print("\n2. 测试查询...")
    test_queries = [
        "什么是人工智能？",
        "机器学习和深度学习的关系是什么？"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        run_query(query)
    
    print("\n⚡ 快速测试完成！")


def main():
    """主函数"""
    # 设置日志
    setup_logging()
    
    # 加载环境配置
    load_env_config()
    
    # 创建目录
    create_directories()
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="RAG个人知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py build                    # 构建知识库
  python main.py chat                     # 交互式对话
  python main.py query "什么是AI？"        # 单次查询
  python main.py status                   # 查看状态
  python main.py clear                    # 清空知识库
  python main.py quick                    # 快速测试

注意：所有参数都在config.py中配置，无需手动传参
        """,
    )
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 构建命令
    build_parser = subparsers.add_parser("build", help="构建知识库")
    
    # 对话命令
    chat_parser = subparsers.add_parser("chat", help="交互式对话")
    
    # 查询命令
    query_parser = subparsers.add_parser("query", help="单次查询")
    query_parser.add_argument("text", help="查询文本")
    
    # 状态命令
    status_parser = subparsers.add_parser("status", help="查看系统状态")
    
    # 清空命令
    clear_parser = subparsers.add_parser("clear", help="清空知识库")
    
    # 快速测试命令
    quick_parser = subparsers.add_parser("quick", help="快速测试")
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有提供命令，显示帮助
    if args.command is None:
        parser.print_help()
        return
    
    try:
        # 执行对应的命令
        if args.command == "build":
            run_build()
        elif args.command == "chat":
            run_chat()
        elif args.command == "query":
            run_query(args.text)
        elif args.command == "status":
            run_status()
        elif args.command == "clear":
            run_clear()
        elif args.command == "quick":
            run_quick()
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
