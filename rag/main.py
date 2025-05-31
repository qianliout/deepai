#!/usr/bin/env python3
"""
RAG个人知识库主入口文件

该文件提供命令行接口来管理和使用RAG知识库系统。
支持文档导入、知识库构建、问答查询等功能。

使用方法：
    python main.py quick                    # 快速测试
    python main.py build --docs ./docs     # 构建知识库
    python main.py query "你的问题"         # 查询知识库
    python main.py chat                     # 交互式对话
    python main.py serve                    # 启动API服务
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, List
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config import config
from logger import setup_logger, get_logger
from embeddings import EmbeddingManager
from vector_store import VectorStoreManager
from llm import LLMManager
from document_loader import DocumentLoader

# 初始化控制台和日志
console = Console()
setup_logger()
logger = get_logger("RAGMain")


class RAGSystem:
    """RAG系统主类

    整合所有组件，提供统一的系统接口
    """

    def __init__(self):
        """初始化RAG系统"""
        self.embedding_manager = None
        self.vector_store = None
        self.llm_manager = None
        self.document_loader = None
        self._initialized = False

    def initialize(self):
        """初始化所有组件"""
        if self._initialized:
            return

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:

            # 初始化嵌入管理器
            task1 = progress.add_task("正在初始化嵌入模型...", total=None)
            self.embedding_manager = EmbeddingManager()
            progress.update(task1, completed=True)

            # 初始化向量存储
            task2 = progress.add_task("正在初始化向量存储...", total=None)
            self.vector_store = VectorStoreManager(self.embedding_manager)
            progress.update(task2, completed=True)

            # 初始化LLM
            task3 = progress.add_task("正在初始化大语言模型...", total=None)
            self.llm_manager = LLMManager()
            progress.update(task3, completed=True)

            # 初始化文档加载器
            task4 = progress.add_task("正在初始化文档加载器...", total=None)
            self.document_loader = DocumentLoader()
            progress.update(task4, completed=True)

        self._initialized = True
        console.print("✅ RAG系统初始化完成", style="green")

    def build_knowledge_base(self, docs_path: str, clear_existing: bool = False):
        """构建知识库

        Args:
            docs_path: 文档目录路径
            clear_existing: 是否清空现有知识库
        """
        if not self._initialized:
            self.initialize()

        docs_path = Path(docs_path)
        if not docs_path.exists():
            console.print(f"❌ 文档目录不存在: {docs_path}", style="red")
            return

        console.print(f"📚 开始构建知识库，文档目录: {docs_path}")

        if clear_existing:
            console.print("🗑️ 清空现有知识库...")
            self.vector_store.clear()

        # 加载文档
        console.print("📖 正在加载文档...")
        documents = self.document_loader.load_directory(docs_path, recursive=True)

        if not documents:
            console.print("⚠️ 未找到可处理的文档", style="yellow")
            return

        console.print(f"📄 成功加载 {len(documents)} 个文档片段")

        # 添加到向量存储
        console.print("🔍 正在构建向量索引...")
        with Progress(console=console) as progress:
            task = progress.add_task("处理文档...", total=len(documents))

            # 批量处理文档
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                self.vector_store.add_documents(batch)
                progress.update(task, advance=len(batch))

        # 显示统计信息
        stats = self.vector_store.get_stats()
        self._display_stats(stats)

        console.print("✅ 知识库构建完成", style="green")

    def query_knowledge_base(self, query: str, top_k: int = 5) -> str:
        """查询知识库

        Args:
            query: 查询问题
            top_k: 返回结果数量

        Returns:
            回答结果
        """
        if not self._initialized:
            self.initialize()

        console.print(f"🔍 正在查询: {query}")

        # 检索相关文档
        with console.status("正在检索相关文档..."):
            results = self.vector_store.similarity_search(query, k=top_k)

        if not results:
            return "抱歉，没有找到相关信息。"

        # 构建上下文
        context_parts = []
        for i, (doc, score) in enumerate(results):
            context_parts.append(f"文档{i+1} (相似度: {score:.3f}):\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # 生成回答
        with console.status("正在生成回答..."):
            answer = self.llm_manager.generate_with_context(query, context)

        return answer

    def interactive_chat(self):
        """交互式对话模式"""
        if not self._initialized:
            self.initialize()

        console.print(
            Panel.fit(
                "🤖 RAG知识库交互式对话\n\n" "输入 'quit' 或 'exit' 退出\n" "输入 'clear' 清空对话历史\n" "输入 'stats' 查看系统状态",
                title="欢迎使用RAG知识库",
                border_style="blue",
            )
        )

        while True:
            try:
                user_input = console.input("\n[bold blue]您:[/bold blue] ")

                if user_input.lower() in ["quit", "exit", "退出"]:
                    console.print("👋 再见！", style="green")
                    break

                if user_input.lower() in ["clear", "清空"]:
                    self.llm_manager.clear_history()
                    console.print("🗑️ 对话历史已清空", style="yellow")
                    continue

                if user_input.lower() in ["stats", "状态"]:
                    stats = self.vector_store.get_stats()
                    self._display_stats(stats)
                    continue

                if not user_input.strip():
                    continue

                # 查询并回答
                answer = self.query_knowledge_base(user_input)
                console.print(f"\n[bold green]助手:[/bold green] {answer}")

            except KeyboardInterrupt:
                console.print("\n👋 再见！", style="green")
                break
            except Exception as e:
                console.print(f"❌ 发生错误: {e}", style="red")
                logger.error(f"交互式对话错误: {e}")

    def _display_stats(self, stats: dict):
        """显示系统统计信息"""
        table = Table(title="📊 系统状态")
        table.add_column("项目", style="cyan")
        table.add_column("值", style="green")

        table.add_row("向量存储类型", stats.get("store_type", "未知"))
        table.add_row("文档数量", str(stats.get("document_count", 0)))
        table.add_row("嵌入维度", str(stats.get("embedding_dim", 0)))
        table.add_row("集合名称", stats.get("collection_name", "未知"))

        console.print(table)


# 创建全局RAG系统实例
rag_system = RAGSystem()


@click.group()
@click.option("--debug", is_flag=True, help="启用调试模式")
def cli(debug):
    """RAG个人知识库命令行工具"""
    if debug:
        config.debug = True
        config.logging.level = "DEBUG"


@cli.command()
@click.option("--docs", default="rag/data/samples", help="示例文档目录")
def quick(docs):
    """快速测试RAG系统"""
    console.print(Panel.fit("🚀 RAG系统快速测试\n\n" "这将使用示例文档快速测试系统功能", title="快速测试", border_style="green"))

    # 创建示例文档
    docs_path = Path(docs)
    docs_path.mkdir(parents=True, exist_ok=True)

    sample_doc = docs_path / "sample.txt"
    if not sample_doc.exists():
        sample_content = """
# RAG系统介绍

RAG（Retrieval-Augmented Generation）是一种结合了检索和生成的AI技术。

## 主要特点

1. **检索增强**: 从知识库中检索相关信息
2. **生成回答**: 基于检索到的信息生成准确回答
3. **知识更新**: 可以动态更新知识库内容

## 应用场景

- 智能问答系统
- 文档助手
- 知识管理
- 客服机器人

RAG技术能够有效解决大语言模型的知识局限性问题。
        """
        sample_doc.write_text(sample_content, encoding="utf-8")

    # 构建知识库
    rag_system.build_knowledge_base(docs, clear_existing=True)

    # 测试查询
    test_queries = ["什么是RAG？", "RAG有哪些应用场景？", "RAG的主要特点是什么？"]

    console.print("\n🧪 开始测试查询...")
    for query in test_queries:
        console.print(f"\n[bold blue]测试问题:[/bold blue] {query}")
        answer = rag_system.query_knowledge_base(query)
        console.print(f"[bold green]回答:[/bold green] {answer}")

    console.print("\n✅ 快速测试完成！", style="green")


@cli.command()
@click.option("--docs", required=True, help="文档目录路径")
@click.option("--clear", is_flag=True, help="清空现有知识库")
def build(docs, clear):
    """构建知识库"""
    rag_system.build_knowledge_base(docs, clear_existing=clear)


@cli.command()
@click.argument("question")
@click.option("--top-k", default=5, help="检索文档数量")
def query(question, top_k):
    """查询知识库"""
    answer = rag_system.query_knowledge_base(question, top_k=top_k)
    console.print(f"\n[bold green]回答:[/bold green] {answer}")


@cli.command()
def chat():
    """启动交互式对话"""
    rag_system.interactive_chat()


@cli.command()
def status():
    """查看系统状态"""
    if not rag_system._initialized:
        rag_system.initialize()

    stats = rag_system.vector_store.get_stats()
    rag_system._display_stats(stats)


@cli.command()
def clear():
    """清空知识库"""
    if not rag_system._initialized:
        rag_system.initialize()

    if click.confirm("确定要清空知识库吗？"):
        rag_system.vector_store.clear()
        console.print("✅ 知识库已清空", style="green")


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n👋 程序已退出", style="yellow")
    except Exception as e:
        console.print(f"❌ 程序异常: {e}", style="red")
        logger.error(f"主程序异常: {e}")
        sys.exit(1)
