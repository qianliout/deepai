"""
系统检查模块

提供全面的系统环境和配置检查功能，确保RAG系统能够正常运行。

检查项目：
1. 依赖库检查
2. 配置文件检查
3. API连接检查
4. 数据库连接检查
5. 模型可用性检查
6. 目录权限检查
"""

import os
import sys
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from config import config
from logger import get_logger


@dataclass
class CheckResult:
    """检查结果数据类"""

    name: str  # 检查项名称
    status: str  # 状态: success, warning, error
    message: str  # 详细信息
    details: Dict[str, Any]  # 额外详情
    duration: float  # 检查耗时(秒)


@dataclass
class SystemCheckReport:
    """系统检查报告数据类"""

    total_checks: int  # 总检查项数
    success_count: int  # 成功数量
    warning_count: int  # 警告数量
    error_count: int  # 错误数量
    total_duration: float  # 总耗时
    results: List[CheckResult]  # 详细结果
    overall_status: str  # 整体状态


class SystemChecker:
    """系统检查器

    执行全面的系统环境检查，确保RAG系统正常运行
    """

    def __init__(self):
        """初始化系统检查器"""
        self.logger = get_logger("SystemChecker")
        self.results: List[CheckResult] = []

    def run_all_checks(self) -> SystemCheckReport:
        """运行所有检查项

        Returns:
            系统检查报告
        """
        start_time = time.time()
        self.results = []

        self.logger.info("🔍 开始系统检查...")

        # 执行各项检查
        self._check_dependencies()
        self._check_configuration()
        self._check_directories()
        self._check_api_connections()
        self._check_database_connections()
        self._check_model_availability()
        self._check_system_resources()

        # 生成报告
        total_duration = time.time() - start_time
        report = self._generate_report(total_duration)

        self._print_report(report)
        return report

    def _check_dependencies(self):
        """检查依赖库"""
        self.logger.info("📦 检查依赖库...")

        required_packages = {
            "langchain": "LangChain框架",
            "chromadb": "ChromaDB向量数据库",
            "sentence_transformers": "Sentence Transformers嵌入模型",
            "dashscope": "通义百炼API",
            "redis": "Redis客户端",
            "jieba": "中文分词",
            "loguru": "日志库",
            "pydantic": "数据验证",
            "numpy": "数值计算",
            "torch": "PyTorch深度学习框架",
        }

        missing_packages = []
        available_packages = []

        for package, description in required_packages.items():
            try:
                __import__(package)
                available_packages.append(f"{package} ({description})")
            except ImportError:
                missing_packages.append(f"{package} ({description})")

        if missing_packages:
            self._add_result(
                "依赖库检查",
                "error",
                f"缺少 {len(missing_packages)} 个依赖库",
                {"missing": missing_packages, "available": available_packages, "total_required": len(required_packages)},
            )
        else:
            self._add_result("依赖库检查", "success", f"所有 {len(required_packages)} 个依赖库已安装", {"available": available_packages})

    def _check_configuration(self):
        """检查配置文件"""
        self.logger.info("⚙️ 检查配置...")

        try:
            # 检查配置对象是否正常
            config_dict = config.model_dump()

            # 检查关键配置项
            issues = []

            # 检查API密钥
            if not config.llm.api_key:
                issues.append("通义百炼API密钥未设置")

            # 检查目录配置
            if not config.data_dir:
                issues.append("数据目录未配置")

            # 检查嵌入模型配置
            if not config.embedding.model_name:
                issues.append("嵌入模型名称未配置")

            if issues:
                self._add_result(
                    "配置检查", "warning", f"发现 {len(issues)} 个配置问题", {"issues": issues, "config_keys": list(config_dict.keys())}
                )
            else:
                self._add_result("配置检查", "success", "配置文件正常", {"config_keys": list(config_dict.keys())})

        except Exception as e:
            self._add_result("配置检查", "error", f"配置文件错误: {e}", {"error": str(e)})

    def _check_directories(self):
        """检查目录权限"""
        self.logger.info("📁 检查目录权限...")

        directories = [
            config.data_dir,
            config.documents_dir,
            config.logging.log_dir,
            config.chromadb.persist_directory,
            config.embedding.cache_dir,
        ]

        issues = []
        created_dirs = []

        for dir_path in directories:
            try:
                path = Path(dir_path)

                # 尝试创建目录
                path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(path))

                # 检查读写权限
                if not os.access(path, os.R_OK | os.W_OK):
                    issues.append(f"目录 {path} 没有读写权限")

            except Exception as e:
                issues.append(f"目录 {dir_path} 创建失败: {e}")

        if issues:
            self._add_result("目录权限检查", "error", f"发现 {len(issues)} 个目录问题", {"issues": issues, "created": created_dirs})
        else:
            self._add_result("目录权限检查", "success", f"所有 {len(directories)} 个目录正常", {"directories": created_dirs})

    def _check_api_connections(self):
        """检查API连接"""
        self.logger.info("🌐 检查API连接...")

        # 检查通义百炼API
        if config.llm.api_key:
            try:
                import dashscope

                dashscope.api_key = config.llm.api_key

                # 尝试调用API
                response = dashscope.Generation.call(
                    model=config.llm.model_name, messages=[{"role": "user", "content": "测试连接"}], max_tokens=10
                )

                if response.status_code == 200:
                    self._add_result(
                        "通义百炼API检查", "success", "API连接正常", {"model": config.llm.model_name, "status_code": response.status_code}
                    )
                else:
                    self._add_result(
                        "通义百炼API检查",
                        "error",
                        f"API调用失败: {response.message}",
                        {"status_code": response.status_code, "error": response.message},
                    )

            except Exception as e:
                self._add_result("通义百炼API检查", "error", f"API连接失败: {e}", {"error": str(e)})
        else:
            self._add_result("通义百炼API检查", "warning", "API密钥未设置，跳过检查", {"api_key_set": False})

    def _check_database_connections(self):
        """检查数据库连接"""
        self.logger.info("🗄️ 检查数据库连接...")

        # 检查Redis连接
        try:
            import redis

            r = redis.Redis(
                host=config.redis.host, port=config.redis.port, password=config.redis.password or None, db=config.redis.db, socket_timeout=5
            )

            # 测试连接
            r.ping()

            self._add_result(
                "Redis连接检查", "success", "Redis连接正常", {"host": config.redis.host, "port": config.redis.port, "db": config.redis.db}
            )

        except Exception as e:
            self._add_result(
                "Redis连接检查", "warning", f"Redis连接失败: {e}", {"error": str(e), "host": config.redis.host, "port": config.redis.port}
            )

        # 检查ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(path=config.chromadb.persist_directory, settings=Settings(anonymized_telemetry=False))

            # 测试创建集合
            test_collection = client.get_or_create_collection("test_connection")
            client.delete_collection("test_connection")

            self._add_result("ChromaDB检查", "success", "ChromaDB连接正常", {"persist_directory": config.chromadb.persist_directory})

        except Exception as e:
            self._add_result("ChromaDB检查", "error", f"ChromaDB连接失败: {e}", {"error": str(e)})

    def _check_model_availability(self):
        """检查模型可用性"""
        self.logger.info("🤖 检查模型可用性...")

        # 检查嵌入模型
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(config.embedding.model_name, cache_folder=config.embedding.cache_dir)

            # 测试嵌入
            test_embedding = model.encode(["测试文本"])

            self._add_result(
                "嵌入模型检查",
                "success",
                f"模型 {config.embedding.model_name} 可用",
                {"model_name": config.embedding.model_name, "embedding_dim": len(test_embedding[0]), "device": config.get_device()},
            )

        except Exception as e:
            self._add_result(
                "嵌入模型检查", "error", f"嵌入模型加载失败: {e}", {"error": str(e), "model_name": config.embedding.model_name}
            )

    def _check_system_resources(self):
        """检查系统资源"""
        self.logger.info("💻 检查系统资源...")

        try:
            import psutil

            # 内存检查
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)

            # 磁盘检查
            disk = psutil.disk_usage(".")
            disk_free_gb = disk.free / (1024**3)

            # CPU检查
            cpu_count = psutil.cpu_count()

            issues = []
            if memory_gb < 4:
                issues.append(f"内存不足，建议至少4GB，当前: {memory_gb:.1f}GB")

            if disk_free_gb < 2:
                issues.append(f"磁盘空间不足，建议至少2GB，当前: {disk_free_gb:.1f}GB")

            status = "warning" if issues else "success"
            message = f"发现 {len(issues)} 个资源问题" if issues else "系统资源充足"

            self._add_result(
                "系统资源检查",
                status,
                message,
                {"memory_gb": round(memory_gb, 1), "disk_free_gb": round(disk_free_gb, 1), "cpu_count": cpu_count, "issues": issues},
            )

        except ImportError:
            self._add_result("系统资源检查", "warning", "psutil未安装，跳过资源检查", {"psutil_available": False})
        except Exception as e:
            self._add_result("系统资源检查", "error", f"资源检查失败: {e}", {"error": str(e)})

    def _add_result(self, name: str, status: str, message: str, details: Dict[str, Any]):
        """添加检查结果"""
        result = CheckResult(name=name, status=status, message=message, details=details, duration=0.0)  # 这里简化处理
        self.results.append(result)

    def _generate_report(self, total_duration: float) -> SystemCheckReport:
        """生成检查报告"""
        success_count = sum(1 for r in self.results if r.status == "success")
        warning_count = sum(1 for r in self.results if r.status == "warning")
        error_count = sum(1 for r in self.results if r.status == "error")

        # 确定整体状态
        if error_count > 0:
            overall_status = "error"
        elif warning_count > 0:
            overall_status = "warning"
        else:
            overall_status = "success"

        return SystemCheckReport(
            total_checks=len(self.results),
            success_count=success_count,
            warning_count=warning_count,
            error_count=error_count,
            total_duration=total_duration,
            results=self.results,
            overall_status=overall_status,
        )

    def _print_report(self, report: SystemCheckReport):
        """打印检查报告"""
        print("\n" + "=" * 60)
        print("🔍 RAG系统检查报告")
        print("=" * 60)

        # 总体状态
        status_emoji = {"success": "✅", "warning": "⚠️", "error": "❌"}
        print(f"\n整体状态: {status_emoji[report.overall_status]} {report.overall_status.upper()}")
        print(f"检查项目: {report.total_checks}")
        print(f"成功: {report.success_count} | 警告: {report.warning_count} | 错误: {report.error_count}")
        print(f"总耗时: {report.total_duration:.2f}秒")

        # 详细结果
        print("\n详细结果:")
        print("-" * 60)

        for result in report.results:
            emoji = status_emoji[result.status]
            print(f"{emoji} {result.name}: {result.message}")

            # 显示重要详情
            if result.status == "error" and "error" in result.details:
                print(f"   错误详情: {result.details['error']}")
            elif result.status == "warning" and "issues" in result.details:
                for issue in result.details["issues"]:
                    print(f"   - {issue}")

        print("\n" + "=" * 60)
