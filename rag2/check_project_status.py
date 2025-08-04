#!/usr/bin/env python3
"""
RAG2项目状态检查脚本
检查项目完整性和各组件状态
"""

import os
import sys
from pathlib import Path
import subprocess

def check_file_exists(file_path: str, description: str) -> bool:
    """检查文件是否存在"""
    path = Path(file_path)
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {file_path}")
    return exists

def check_directory_exists(dir_path: str, description: str) -> bool:
    """检查目录是否存在"""
    path = Path(dir_path)
    exists = path.exists() and path.is_dir()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dir_path}")
    return exists

def check_docker_service(service_name: str) -> bool:
    """检查Docker服务状态"""
    try:
        result = subprocess.run([
            "docker", "ps", "--filter", f"name={service_name}", "--format", "{{.Status}}"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            status = result.stdout.strip()
            if "Up" in status:
                print(f"✅ Docker服务 {service_name}: {status}")
                return True
            else:
                print(f"⚠️  Docker服务 {service_name}: {status}")
                return False
        else:
            print(f"❌ Docker服务 {service_name}: 未运行")
            return False
    except Exception as e:
        print(f"❌ Docker服务 {service_name}: 检查失败 - {str(e)}")
        return False

def main():
    """主检查函数"""
    print("🔍 RAG2项目状态检查")
    print("=" * 60)
    
    # 检查核心配置文件
    print("\n📁 核心配置文件:")
    config_files = [
        ("config/config.py", "主配置文件"),
        ("config/environment_config.py", "环境配置文件"),
        ("requirements.txt", "Python依赖文件"),
        ("docker-compose.yml", "Docker编排文件"),
        (".env.example", "环境变量示例"),
        ("README.md", "项目文档")
    ]
    
    config_score = 0
    for file_path, desc in config_files:
        if check_file_exists(file_path, desc):
            config_score += 1
    
    # 检查核心模块
    print("\n🧩 核心模块:")
    core_modules = [
        ("core/document_processor.py", "文档处理器"),
        ("core/rag_pipeline.py", "RAG管道"),
        ("models/llm_client.py", "LLM客户端"),
        ("models/embeddings.py", "嵌入模型管理器"),
        ("models/rerank_models.py", "重排序模型管理器"),
        ("retrieval/base_retriever.py", "基础检索器"),
        ("retrieval/semantic_retriever.py", "语义检索器")
    ]
    
    core_score = 0
    for file_path, desc in core_modules:
        if check_file_exists(file_path, desc):
            core_score += 1
    
    # 检查存储模块
    print("\n💾 存储模块:")
    storage_modules = [
        ("storage/postgresql_manager.py", "PostgreSQL管理器"),
        ("storage/mysql_manager.py", "MySQL管理器"),
        ("storage/redis_manager.py", "Redis管理器")
    ]
    
    storage_score = 0
    for file_path, desc in storage_modules:
        if check_file_exists(file_path, desc):
            storage_score += 1
    
    # 检查API模块
    print("\n🌐 API模块:")
    api_modules = [
        ("api/main.py", "FastAPI主应用"),
        ("api/routes/query.py", "查询API路由"),
        ("api/routes/document.py", "文档管理API路由"),
        ("api/routes/admin.py", "管理API路由"),
        ("api/schemas/__init__.py", "API数据模型")
    ]
    
    api_score = 0
    for file_path, desc in api_modules:
        if check_file_exists(file_path, desc):
            api_score += 1
    
    # 检查测试文件
    print("\n🧪 测试文件:")
    test_files = [
        ("test_basic_setup.py", "基础设置测试"),
        ("test_complete_system.py", "完整系统测试"),
        ("quick_start.py", "快速启动脚本")
    ]
    
    test_score = 0
    for file_path, desc in test_files:
        if check_file_exists(file_path, desc):
            test_score += 1
    
    # 检查数据库初始化脚本
    print("\n🗄️  数据库初始化:")
    db_scripts = [
        ("deployment/sql/init_postgresql.sql", "PostgreSQL初始化脚本"),
        ("deployment/sql/init_mysql.sql", "MySQL初始化脚本"),
        ("deployment/neo4j/init_schema.cypher", "Neo4j初始化脚本"),
        ("deployment/docker/redis.conf", "Redis配置文件")
    ]
    
    db_score = 0
    for file_path, desc in db_scripts:
        if check_file_exists(file_path, desc):
            db_score += 1
    
    # 检查工具文件
    print("\n🛠️  工具文件:")
    utils_files = [
        ("utils/logger.py", "日志工具")
    ]
    
    utils_score = 0
    for file_path, desc in utils_files:
        if check_file_exists(file_path, desc):
            utils_score += 1
    
    # 检查目录结构
    print("\n📂 目录结构:")
    directories = [
        ("data/logs", "日志目录"),
        ("data/documents", "文档目录"),
        ("data/mock", "模拟数据目录")
    ]
    
    dir_score = 0
    for dir_path, desc in directories:
        if check_directory_exists(dir_path, desc):
            dir_score += 1
    
    # 检查Docker服务
    print("\n🐳 Docker服务状态:")
    docker_services = [
        "rag2_postgres",
        "rag2_mysql", 
        "rag2_redis",
        "rag2_ollama"
    ]
    
    docker_score = 0
    for service in docker_services:
        if check_docker_service(service):
            docker_score += 1
    
    # 计算总分
    total_files = len(config_files) + len(core_modules) + len(storage_modules) + len(api_modules) + len(test_files) + len(db_scripts) + len(utils_files)
    total_score = config_score + core_score + storage_score + api_score + test_score + db_score + utils_score
    
    # 显示汇总结果
    print("\n" + "=" * 60)
    print("📊 项目完整性汇总")
    print("=" * 60)
    
    print(f"📁 配置文件: {config_score}/{len(config_files)}")
    print(f"🧩 核心模块: {core_score}/{len(core_modules)}")
    print(f"💾 存储模块: {storage_score}/{len(storage_modules)}")
    print(f"🌐 API模块: {api_score}/{len(api_modules)}")
    print(f"🧪 测试文件: {test_score}/{len(test_files)}")
    print(f"🗄️  数据库脚本: {db_score}/{len(db_scripts)}")
    print(f"🛠️  工具文件: {utils_score}/{len(utils_files)}")
    print(f"📂 目录结构: {dir_score}/{len(directories)}")
    print(f"🐳 Docker服务: {docker_score}/{len(docker_services)}")
    
    print("-" * 60)
    print(f"📈 总体完成度: {total_score}/{total_files} ({total_score/total_files*100:.1f}%)")
    
    # 给出建议
    print("\n💡 建议:")
    
    if total_score == total_files:
        print("🎉 项目文件完整！可以开始测试和使用。")
    elif total_score >= total_files * 0.8:
        print("✅ 项目基本完整，可以进行基础测试。")
        print("   建议补充缺失的文件以获得完整功能。")
    else:
        print("⚠️  项目还需要完善，建议先补充核心文件。")
    
    if docker_score < len(docker_services):
        print("🐳 部分Docker服务未运行，请执行: docker-compose up -d")
    
    print("\n🚀 下一步操作:")
    print("1. 如果Docker服务未启动: docker-compose up -d")
    print("2. 安装Python依赖: pip install -r requirements.txt")
    print("3. 运行快速启动: python quick_start.py")
    print("4. 运行基础测试: python test_basic_setup.py")
    print("5. 运行完整测试: python test_complete_system.py")
    print("6. 启动API服务: python -m api.main")
    
    return 0 if total_score >= total_files * 0.8 else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n检查被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n检查过程中发生错误: {str(e)}")
        sys.exit(1)
