#!/usr/bin/env python3
"""
测试API功能
"""

import sys
import os
from pathlib import Path
import time
import subprocess
import requests

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def test_api_import():
    """测试API模块导入"""
    print("🔍 测试API模块导入...")
    
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI和Uvicorn导入成功")
        
        from run_simple_api import create_simple_api
        app = create_simple_api()
        print("✅ 简化API应用创建成功")
        
        return True
    except Exception as e:
        print(f"❌ API模块导入失败: {e}")
        return False

def test_config_in_api():
    """测试API中的配置加载"""
    print("\n⚙️  测试API配置加载...")
    
    try:
        from config.config import get_config
        config = get_config()
        print(f"✅ 配置加载成功: 环境={config.environment}")
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def start_api_server():
    """启动API服务器"""
    print("\n🚀 启动API服务器...")
    
    try:
        # 启动服务器进程
        process = subprocess.Popen([
            sys.executable, "run_simple_api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待服务器启动
        print("⏳ 等待服务器启动...")
        time.sleep(3)
        
        # 检查进程是否还在运行
        if process.poll() is None:
            print("✅ API服务器启动成功")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ API服务器启动失败")
            print(f"stdout: {stdout.decode()}")
            print(f"stderr: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ 启动API服务器异常: {e}")
        return None

def test_api_endpoints(max_retries=3):
    """测试API端点"""
    print("\n🌐 测试API端点...")
    
    base_url = "http://localhost:8000"
    
    for attempt in range(max_retries):
        try:
            # 测试根路径
            response = requests.get(f"{base_url}/", timeout=5)
            if response.status_code == 200:
                print("✅ 根路径 (/) 响应正常")
                data = response.json()
                print(f"  响应: {data}")
                
                # 测试健康检查
                response = requests.get(f"{base_url}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ 健康检查 (/health) 响应正常")
                    health_data = response.json()
                    print(f"  状态: {health_data.get('status')}")
                else:
                    print(f"⚠️  健康检查响应异常: {response.status_code}")
                
                # 测试信息接口
                response = requests.get(f"{base_url}/info", timeout=5)
                if response.status_code == 200:
                    print("✅ 信息接口 (/info) 响应正常")
                else:
                    print(f"⚠️  信息接口响应异常: {response.status_code}")
                
                return True
            else:
                print(f"❌ 根路径响应异常: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"⏳ 连接失败，重试 {attempt + 1}/{max_retries}...")
            time.sleep(2)
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            break
    
    print("❌ API端点测试失败")
    return False

def main():
    """主函数"""
    print("🧪 RAG2 API功能测试")
    print("=" * 50)
    
    # 测试导入
    if not test_api_import():
        print("❌ API模块导入失败，无法继续测试")
        return 1
    
    # 测试配置
    if not test_config_in_api():
        print("❌ API配置测试失败")
        return 1
    
    # 启动API服务器
    server_process = start_api_server()
    if server_process is None:
        print("❌ 无法启动API服务器")
        return 1
    
    try:
        # 测试API端点
        if test_api_endpoints():
            print("\n🎉 所有API测试通过！")
            result = 0
        else:
            print("\n⚠️  API端点测试失败")
            result = 1
    
    finally:
        # 停止服务器
        print("\n🔄 停止API服务器...")
        server_process.terminate()
        server_process.wait()
        print("✅ API服务器已停止")
    
    return result

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        sys.exit(1)
