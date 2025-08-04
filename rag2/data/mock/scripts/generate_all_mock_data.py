#!/usr/bin/env python3
"""
RAG2 AIOps项目 - Mock数据生成总脚本
生成所有测试数据并导入到各个数据库中
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os

# 确保脚本可以找到项目模块
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def generate_hosts_data(count: int = 100) -> List[Dict[str, Any]]:
    """生成主机数据"""
    hosts = []
    for i in range(count):
        host = {
            "id": str(uuid.uuid4()),
            "hostname": f"host-{i+1:03d}",
            "ip_address": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "os": random.choice(["Ubuntu 20.04", "CentOS 7", "RHEL 8", "Debian 11"]),
            "cpu_cores": random.choice([2, 4, 8, 16]),
            "memory_gb": random.choice([4, 8, 16, 32, 64]),
            "environment": random.choice(["production", "staging", "development"]),
            "datacenter": random.choice(["dc1", "dc2", "dc3"]),
            "created_at": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            "status": random.choice(["active", "maintenance", "offline"])
        }
        hosts.append(host)
    return hosts

def generate_images_data(count: int = 200) -> List[Dict[str, Any]]:
    """生成镜像数据"""
    base_images = [
        "nginx", "redis", "mysql", "postgres", "mongodb", 
        "elasticsearch", "kibana", "grafana", "prometheus",
        "node", "python", "java", "golang", "ubuntu", "alpine"
    ]
    
    images = []
    for i in range(count):
        base_name = random.choice(base_images)
        version = f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        
        image = {
            "id": str(uuid.uuid4()),
            "name": f"{base_name}:{version}",
            "base_image": base_name,
            "version": version,
            "size_mb": random.randint(50, 2000),
            "architecture": random.choice(["amd64", "arm64"]),
            "created_at": (datetime.now() - timedelta(days=random.randint(1, 180))).isoformat(),
            "registry": random.choice(["docker.io", "gcr.io", "quay.io"]),
            "tags": [f"v{version}", "latest"] if random.random() > 0.5 else [f"v{version}"],
            "scan_status": random.choice(["scanned", "scanning", "failed", "pending"])
        }
        images.append(image)
    return images

def generate_vulnerabilities_data(count: int = 500) -> List[Dict[str, Any]]:
    """生成漏洞数据"""
    cve_prefixes = ["CVE-2023-", "CVE-2024-"]
    severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    categories = [
        "Buffer Overflow", "SQL Injection", "Cross-Site Scripting", 
        "Remote Code Execution", "Privilege Escalation", "Information Disclosure",
        "Denial of Service", "Authentication Bypass"
    ]
    
    vulnerabilities = []
    for i in range(count):
        cve_year = random.choice(["2023", "2024"])
        cve_number = f"{random.randint(1000, 9999)}"
        
        vuln = {
            "id": str(uuid.uuid4()),
            "cve_id": f"CVE-{cve_year}-{cve_number}",
            "title": f"Vulnerability in {random.choice(['OpenSSL', 'Apache', 'Nginx', 'MySQL', 'PostgreSQL'])}",
            "description": f"A {random.choice(categories).lower()} vulnerability was discovered...",
            "severity": random.choice(severities),
            "cvss_score": round(random.uniform(1.0, 10.0), 1),
            "category": random.choice(categories),
            "published_date": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            "modified_date": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
            "affected_packages": [
                f"package-{random.randint(1, 100)}" for _ in range(random.randint(1, 3))
            ],
            "fix_available": random.choice([True, False]),
            "fix_version": f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}" if random.random() > 0.3 else None,
            "references": [
                f"https://nvd.nist.gov/vuln/detail/CVE-{cve_year}-{cve_number}",
                f"https://security.example.com/advisory/{cve_number}"
            ]
        }
        vulnerabilities.append(vuln)
    return vulnerabilities

def generate_relationships_data(hosts: List[Dict], images: List[Dict], vulns: List[Dict]) -> List[Dict[str, Any]]:
    """生成实体关系数据"""
    relationships = []
    
    # 主机-镜像关系
    for host in hosts:
        # 每台主机运行1-5个镜像
        host_images = random.sample(images, random.randint(1, 5))
        for image in host_images:
            rel = {
                "id": str(uuid.uuid4()),
                "type": "HOST_RUNS_IMAGE",
                "source_id": host["id"],
                "source_type": "host",
                "target_id": image["id"],
                "target_type": "image",
                "properties": {
                    "container_name": f"container-{random.randint(1000, 9999)}",
                    "status": random.choice(["running", "stopped", "paused"]),
                    "created_at": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
                }
            }
            relationships.append(rel)
    
    # 镜像-漏洞关系
    for image in images:
        # 每个镜像有0-10个漏洞
        image_vulns = random.sample(vulns, random.randint(0, 10))
        for vuln in image_vulns:
            rel = {
                "id": str(uuid.uuid4()),
                "type": "IMAGE_HAS_VULNERABILITY",
                "source_id": image["id"],
                "source_type": "image",
                "target_id": vuln["id"],
                "target_type": "vulnerability",
                "properties": {
                    "detected_at": (datetime.now() - timedelta(days=random.randint(1, 60))).isoformat(),
                    "scanner": random.choice(["trivy", "clair", "snyk"]),
                    "confidence": random.choice(["high", "medium", "low"])
                }
            }
            relationships.append(rel)
    
    return relationships

def generate_sample_queries() -> List[Dict[str, Any]]:
    """生成测试查询样例"""
    queries = [
        {
            "id": str(uuid.uuid4()),
            "query": "CVE-2024-1234这个漏洞危险吗？",
            "type": "vulnerability_inquiry",
            "expected_entities": ["CVE-2024-1234"],
            "expected_response_type": "vulnerability_analysis"
        },
        {
            "id": str(uuid.uuid4()),
            "query": "镜像redis:v2有哪些漏洞？",
            "type": "image_vulnerability_check",
            "expected_entities": ["redis:v2"],
            "expected_response_type": "vulnerability_list"
        },
        {
            "id": str(uuid.uuid4()),
            "query": "哪些主机需要处理高危漏洞？",
            "type": "host_security_status",
            "expected_entities": [],
            "expected_response_type": "host_list"
        },
        {
            "id": str(uuid.uuid4()),
            "query": "如何修复SQL注入漏洞？",
            "type": "remediation_guidance",
            "expected_entities": ["SQL注入"],
            "expected_response_type": "fix_instructions"
        },
        {
            "id": str(uuid.uuid4()),
            "query": "生产环境中有多少个容器存在安全风险？",
            "type": "security_statistics",
            "expected_entities": ["生产环境"],
            "expected_response_type": "statistics"
        }
    ]
    return queries

def save_mock_data():
    """保存所有Mock数据到JSON文件"""
    print("开始生成Mock数据...")
    
    # 创建数据目录
    data_dir = os.path.dirname(__file__) + "/../structured"
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成数据
    hosts = generate_hosts_data(100)
    images = generate_images_data(200)
    vulnerabilities = generate_vulnerabilities_data(500)
    relationships = generate_relationships_data(hosts, images, vulnerabilities)
    sample_queries = generate_sample_queries()
    
    # 保存到文件
    with open(f"{data_dir}/hosts.json", "w", encoding="utf-8") as f:
        json.dump(hosts, f, ensure_ascii=False, indent=2)
    
    with open(f"{data_dir}/images.json", "w", encoding="utf-8") as f:
        json.dump(images, f, ensure_ascii=False, indent=2)
    
    with open(f"{data_dir}/vulnerabilities.json", "w", encoding="utf-8") as f:
        json.dump(vulnerabilities, f, ensure_ascii=False, indent=2)
    
    with open(f"{data_dir}/relationships.json", "w", encoding="utf-8") as f:
        json.dump(relationships, f, ensure_ascii=False, indent=2)
    
    with open(f"{data_dir}/../sample_queries.json", "w", encoding="utf-8") as f:
        json.dump(sample_queries, f, ensure_ascii=False, indent=2)
    
    print(f"Mock数据生成完成:")
    print(f"- 主机数据: {len(hosts)} 条")
    print(f"- 镜像数据: {len(images)} 条")
    print(f"- 漏洞数据: {len(vulnerabilities)} 条")
    print(f"- 关系数据: {len(relationships)} 条")
    print(f"- 测试查询: {len(sample_queries)} 条")

if __name__ == "__main__":
    save_mock_data()
