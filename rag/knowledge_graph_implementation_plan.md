# AIOps RAG系统知识图谱集成实施方案

## 项目概述
为AIOps运维系统的RAG知识库增加知识图谱功能，使用Neo4j图数据库存储主机、镜像、漏洞之间的关系，提升漏洞影响分析和处理决策能力。

## 数据模型设计

### MySQL数据表结构
```sql
-- 主机节点表
CREATE TABLE hosts (
    id INT PRIMARY KEY AUTO_INCREMENT,
    hostname VARCHAR(255) NOT NULL,
    ip_address VARCHAR(45) NOT NULL,
    os_type VARCHAR(100),
    os_version VARCHAR(100),
    status ENUM('online', 'offline', 'maintenance') DEFAULT 'online',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 镜像表
CREATE TABLE images (
    id INT PRIMARY KEY AUTO_INCREMENT,
    image_name VARCHAR(255) NOT NULL,
    image_tag VARCHAR(100) NOT NULL,
    registry VARCHAR(255),
    size_mb BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_image (image_name, image_tag)
);

-- 漏洞表
CREATE TABLE vulnerabilities (
    id INT PRIMARY KEY AUTO_INCREMENT,
    cve_id VARCHAR(50) UNIQUE NOT NULL,
    severity ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') NOT NULL,
    cvss_score DECIMAL(3,1),
    description TEXT,
    fix_suggestion TEXT,
    published_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 主机镜像关系表
CREATE TABLE host_images (
    id INT PRIMARY KEY AUTO_INCREMENT,
    host_id INT NOT NULL,
    image_id INT NOT NULL,
    container_name VARCHAR(255),
    status ENUM('running', 'stopped', 'paused') DEFAULT 'running',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (host_id) REFERENCES hosts(id) ON DELETE CASCADE,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    UNIQUE KEY unique_host_image (host_id, image_id, container_name)
);

-- 镜像漏洞关系表
CREATE TABLE image_vulnerabilities (
    id INT PRIMARY KEY AUTO_INCREMENT,
    image_id INT NOT NULL,
    vulnerability_id INT NOT NULL,
    affected_package VARCHAR(255),
    fixed_version VARCHAR(100),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY (vulnerability_id) REFERENCES vulnerabilities(id) ON DELETE CASCADE,
    UNIQUE KEY unique_image_vuln (image_id, vulnerability_id, affected_package)
);
```

### Neo4j图数据模型
```cypher
-- 节点类型定义
(:Host {id, hostname, ip_address, os_type, os_version, status})
(:Image {id, image_name, image_tag, registry, size_mb})
(:Vulnerability {id, cve_id, severity, cvss_score, description, fix_suggestion})

-- 关系类型定义
(:Host)-[:HAS_IMAGE {container_name, status}]->(:Image)
(:Image)-[:HAS_VULNERABILITY {affected_package, fixed_version}]->(:Vulnerability)
```

## 实施步骤

### 第一步：Neo4j Docker部署

#### 1.1 创建docker-compose.yml
```yaml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.15-community
    container_name: aiops-neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/aiops123456
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
      - NEO4J_dbms_memory_heap_initial__size=512m
      - NEO4J_dbms_memory_heap_max__size=2G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
      - neo4j_plugins:/plugins
    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  neo4j_plugins:
```

#### 1.2 启动命令
```bash
# 在rag目录下创建docker-compose.yml后执行
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs neo4j
```

#### 1.3 访问Neo4j
- Web界面: http://localhost:7474
- 用户名: neo4j
- 密码: aiops123456

### 第二步：配置管理

#### 2.1 在config.py中添加Neo4j配置
```python
class Neo4jConfig(BaseModel):
    """Neo4j图数据库配置"""
    
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j连接URI")
    username: str = Field(default="neo4j", description="用户名")
    password: str = Field(default="aiops123456", description="密码")
    database: str = Field(default="neo4j", description="数据库名称")
    max_connection_lifetime: int = Field(default=3600, description="连接最大生存时间")
    max_connection_pool_size: int = Field(default=50, description="连接池最大大小")
    connection_timeout: int = Field(default=30, description="连接超时时间")
    
    class Config:
        extra = "forbid"
```

### 第三步：测试数据准备

#### 3.1 MySQL测试数据SQL脚本
```sql
-- 插入主机数据
INSERT INTO hosts (hostname, ip_address, os_type, os_version, status) VALUES
('web-server-01', '192.168.1.10', 'Ubuntu', '20.04', 'online'),
('web-server-02', '192.168.1.11', 'Ubuntu', '20.04', 'online'),
('db-server-01', '192.168.1.20', 'CentOS', '7.9', 'online'),
('app-server-01', '192.168.1.30', 'Ubuntu', '22.04', 'online'),
('monitoring-01', '192.168.1.40', 'Alpine', '3.16', 'online');

-- 插入镜像数据
INSERT INTO images (image_name, image_tag, registry, size_mb) VALUES
('nginx', '1.20.2', 'docker.io', 142),
('nginx', '1.21.6', 'docker.io', 145),
('mysql', '8.0.32', 'docker.io', 521),
('redis', '7.0.8', 'docker.io', 117),
('node', '16.19.0', 'docker.io', 993),
('python', '3.9.16', 'docker.io', 885),
('prometheus', '2.42.0', 'docker.io', 201);

-- 插入漏洞数据
INSERT INTO vulnerabilities (cve_id, severity, cvss_score, description, fix_suggestion, published_date) VALUES
('CVE-2023-44487', 'HIGH', 7.5, 'HTTP/2 Rapid Reset attack vulnerability', 'Upgrade to patched version or disable HTTP/2', '2023-10-10'),
('CVE-2023-38545', 'HIGH', 9.8, 'SOCKS5 heap buffer overflow in curl', 'Update curl to version 8.4.0 or later', '2023-10-11'),
('CVE-2023-4911', 'HIGH', 7.8, 'Buffer overflow in glibc dynamic loader', 'Update glibc to patched version', '2023-10-03'),
('CVE-2023-5678', 'MEDIUM', 5.3, 'OpenSSL denial of service vulnerability', 'Update OpenSSL to 3.0.12 or 1.1.1w', '2023-11-15'),
('CVE-2023-1234', 'CRITICAL', 9.9, 'Remote code execution in Node.js', 'Update Node.js to 16.20.2 or later', '2023-09-20');

-- 插入主机镜像关系
INSERT INTO host_images (host_id, image_id, container_name, status) VALUES
(1, 1, 'web-nginx-01', 'running'),
(1, 5, 'web-app-01', 'running'),
(2, 2, 'web-nginx-02', 'running'),
(2, 5, 'web-app-02', 'running'),
(3, 3, 'mysql-primary', 'running'),
(3, 4, 'redis-cache', 'running'),
(4, 6, 'python-api', 'running'),
(5, 7, 'prometheus-monitor', 'running');

-- 插入镜像漏洞关系
INSERT INTO image_vulnerabilities (image_id, vulnerability_id, affected_package, fixed_version) VALUES
(1, 1, 'nginx', '1.20.3'),
(2, 1, 'nginx', '1.21.7'),
(3, 4, 'openssl', '3.0.12'),
(5, 5, 'nodejs', '16.20.2'),
(5, 2, 'curl', '8.4.0'),
(6, 3, 'glibc', '2.35-1'),
(7, 1, 'golang', '1.19.5');
```

### 第四步：开发计划

#### 4.1 Neo4j管理器开发 (neo4j_manager.py)
- 连接管理和事务处理
- 节点和关系的CRUD操作
- 批量数据导入功能
- 图查询接口

#### 4.2 数据同步功能
- MySQL数据读取
- 数据转换和映射
- Neo4j批量写入
- 增量同步机制

#### 4.3 RAG系统集成
- 图查询功能集成
- 漏洞影响分析查询
- 智能问答增强

### 第五步：典型查询场景

#### 5.1 漏洞影响分析
```cypher
// 查找CVE-2023-44487影响的所有主机和镜像
MATCH (v:Vulnerability {cve_id: 'CVE-2023-44487'})<-[:HAS_VULNERABILITY]-(i:Image)<-[:HAS_IMAGE]-(h:Host)
RETURN h.hostname, h.ip_address, i.image_name, i.image_tag, v.severity
```

#### 5.2 主机风险评估
```cypher
// 查看web-server-01的所有漏洞
MATCH (h:Host {hostname: 'web-server-01'})-[:HAS_IMAGE]->(i:Image)-[:HAS_VULNERABILITY]->(v:Vulnerability)
RETURN h.hostname, i.image_name, v.cve_id, v.severity, v.cvss_score
ORDER BY v.cvss_score DESC
```

#### 5.3 修复优先级排序
```cypher
// 按漏洞影响范围和严重程度排序
MATCH (v:Vulnerability)<-[:HAS_VULNERABILITY]-(i:Image)<-[:HAS_IMAGE]-(h:Host)
WITH v, count(DISTINCT h) as affected_hosts, count(DISTINCT i) as affected_images
RETURN v.cve_id, v.severity, v.cvss_score, affected_hosts, affected_images
ORDER BY v.cvss_score DESC, affected_hosts DESC
```

## 下一步行动
1. 创建docker-compose.yml并启动Neo4j
2. 更新config.py添加Neo4j配置
3. 执行MySQL建表和测试数据SQL
4. 开发neo4j_manager.py
5. 实现数据同步功能
6. 集成到RAG系统

## 预期效果
- 快速定位漏洞影响范围
- 智能化的修复建议
- 可视化的依赖关系分析
- 增强的运维决策支持
