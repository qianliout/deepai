# AIOps RAG系统知识图谱部署指南

## 概述
本指南将帮助你部署Neo4j图数据库，创建MySQL测试数据，并为RAG系统集成知识图谱功能。

## 前置条件
- Docker 和 Docker Compose 已安装
- MySQL 数据库已安装并运行
- Python 3.9+ 环境
- 足够的磁盘空间（建议至少5GB）

## 部署步骤

### 第一步：Neo4j Docker部署

#### 1.1 自动部署（推荐）
```bash
# 进入rag目录
cd /Users/liuqianli/work/python/deepai/rag

# 给脚本执行权限
chmod +x setup_neo4j.sh

# 运行部署脚本
./setup_neo4j.sh
```

#### 1.2 手动部署
```bash
# 启动Neo4j服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs neo4j

# 等待服务完全启动（约30-60秒）
```

#### 1.3 验证Neo4j部署
1. 访问Web界面：http://localhost:7474
2. 使用以下凭据登录：
   - 用户名：`neo4j`
   - 密码：`aiops123456`
3. 在查询界面执行：`RETURN "Hello Neo4j!" as message`

### 第二步：MySQL测试数据准备

#### 2.1 创建数据库和表
```bash
# 连接到MySQL（请根据你的实际配置修改）
mysql -u root -p

# 执行测试数据脚本
source aiops_test_data.sql

# 或者直接导入
mysql -u root -p < aiops_test_data.sql
```

#### 2.2 验证MySQL数据
```sql
-- 连接到aiops_system数据库
USE aiops_system;

-- 查看数据统计
SELECT '主机数量' as 项目, COUNT(*) as 数量 FROM hosts
UNION ALL
SELECT '镜像数量', COUNT(*) FROM images
UNION ALL
SELECT '漏洞数量', COUNT(*) FROM vulnerabilities
UNION ALL
SELECT '主机镜像关系', COUNT(*) FROM host_images
UNION ALL
SELECT '镜像漏洞关系', COUNT(*) FROM image_vulnerabilities;

-- 查看漏洞严重程度分布
SELECT severity as 严重程度, COUNT(*) as 数量 
FROM vulnerabilities 
GROUP BY severity 
ORDER BY FIELD(severity, 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW');
```

### 第三步：Neo4j初始化

#### 3.1 创建约束和索引
```bash
# 执行初始化脚本
docker exec aiops-neo4j cypher-shell -u neo4j -p aiops123456 -f /var/lib/neo4j/import/neo4j_init.cypher
```

#### 3.2 手动执行（如果脚本方式失败）
```bash
# 进入Neo4j容器
docker exec -it aiops-neo4j cypher-shell -u neo4j -p aiops123456

# 然后复制粘贴neo4j_init.cypher中的内容执行
```

### 第四步：验证部署

#### 4.1 Neo4j验证查询
在Neo4j Web界面执行以下查询：

```cypher
// 查看所有节点类型和数量
MATCH (n) RETURN labels(n) as NodeType, count(n) as Count;

// 查看所有关系类型和数量
MATCH ()-[r]->() RETURN type(r) as RelationType, count(r) as Count;

// 查询漏洞影响分析示例
MATCH (v:Vulnerability {cve_id: 'CVE-2023-44487'})<-[:HAS_VULNERABILITY]-(i:Image)<-[:HAS_IMAGE]-(h:Host)
RETURN h.hostname, h.ip_address, i.image_name, i.image_tag, v.severity;
```

#### 4.2 MySQL验证查询
```sql
-- 查看主机漏洞汇总视图
SELECT * FROM host_vulnerability_summary ORDER BY max_cvss_score DESC;

-- 查看特定主机的详细漏洞信息
SELECT 
    h.hostname,
    i.image_name,
    i.image_tag,
    v.cve_id,
    v.severity,
    v.cvss_score,
    iv.affected_package,
    iv.fixed_version
FROM hosts h
JOIN host_images hi ON h.id = hi.host_id
JOIN images i ON hi.image_id = i.id
JOIN image_vulnerabilities iv ON i.id = iv.image_id
JOIN vulnerabilities v ON iv.vulnerability_id = v.id
WHERE h.hostname = 'web-server-01'
ORDER BY v.cvss_score DESC;
```

## 配置说明

### Neo4j配置参数
在 `config.py` 中的 `Neo4jConfig` 类包含以下配置：

```python
uri: "bolt://localhost:7687"          # Neo4j连接地址
username: "neo4j"                     # 用户名
password: "aiops123456"               # 密码
database: "neo4j"                     # 数据库名称
max_connection_pool_size: 50          # 连接池大小
connection_timeout: 30                # 连接超时时间
```

### MySQL配置参数
在 `config.py` 中的 `MySQLConfig` 类包含以下配置：

```python
host: "localhost"                     # MySQL主机地址
port: 3306                           # MySQL端口
username: "root"                     # MySQL用户名
password: "root"                     # MySQL密码（请修改为实际密码）
database: "aiops_system"            # 数据库名称
```

## 常见问题排查

### Neo4j相关问题

#### 问题1：Neo4j启动失败
```bash
# 查看详细日志
docker-compose logs neo4j

# 检查端口占用
lsof -i :7474
lsof -i :7687

# 重启服务
docker-compose restart neo4j
```

#### 问题2：连接超时
```bash
# 检查防火墙设置
# 确保7474和7687端口开放

# 检查Docker网络
docker network ls
docker network inspect aiops-network
```

#### 问题3：内存不足
```yaml
# 在docker-compose.yml中调整内存设置
environment:
  - NEO4J_dbms_memory_heap_initial__size=256m
  - NEO4J_dbms_memory_heap_max__size=1G
```

### MySQL相关问题

#### 问题1：权限不足
```sql
-- 创建专用用户
CREATE USER 'aiops'@'localhost' IDENTIFIED BY 'aiops123456';
GRANT ALL PRIVILEGES ON aiops_system.* TO 'aiops'@'localhost';
FLUSH PRIVILEGES;
```

#### 问题2：字符集问题
```sql
-- 检查字符集
SHOW VARIABLES LIKE 'character_set%';

-- 修改数据库字符集
ALTER DATABASE aiops_system CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

## 性能优化建议

### Neo4j优化
1. **内存配置**：根据数据量调整堆内存和页缓存
2. **索引优化**：为常用查询字段创建索引
3. **查询优化**：使用EXPLAIN分析查询性能

### MySQL优化
1. **索引优化**：为外键和常用查询字段创建索引
2. **连接池**：配置合适的连接池大小
3. **查询缓存**：启用查询缓存提升性能

## 下一步计划

1. **开发Neo4j管理器**：创建 `neo4j_manager.py`
2. **实现数据同步**：MySQL到Neo4j的数据同步功能
3. **集成RAG系统**：将知识图谱查询集成到现有RAG流程
4. **开发查询接口**：实现漏洞影响分析等业务查询
5. **性能测试**：测试大数据量下的查询性能

## 联系支持

如果在部署过程中遇到问题，请：
1. 检查日志文件
2. 参考常见问题排查部分
3. 确保所有前置条件满足
4. 验证网络连接和端口开放情况
