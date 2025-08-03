#!/bin/bash

# AIOps Neo4j 部署和初始化脚本
# 使用方法: chmod +x setup_neo4j.sh && ./setup_neo4j.sh

set -e

echo "🚀 开始部署AIOps Neo4j知识图谱系统"
echo "=" * 50

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 检查当前目录是否有docker-compose.yml
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ 未找到docker-compose.yml文件，请确保在正确的目录下运行此脚本"
    exit 1
fi

echo "✅ Docker环境检查通过"

# 停止并删除现有容器（如果存在）
echo "🛑 停止现有Neo4j容器（如果存在）..."
docker-compose down -v 2>/dev/null || true

# 清理旧的数据卷（可选，谨慎使用）
read -p "是否清理旧的Neo4j数据？这将删除所有现有数据 (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️ 清理旧数据卷..."
    docker volume rm $(docker volume ls -q | grep neo4j) 2>/dev/null || true
fi

# 启动Neo4j服务
echo "🚀 启动Neo4j服务..."
docker-compose up -d

# 等待Neo4j启动
echo "⏳ 等待Neo4j启动..."
sleep 30

# 检查Neo4j是否启动成功
echo "🔍 检查Neo4j服务状态..."
if docker-compose ps | grep -q "Up"; then
    echo "✅ Neo4j服务启动成功"
else
    echo "❌ Neo4j服务启动失败，查看日志："
    docker-compose logs neo4j
    exit 1
fi

# 等待Neo4j完全就绪
echo "⏳ 等待Neo4j完全就绪..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if docker exec aiops-neo4j cypher-shell -u neo4j -p aiops123456 "RETURN 1" &>/dev/null; then
        echo "✅ Neo4j已就绪"
        break
    fi
    
    echo "⏳ 等待Neo4j就绪... (尝试 $attempt/$max_attempts)"
    sleep 5
    ((attempt++))
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Neo4j启动超时，请检查日志"
    docker-compose logs neo4j
    exit 1
fi

# 创建约束和索引
echo "📊 创建Neo4j约束和索引..."
docker exec aiops-neo4j cypher-shell -u neo4j -p aiops123456 << 'EOF'
// 创建唯一约束
CREATE CONSTRAINT host_id_unique IF NOT EXISTS FOR (h:Host) REQUIRE h.id IS UNIQUE;
CREATE CONSTRAINT image_id_unique IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE;
CREATE CONSTRAINT vulnerability_id_unique IF NOT EXISTS FOR (v:Vulnerability) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT cve_id_unique IF NOT EXISTS FOR (v:Vulnerability) REQUIRE v.cve_id IS UNIQUE;

// 创建索引
CREATE INDEX host_hostname_index IF NOT EXISTS FOR (h:Host) ON (h.hostname);
CREATE INDEX host_ip_index IF NOT EXISTS FOR (h:Host) ON (h.ip_address);
CREATE INDEX host_status_index IF NOT EXISTS FOR (h:Host) ON (h.status);
CREATE INDEX image_name_index IF NOT EXISTS FOR (i:Image) ON (i.image_name);
CREATE INDEX image_tag_index IF NOT EXISTS FOR (i:Image) ON (i.image_tag);
CREATE INDEX vulnerability_severity_index IF NOT EXISTS FOR (v:Vulnerability) ON (v.severity);
CREATE INDEX vulnerability_cvss_index IF NOT EXISTS FOR (v:Vulnerability) ON (v.cvss_score);

RETURN "约束和索引创建完成" as result;
EOF

if [ $? -eq 0 ]; then
    echo "✅ Neo4j约束和索引创建成功"
else
    echo "❌ Neo4j约束和索引创建失败"
    exit 1
fi

# 显示连接信息
echo ""
echo "🎉 Neo4j部署完成！"
echo "=" * 50
echo "📊 连接信息："
echo "  Web界面: http://localhost:7474"
echo "  Bolt连接: bolt://localhost:7687"
echo "  用户名: neo4j"
echo "  密码: aiops123456"
echo ""
echo "🔧 管理命令："
echo "  查看状态: docker-compose ps"
echo "  查看日志: docker-compose logs neo4j"
echo "  停止服务: docker-compose down"
echo "  重启服务: docker-compose restart"
echo ""
echo "📝 下一步："
echo "  1. 访问 http://localhost:7474 验证Neo4j是否正常运行"
echo "  2. 执行MySQL测试数据脚本: mysql -u root -p < aiops_test_data.sql"
echo "  3. 运行数据同步脚本将MySQL数据导入Neo4j"
echo ""
echo "✅ 部署完成！"
