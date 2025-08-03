// AIOps知识图谱初始化Cypher脚本
// 用于在Neo4j中创建约束、索引和示例数据

// ========================================
// 1. 清理现有数据（谨慎使用）
// ========================================
// MATCH (n) DETACH DELETE n;

// ========================================
// 2. 创建约束
// ========================================

// 主机节点唯一约束
CREATE CONSTRAINT host_id_unique IF NOT EXISTS FOR (h:Host) REQUIRE h.id IS UNIQUE;

// 镜像节点唯一约束
CREATE CONSTRAINT image_id_unique IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE;

// 漏洞节点唯一约束
CREATE CONSTRAINT vulnerability_id_unique IF NOT EXISTS FOR (v:Vulnerability) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT cve_id_unique IF NOT EXISTS FOR (v:Vulnerability) REQUIRE v.cve_id IS UNIQUE;

// ========================================
// 3. 创建索引
// ========================================

// 主机索引
CREATE INDEX host_hostname_index IF NOT EXISTS FOR (h:Host) ON (h.hostname);
CREATE INDEX host_ip_index IF NOT EXISTS FOR (h:Host) ON (h.ip_address);
CREATE INDEX host_status_index IF NOT EXISTS FOR (h:Host) ON (h.status);
CREATE INDEX host_location_index IF NOT EXISTS FOR (h:Host) ON (h.location);

// 镜像索引
CREATE INDEX image_name_index IF NOT EXISTS FOR (i:Image) ON (i.image_name);
CREATE INDEX image_tag_index IF NOT EXISTS FOR (i:Image) ON (i.image_tag);
CREATE INDEX image_registry_index IF NOT EXISTS FOR (i:Image) ON (i.registry);

// 漏洞索引
CREATE INDEX vulnerability_severity_index IF NOT EXISTS FOR (v:Vulnerability) ON (v.severity);
CREATE INDEX vulnerability_cvss_index IF NOT EXISTS FOR (v:Vulnerability) ON (v.cvss_score);
CREATE INDEX vulnerability_published_index IF NOT EXISTS FOR (v:Vulnerability) ON (v.published_date);

// ========================================
// 4. 创建示例数据（用于测试）
// ========================================

// 创建主机节点
CREATE (h1:Host {
    id: 1,
    hostname: 'web-server-01',
    ip_address: '192.168.1.10',
    os_type: 'Ubuntu',
    os_version: '20.04.6',
    cpu_cores: 4,
    memory_gb: 8,
    disk_gb: 100,
    status: 'online',
    location: '北京机房A',
    created_at: datetime('2023-01-15T10:00:00Z'),
    updated_at: datetime('2023-12-01T15:30:00Z')
});

CREATE (h2:Host {
    id: 2,
    hostname: 'web-server-02',
    ip_address: '192.168.1.11',
    os_type: 'Ubuntu',
    os_version: '20.04.6',
    cpu_cores: 4,
    memory_gb: 8,
    disk_gb: 100,
    status: 'online',
    location: '北京机房A',
    created_at: datetime('2023-01-15T10:00:00Z'),
    updated_at: datetime('2023-12-01T15:30:00Z')
});

CREATE (h3:Host {
    id: 3,
    hostname: 'db-server-01',
    ip_address: '192.168.1.20',
    os_type: 'CentOS',
    os_version: '7.9.2009',
    cpu_cores: 8,
    memory_gb: 32,
    disk_gb: 500,
    status: 'online',
    location: '北京机房B',
    created_at: datetime('2023-01-15T10:00:00Z'),
    updated_at: datetime('2023-12-01T15:30:00Z')
});

// 创建镜像节点
CREATE (i1:Image {
    id: 1,
    image_name: 'nginx',
    image_tag: '1.20.2',
    registry: 'docker.io',
    size_mb: 142,
    architecture: 'amd64',
    os: 'linux',
    created_at: datetime('2023-01-01T00:00:00Z'),
    updated_at: datetime('2023-01-01T00:00:00Z')
});

CREATE (i2:Image {
    id: 4,
    image_name: 'mysql',
    image_tag: '8.0.32',
    registry: 'docker.io',
    size_mb: 521,
    architecture: 'amd64',
    os: 'linux',
    created_at: datetime('2023-01-01T00:00:00Z'),
    updated_at: datetime('2023-01-01T00:00:00Z')
});

CREATE (i3:Image {
    id: 8,
    image_name: 'node',
    image_tag: '16.19.0',
    registry: 'docker.io',
    size_mb: 993,
    architecture: 'amd64',
    os: 'linux',
    created_at: datetime('2023-01-01T00:00:00Z'),
    updated_at: datetime('2023-01-01T00:00:00Z')
});

// 创建漏洞节点
CREATE (v1:Vulnerability {
    id: 1,
    cve_id: 'CVE-2023-44487',
    severity: 'HIGH',
    cvss_score: 7.5,
    cvss_vector: 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H',
    description: 'HTTP/2 Rapid Reset attack vulnerability allowing DoS attacks',
    fix_suggestion: 'Upgrade to patched version or disable HTTP/2 if not needed',
    published_date: date('2023-10-10'),
    source: 'NVD',
    created_at: datetime('2023-10-10T00:00:00Z'),
    updated_at: datetime('2023-10-10T00:00:00Z')
});

CREATE (v2:Vulnerability {
    id: 5,
    cve_id: 'CVE-2023-1234',
    severity: 'CRITICAL',
    cvss_score: 9.9,
    cvss_vector: 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H',
    description: 'Remote code execution vulnerability in Node.js HTTP parser',
    fix_suggestion: 'Update Node.js to 16.20.2, 18.17.1, or 20.6.0',
    published_date: date('2023-09-20'),
    source: 'NVD',
    created_at: datetime('2023-09-20T00:00:00Z'),
    updated_at: datetime('2023-09-20T00:00:00Z')
});

CREATE (v3:Vulnerability {
    id: 7,
    cve_id: 'CVE-2023-5432',
    severity: 'HIGH',
    cvss_score: 8.1,
    cvss_vector: 'CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H',
    description: 'SQL injection vulnerability in MySQL connector',
    fix_suggestion: 'Update MySQL to 8.0.36 or apply security patch',
    published_date: date('2023-11-20'),
    source: 'NVD',
    created_at: datetime('2023-11-20T00:00:00Z'),
    updated_at: datetime('2023-11-20T00:00:00Z')
});

// ========================================
// 5. 创建关系
// ========================================

// 主机-镜像关系
MATCH (h1:Host {id: 1}), (i1:Image {id: 1})
CREATE (h1)-[:HAS_IMAGE {
    container_name: 'web-nginx-01',
    container_id: 'abc123def456',
    status: 'running',
    ports: '80:80,443:443',
    volumes: '/var/log/nginx:/var/log/nginx',
    created_at: datetime('2023-01-15T10:30:00Z')
}]->(i1);

MATCH (h1:Host {id: 1}), (i3:Image {id: 8})
CREATE (h1)-[:HAS_IMAGE {
    container_name: 'web-app-01',
    container_id: 'def456ghi789',
    status: 'running',
    ports: '3000:3000',
    volumes: '/app/data:/data',
    created_at: datetime('2023-01-15T10:30:00Z')
}]->(i3);

MATCH (h3:Host {id: 3}), (i2:Image {id: 4})
CREATE (h3)-[:HAS_IMAGE {
    container_name: 'mysql-primary',
    container_id: 'mno345pqr678',
    status: 'running',
    ports: '3306:3306',
    volumes: '/var/lib/mysql:/var/lib/mysql',
    created_at: datetime('2023-01-15T10:30:00Z')
}]->(i2);

// 镜像-漏洞关系
MATCH (i1:Image {id: 1}), (v1:Vulnerability {id: 1})
CREATE (i1)-[:HAS_VULNERABILITY {
    affected_package: 'nginx',
    package_version: '1.20.2',
    fixed_version: '1.20.3',
    layer_hash: 'sha256:abc123',
    detected_at: datetime('2023-10-15T00:00:00Z')
}]->(v1);

MATCH (i3:Image {id: 8}), (v2:Vulnerability {id: 5})
CREATE (i3)-[:HAS_VULNERABILITY {
    affected_package: 'nodejs',
    package_version: '16.19.0',
    fixed_version: '16.20.2',
    layer_hash: 'sha256:efg123',
    detected_at: datetime('2023-09-25T00:00:00Z')
}]->(v2);

MATCH (i2:Image {id: 4}), (v3:Vulnerability {id: 7})
CREATE (i2)-[:HAS_VULNERABILITY {
    affected_package: 'mysql-server',
    package_version: '8.0.32',
    fixed_version: '8.0.36',
    layer_hash: 'sha256:pqr678',
    detected_at: datetime('2023-11-25T00:00:00Z')
}]->(v3);

// ========================================
// 6. 验证数据
// ========================================

// 统计节点数量
MATCH (h:Host) RETURN 'Host节点数量' as type, count(h) as count
UNION ALL
MATCH (i:Image) RETURN 'Image节点数量' as type, count(i) as count
UNION ALL
MATCH (v:Vulnerability) RETURN 'Vulnerability节点数量' as type, count(v) as count;

// 统计关系数量
MATCH ()-[r:HAS_IMAGE]->() RETURN 'HAS_IMAGE关系数量' as type, count(r) as count
UNION ALL
MATCH ()-[r:HAS_VULNERABILITY]->() RETURN 'HAS_VULNERABILITY关系数量' as type, count(r) as count;

// ========================================
// 7. 示例查询
// ========================================

// 查询所有主机及其镜像
MATCH (h:Host)-[r:HAS_IMAGE]->(i:Image)
RETURN h.hostname, h.ip_address, i.image_name, i.image_tag, r.container_name, r.status
ORDER BY h.hostname;

// 查询所有漏洞及其影响的主机
MATCH (h:Host)-[:HAS_IMAGE]->(i:Image)-[:HAS_VULNERABILITY]->(v:Vulnerability)
RETURN v.cve_id, v.severity, v.cvss_score, 
       collect(DISTINCT h.hostname) as affected_hosts,
       collect(DISTINCT i.image_name + ':' + i.image_tag) as affected_images
ORDER BY v.cvss_score DESC;

// 查询特定主机的风险评估
MATCH (h:Host {hostname: 'web-server-01'})-[:HAS_IMAGE]->(i:Image)-[:HAS_VULNERABILITY]->(v:Vulnerability)
RETURN h.hostname, i.image_name, i.image_tag, v.cve_id, v.severity, v.cvss_score
ORDER BY v.cvss_score DESC;
