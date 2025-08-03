-- AIOps知识图谱测试数据SQL脚本
-- 使用前请确保已连接到MySQL数据库

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS aiops_system CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE aiops_system;

-- 1. 主机节点表
DROP TABLE IF EXISTS host_images;
DROP TABLE IF EXISTS image_vulnerabilities;
DROP TABLE IF EXISTS hosts;
DROP TABLE IF EXISTS images;
DROP TABLE IF EXISTS vulnerabilities;

CREATE TABLE hosts (
    id INT PRIMARY KEY AUTO_INCREMENT,
    hostname VARCHAR(255) NOT NULL UNIQUE,
    ip_address VARCHAR(45) NOT NULL UNIQUE,
    os_type VARCHAR(100),
    os_version VARCHAR(100),
    cpu_cores INT DEFAULT 0,
    memory_gb INT DEFAULT 0,
    disk_gb INT DEFAULT 0,
    status ENUM('online', 'offline', 'maintenance') DEFAULT 'online',
    location VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_hostname (hostname),
    INDEX idx_ip (ip_address),
    INDEX idx_status (status)
);

-- 2. 镜像表
CREATE TABLE images (
    id INT PRIMARY KEY AUTO_INCREMENT,
    image_name VARCHAR(255) NOT NULL,
    image_tag VARCHAR(100) NOT NULL,
    registry VARCHAR(255) DEFAULT 'docker.io',
    size_mb BIGINT DEFAULT 0,
    architecture VARCHAR(50) DEFAULT 'amd64',
    os VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_image (image_name, image_tag, registry),
    INDEX idx_image_name (image_name),
    INDEX idx_registry (registry)
);

-- 3. 漏洞表
CREATE TABLE vulnerabilities (
    id INT PRIMARY KEY AUTO_INCREMENT,
    cve_id VARCHAR(50) UNIQUE NOT NULL,
    severity ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') NOT NULL,
    cvss_score DECIMAL(3,1) DEFAULT 0.0,
    cvss_vector VARCHAR(255),
    description TEXT,
    fix_suggestion TEXT,
    published_date DATE,
    modified_date DATE,
    source VARCHAR(100) DEFAULT 'NVD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_cve_id (cve_id),
    INDEX idx_severity (severity),
    INDEX idx_cvss_score (cvss_score),
    INDEX idx_published_date (published_date)
);

-- 4. 主机镜像关系表
CREATE TABLE host_images (
    id INT PRIMARY KEY AUTO_INCREMENT,
    host_id INT NOT NULL,
    image_id INT NOT NULL,
    container_name VARCHAR(255),
    container_id VARCHAR(64),
    status ENUM('running', 'stopped', 'paused', 'restarting') DEFAULT 'running',
    ports VARCHAR(500),
    volumes VARCHAR(1000),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (host_id) REFERENCES hosts(id) ON DELETE CASCADE,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    UNIQUE KEY unique_host_container (host_id, container_name),
    INDEX idx_host_id (host_id),
    INDEX idx_image_id (image_id),
    INDEX idx_status (status)
);

-- 5. 镜像漏洞关系表
CREATE TABLE image_vulnerabilities (
    id INT PRIMARY KEY AUTO_INCREMENT,
    image_id INT NOT NULL,
    vulnerability_id INT NOT NULL,
    affected_package VARCHAR(255),
    package_version VARCHAR(100),
    fixed_version VARCHAR(100),
    layer_hash VARCHAR(64),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY (vulnerability_id) REFERENCES vulnerabilities(id) ON DELETE CASCADE,
    UNIQUE KEY unique_image_vuln_pkg (image_id, vulnerability_id, affected_package),
    INDEX idx_image_id (image_id),
    INDEX idx_vulnerability_id (vulnerability_id),
    INDEX idx_package (affected_package)
);

-- 插入测试数据

-- 插入主机数据
INSERT INTO hosts (hostname, ip_address, os_type, os_version, cpu_cores, memory_gb, disk_gb, status, location) VALUES
('web-server-01', '192.168.1.10', 'Ubuntu', '20.04.6', 4, 8, 100, 'online', '北京机房A'),
('web-server-02', '192.168.1.11', 'Ubuntu', '20.04.6', 4, 8, 100, 'online', '北京机房A'),
('db-server-01', '192.168.1.20', 'CentOS', '7.9.2009', 8, 32, 500, 'online', '北京机房B'),
('app-server-01', '192.168.1.30', 'Ubuntu', '22.04.3', 6, 16, 200, 'online', '上海机房A'),
('monitoring-01', '192.168.1.40', 'Alpine', '3.16.7', 2, 4, 50, 'online', '深圳机房A'),
('cache-server-01', '192.168.1.50', 'Ubuntu', '20.04.6', 4, 16, 100, 'online', '北京机房B'),
('lb-server-01', '192.168.1.60', 'CentOS', '8.5.2111', 2, 8, 50, 'online', '北京机房A');

-- 插入镜像数据
INSERT INTO images (image_name, image_tag, registry, size_mb, architecture, os) VALUES
('nginx', '1.20.2', 'docker.io', 142, 'amd64', 'linux'),
('nginx', '1.21.6', 'docker.io', 145, 'amd64', 'linux'),
('nginx', '1.24.0', 'docker.io', 148, 'amd64', 'linux'),
('mysql', '8.0.32', 'docker.io', 521, 'amd64', 'linux'),
('mysql', '8.0.35', 'docker.io', 528, 'amd64', 'linux'),
('redis', '7.0.8', 'docker.io', 117, 'amd64', 'linux'),
('redis', '7.2.3', 'docker.io', 121, 'amd64', 'linux'),
('node', '16.19.0', 'docker.io', 993, 'amd64', 'linux'),
('node', '18.18.2', 'docker.io', 1024, 'amd64', 'linux'),
('python', '3.9.16', 'docker.io', 885, 'amd64', 'linux'),
('python', '3.11.6', 'docker.io', 912, 'amd64', 'linux'),
('prometheus', '2.42.0', 'docker.io', 201, 'amd64', 'linux'),
('grafana/grafana', '9.5.15', 'docker.io', 312, 'amd64', 'linux'),
('elasticsearch', '8.11.0', 'docker.io', 1200, 'amd64', 'linux');

-- 插入漏洞数据
INSERT INTO vulnerabilities (cve_id, severity, cvss_score, cvss_vector, description, fix_suggestion, published_date, source) VALUES
('CVE-2023-44487', 'HIGH', 7.5, 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H', 'HTTP/2 Rapid Reset attack vulnerability allowing DoS attacks', 'Upgrade to patched version or disable HTTP/2 if not needed', '2023-10-10', 'NVD'),
('CVE-2023-38545', 'HIGH', 9.8, 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H', 'SOCKS5 heap buffer overflow in curl library', 'Update curl to version 8.4.0 or later', '2023-10-11', 'NVD'),
('CVE-2023-4911', 'HIGH', 7.8, 'CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H', 'Buffer overflow in glibc dynamic loader (looney tunables)', 'Update glibc to patched version 2.38-1 or later', '2023-10-03', 'NVD'),
('CVE-2023-5678', 'MEDIUM', 5.3, 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L', 'OpenSSL denial of service vulnerability in X.509 certificate verification', 'Update OpenSSL to 3.0.12, 1.1.1w, or 3.1.4', '2023-11-15', 'NVD'),
('CVE-2023-1234', 'CRITICAL', 9.9, 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H', 'Remote code execution vulnerability in Node.js HTTP parser', 'Update Node.js to 16.20.2, 18.17.1, or 20.6.0', '2023-09-20', 'NVD'),
('CVE-2023-9876', 'MEDIUM', 6.1, 'CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N', 'Cross-site scripting vulnerability in Grafana dashboard', 'Update Grafana to version 9.5.16 or later', '2023-12-01', 'NVD'),
('CVE-2023-5432', 'HIGH', 8.1, 'CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H', 'SQL injection vulnerability in MySQL connector', 'Update MySQL to 8.0.36 or apply security patch', '2023-11-20', 'NVD'),
('CVE-2023-7890', 'LOW', 3.7, 'CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:N/A:L', 'Information disclosure in Redis AUTH command', 'Update Redis to 7.2.4 or disable AUTH if not needed', '2023-12-15', 'NVD');

-- 插入主机镜像关系数据
INSERT INTO host_images (host_id, image_id, container_name, container_id, status, ports, volumes) VALUES
(1, 1, 'web-nginx-01', 'abc123def456', 'running', '80:80,443:443', '/var/log/nginx:/var/log/nginx'),
(1, 8, 'web-app-01', 'def456ghi789', 'running', '3000:3000', '/app/data:/data'),
(2, 2, 'web-nginx-02', 'ghi789jkl012', 'running', '80:80,443:443', '/var/log/nginx:/var/log/nginx'),
(2, 9, 'web-app-02', 'jkl012mno345', 'running', '3000:3000', '/app/data:/data'),
(3, 4, 'mysql-primary', 'mno345pqr678', 'running', '3306:3306', '/var/lib/mysql:/var/lib/mysql'),
(3, 6, 'redis-cache', 'pqr678stu901', 'running', '6379:6379', '/data:/data'),
(4, 10, 'python-api', 'stu901vwx234', 'running', '8000:8000', '/app:/app'),
(4, 7, 'redis-session', 'vwx234yza567', 'running', '6380:6379', '/data:/data'),
(5, 12, 'prometheus-monitor', 'yza567bcd890', 'running', '9090:9090', '/prometheus:/prometheus'),
(5, 13, 'grafana-dashboard', 'bcd890efg123', 'running', '3001:3000', '/var/lib/grafana:/var/lib/grafana'),
(6, 6, 'redis-cluster-01', 'efg123hij456', 'running', '7000:6379', '/data:/data'),
(7, 3, 'lb-nginx', 'hij456klm789', 'running', '80:80,443:443', '/etc/nginx:/etc/nginx');

-- 插入镜像漏洞关系数据
INSERT INTO image_vulnerabilities (image_id, vulnerability_id, affected_package, package_version, fixed_version, layer_hash) VALUES
-- nginx 1.20.2 漏洞
(1, 1, 'nginx', '1.20.2', '1.20.3', 'sha256:abc123'),
(1, 3, 'glibc', '2.31-13', '2.31-14', 'sha256:def456'),
-- nginx 1.21.6 漏洞
(2, 1, 'nginx', '1.21.6', '1.21.7', 'sha256:ghi789'),
-- nginx 1.24.0 (较新版本，漏洞较少)
(3, 4, 'openssl', '3.0.2', '3.0.12', 'sha256:jkl012'),
-- mysql 8.0.32 漏洞
(4, 4, 'openssl', '3.0.2', '3.0.12', 'sha256:mno345'),
(4, 7, 'mysql-server', '8.0.32', '8.0.36', 'sha256:pqr678'),
-- mysql 8.0.35 (较新版本)
(5, 4, 'openssl', '3.0.10', '3.0.12', 'sha256:stu901'),
-- redis 7.0.8 漏洞
(6, 8, 'redis-server', '7.0.8', '7.2.4', 'sha256:vwx234'),
(6, 3, 'glibc', '2.31-13', '2.31-14', 'sha256:yza567'),
-- redis 7.2.3 (较新版本，漏洞较少)
(7, 8, 'redis-server', '7.2.3', '7.2.4', 'sha256:bcd890'),
-- node 16.19.0 漏洞
(8, 5, 'nodejs', '16.19.0', '16.20.2', 'sha256:efg123'),
(8, 2, 'curl', '7.81.0', '8.4.0', 'sha256:hij456'),
-- node 18.18.2 漏洞
(9, 2, 'curl', '7.81.0', '8.4.0', 'sha256:klm789'),
-- python 3.9.16 漏洞
(10, 3, 'glibc', '2.31-13', '2.31-14', 'sha256:nop012'),
(10, 2, 'curl', '7.81.0', '8.4.0', 'sha256:qrs345'),
-- python 3.11.6 漏洞
(11, 2, 'curl', '7.81.0', '8.4.0', 'sha256:tuv678'),
-- prometheus 2.42.0 漏洞
(12, 1, 'golang', '1.19.5', '1.19.13', 'sha256:wxy901'),
-- grafana 9.5.15 漏洞
(13, 6, 'grafana', '9.5.15', '9.5.16', 'sha256:zab234'),
-- elasticsearch 8.11.0 漏洞
(14, 4, 'openssl', '3.0.2', '3.0.12', 'sha256:cde567');

-- 创建视图用于快速查询
CREATE VIEW host_vulnerability_summary AS
SELECT 
    h.hostname,
    h.ip_address,
    h.status as host_status,
    COUNT(DISTINCT i.id) as total_images,
    COUNT(DISTINCT v.id) as total_vulnerabilities,
    COUNT(DISTINCT CASE WHEN v.severity = 'CRITICAL' THEN v.id END) as critical_vulns,
    COUNT(DISTINCT CASE WHEN v.severity = 'HIGH' THEN v.id END) as high_vulns,
    COUNT(DISTINCT CASE WHEN v.severity = 'MEDIUM' THEN v.id END) as medium_vulns,
    COUNT(DISTINCT CASE WHEN v.severity = 'LOW' THEN v.id END) as low_vulns,
    MAX(v.cvss_score) as max_cvss_score
FROM hosts h
LEFT JOIN host_images hi ON h.id = hi.host_id
LEFT JOIN images i ON hi.image_id = i.id
LEFT JOIN image_vulnerabilities iv ON i.id = iv.image_id
LEFT JOIN vulnerabilities v ON iv.vulnerability_id = v.id
GROUP BY h.id, h.hostname, h.ip_address, h.status;

-- 显示数据统计
SELECT '主机数量' as 项目, COUNT(*) as 数量 FROM hosts
UNION ALL
SELECT '镜像数量', COUNT(*) FROM images
UNION ALL
SELECT '漏洞数量', COUNT(*) FROM vulnerabilities
UNION ALL
SELECT '主机镜像关系', COUNT(*) FROM host_images
UNION ALL
SELECT '镜像漏洞关系', COUNT(*) FROM image_vulnerabilities;

-- 显示漏洞严重程度分布
SELECT severity as 严重程度, COUNT(*) as 数量 FROM vulnerabilities GROUP BY severity ORDER BY FIELD(severity, 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW');

COMMIT;
