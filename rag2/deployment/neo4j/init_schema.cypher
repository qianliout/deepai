// RAG2 Neo4j Knowledge Graph Schema Initialization
// Create constraints and indexes for AIOps knowledge graph

// Create constraints for unique identifiers
CREATE CONSTRAINT host_id_unique IF NOT EXISTS FOR (h:Host) REQUIRE h.id IS UNIQUE;
CREATE CONSTRAINT host_hostname_unique IF NOT EXISTS FOR (h:Host) REQUIRE h.hostname IS UNIQUE;
CREATE CONSTRAINT image_id_unique IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE;
CREATE CONSTRAINT vulnerability_id_unique IF NOT EXISTS FOR (v:Vulnerability) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT vulnerability_cve_unique IF NOT EXISTS FOR (v:Vulnerability) REQUIRE v.cve_id IS UNIQUE;

// Create indexes for better query performance
CREATE INDEX host_environment_idx IF NOT EXISTS FOR (h:Host) ON (h.environment);
CREATE INDEX host_datacenter_idx IF NOT EXISTS FOR (h:Host) ON (h.datacenter);
CREATE INDEX host_status_idx IF NOT EXISTS FOR (h:Host) ON (h.status);
CREATE INDEX image_name_idx IF NOT EXISTS FOR (i:Image) ON (i.name);
CREATE INDEX image_base_idx IF NOT EXISTS FOR (i:Image) ON (i.base_image);
CREATE INDEX image_registry_idx IF NOT EXISTS FOR (i:Image) ON (i.registry);
CREATE INDEX vulnerability_severity_idx IF NOT EXISTS FOR (v:Vulnerability) ON (v.severity);
CREATE INDEX vulnerability_category_idx IF NOT EXISTS FOR (v:Vulnerability) ON (v.category);
CREATE INDEX vulnerability_cvss_idx IF NOT EXISTS FOR (v:Vulnerability) ON (v.cvss_score);

// Create full-text search indexes
CREATE FULLTEXT INDEX host_search_idx IF NOT EXISTS FOR (h:Host) ON EACH [h.hostname, h.os];
CREATE FULLTEXT INDEX image_search_idx IF NOT EXISTS FOR (i:Image) ON EACH [i.name, i.base_image];
CREATE FULLTEXT INDEX vulnerability_search_idx IF NOT EXISTS FOR (v:Vulnerability) ON EACH [v.title, v.description, v.cve_id];

// Create sample data structure documentation
// Node types:
// - Host: Represents physical or virtual hosts
// - Image: Represents container images
// - Vulnerability: Represents security vulnerabilities
// - Package: Represents software packages (optional)
// - CVE: Represents Common Vulnerabilities and Exposures

// Relationship types:
// - RUNS: Host runs Image
// - HAS_VULNERABILITY: Image has Vulnerability
// - AFFECTS: Vulnerability affects Package
// - CONTAINS: Image contains Package
// - DEPLOYED_IN: Image deployed in Environment
// - LOCATED_IN: Host located in Datacenter

// Create some sample nodes for testing (will be replaced by actual data)
MERGE (env_prod:Environment {name: 'production', type: 'production'})
MERGE (env_staging:Environment {name: 'staging', type: 'staging'})
MERGE (env_dev:Environment {name: 'development', type: 'development'})

MERGE (dc1:Datacenter {name: 'dc1', location: 'US-East', region: 'us-east-1'})
MERGE (dc2:Datacenter {name: 'dc2', location: 'US-West', region: 'us-west-1'})
MERGE (dc3:Datacenter {name: 'dc3', location: 'EU-Central', region: 'eu-central-1'})

// Create severity levels for vulnerabilities
MERGE (critical:Severity {level: 'CRITICAL', score_min: 9.0, score_max: 10.0})
MERGE (high:Severity {level: 'HIGH', score_min: 7.0, score_max: 8.9})
MERGE (medium:Severity {level: 'MEDIUM', score_min: 4.0, score_max: 6.9})
MERGE (low:Severity {level: 'LOW', score_min: 0.1, score_max: 3.9})

// Create vulnerability categories
MERGE (cat_rce:Category {name: 'Remote Code Execution', type: 'vulnerability_category'})
MERGE (cat_sqli:Category {name: 'SQL Injection', type: 'vulnerability_category'})
MERGE (cat_xss:Category {name: 'Cross-Site Scripting', type: 'vulnerability_category'})
MERGE (cat_priv:Category {name: 'Privilege Escalation', type: 'vulnerability_category'})
MERGE (cat_dos:Category {name: 'Denial of Service', type: 'vulnerability_category'})
MERGE (cat_info:Category {name: 'Information Disclosure', type: 'vulnerability_category'})

// Create common base images
MERGE (base_ubuntu:BaseImage {name: 'ubuntu', official: true})
MERGE (base_alpine:BaseImage {name: 'alpine', official: true})
MERGE (base_centos:BaseImage {name: 'centos', official: true})
MERGE (base_debian:BaseImage {name: 'debian', official: true})

// Create registries
MERGE (reg_docker:Registry {name: 'docker.io', type: 'public', official: true})
MERGE (reg_gcr:Registry {name: 'gcr.io', type: 'public', provider: 'Google'})
MERGE (reg_quay:Registry {name: 'quay.io', type: 'public', provider: 'Red Hat'})
MERGE (reg_private:Registry {name: 'private-registry.company.com', type: 'private'})

// Create useful stored procedures for common queries

// Procedure to find all vulnerabilities for a specific host
CALL apoc.custom.asProcedure(
  'findHostVulnerabilities',
  'MATCH (h:Host {hostname: $hostname})-[:RUNS]->(i:Image)-[:HAS_VULNERABILITY]->(v:Vulnerability) 
   RETURN h.hostname, i.name, v.cve_id, v.severity, v.cvss_score, v.title 
   ORDER BY v.cvss_score DESC',
  'read',
  [['hostname', 'string']],
  [['hostname', 'string'], ['image_name', 'string'], ['cve_id', 'string'], ['severity', 'string'], ['cvss_score', 'float'], ['title', 'string']]
);

// Procedure to find all hosts affected by a specific vulnerability
CALL apoc.custom.asProcedure(
  'findVulnerabilityImpact',
  'MATCH (v:Vulnerability {cve_id: $cve_id})<-[:HAS_VULNERABILITY]-(i:Image)<-[:RUNS]-(h:Host)
   RETURN v.cve_id, v.severity, h.hostname, h.environment, h.datacenter, i.name
   ORDER BY h.environment, h.hostname',
  'read',
  [['cve_id', 'string']],
  [['cve_id', 'string'], ['severity', 'string'], ['hostname', 'string'], ['environment', 'string'], ['datacenter', 'string'], ['image_name', 'string']]
);

// Procedure to get security summary for an environment
CALL apoc.custom.asProcedure(
  'getEnvironmentSecuritySummary',
  'MATCH (h:Host {environment: $environment})-[:RUNS]->(i:Image)-[:HAS_VULNERABILITY]->(v:Vulnerability)
   WITH v.severity as severity, count(*) as count
   RETURN severity, count
   ORDER BY 
     CASE severity 
       WHEN "CRITICAL" THEN 1 
       WHEN "HIGH" THEN 2 
       WHEN "MEDIUM" THEN 3 
       WHEN "LOW" THEN 4 
     END',
  'read',
  [['environment', 'string']],
  [['severity', 'string'], ['count', 'long']]
);

// Create indexes for relationship properties
CREATE INDEX rel_runs_status_idx IF NOT EXISTS FOR ()-[r:RUNS]-() ON (r.status);
CREATE INDEX rel_has_vuln_detected_idx IF NOT EXISTS FOR ()-[r:HAS_VULNERABILITY]-() ON (r.detected_at);
CREATE INDEX rel_has_vuln_scanner_idx IF NOT EXISTS FOR ()-[r:HAS_VULNERABILITY]-() ON (r.scanner);
