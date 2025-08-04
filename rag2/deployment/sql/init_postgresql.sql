-- RAG2 PostgreSQL initialization script
-- Create database and enable pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table for storing document metadata and vectors
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(255),
    document_type VARCHAR(50),
    file_path VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create document_chunks table for storing text chunks and their embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768), -- Default to bge-base dimensions, will be updated for production
    token_count INTEGER,
    chunk_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);

-- Create vector similarity search index
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Create text search index
CREATE INDEX IF NOT EXISTS document_chunks_content_idx 
ON document_chunks USING gin(to_tsvector('english', content));

-- Create index on document_id for faster joins
CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx 
ON document_chunks(document_id);

-- Create hosts table for AIOps data
CREATE TABLE IF NOT EXISTS hosts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hostname VARCHAR(255) NOT NULL UNIQUE,
    ip_address INET NOT NULL,
    os VARCHAR(100),
    cpu_cores INTEGER,
    memory_gb INTEGER,
    environment VARCHAR(50),
    datacenter VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create images table for container images
CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    base_image VARCHAR(100),
    version VARCHAR(50),
    size_mb INTEGER,
    architecture VARCHAR(20),
    registry VARCHAR(100),
    tags TEXT[],
    scan_status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(name, version)
);

-- Create vulnerabilities table
CREATE TABLE IF NOT EXISTS vulnerabilities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cve_id VARCHAR(20) NOT NULL UNIQUE,
    title VARCHAR(500),
    description TEXT,
    severity VARCHAR(20),
    cvss_score DECIMAL(3,1),
    category VARCHAR(100),
    published_date TIMESTAMP WITH TIME ZONE,
    modified_date TIMESTAMP WITH TIME ZONE,
    affected_packages TEXT[],
    fix_available BOOLEAN DEFAULT FALSE,
    fix_version VARCHAR(50),
    reference_urls TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create host_images relationship table
CREATE TABLE IF NOT EXISTS host_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    host_id UUID REFERENCES hosts(id) ON DELETE CASCADE,
    image_id UUID REFERENCES images(id) ON DELETE CASCADE,
    container_name VARCHAR(255),
    container_status VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    properties JSONB DEFAULT '{}'::jsonb,
    UNIQUE(host_id, image_id, container_name)
);

-- Create image_vulnerabilities relationship table
CREATE TABLE IF NOT EXISTS image_vulnerabilities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID REFERENCES images(id) ON DELETE CASCADE,
    vulnerability_id UUID REFERENCES vulnerabilities(id) ON DELETE CASCADE,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    scanner VARCHAR(50),
    confidence VARCHAR(20),
    properties JSONB DEFAULT '{}'::jsonb,
    UNIQUE(image_id, vulnerability_id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS hosts_hostname_idx ON hosts(hostname);
CREATE INDEX IF NOT EXISTS hosts_environment_idx ON hosts(environment);
CREATE INDEX IF NOT EXISTS images_name_idx ON images(name);
CREATE INDEX IF NOT EXISTS images_scan_status_idx ON images(scan_status);
CREATE INDEX IF NOT EXISTS vulnerabilities_cve_id_idx ON vulnerabilities(cve_id);
CREATE INDEX IF NOT EXISTS vulnerabilities_severity_idx ON vulnerabilities(severity);
CREATE INDEX IF NOT EXISTS host_images_host_id_idx ON host_images(host_id);
CREATE INDEX IF NOT EXISTS host_images_image_id_idx ON host_images(image_id);
CREATE INDEX IF NOT EXISTS image_vulnerabilities_image_id_idx ON image_vulnerabilities(image_id);
CREATE INDEX IF NOT EXISTS image_vulnerabilities_vulnerability_id_idx ON image_vulnerabilities(vulnerability_id);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_hosts_updated_at BEFORE UPDATE ON hosts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_images_updated_at BEFORE UPDATE ON images
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vulnerabilities_updated_at BEFORE UPDATE ON vulnerabilities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rag2_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rag2_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO rag2_user;
