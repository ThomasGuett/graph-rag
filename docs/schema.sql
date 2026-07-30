-- GraphRAG core schema
-- Requires: PostgreSQL 16+ recommended, pgvector extension
-- Embedding dimension: 2048 (fixed)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ---------------------------------------------------------------------------
-- nodes: typed entities in the knowledge graph
-- ---------------------------------------------------------------------------
CREATE TABLE nodes (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type        TEXT NOT NULL,
    name        TEXT NOT NULL,
    props       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT nodes_type_nonempty CHECK (length(trim(type)) > 0),
    CONSTRAINT nodes_name_nonempty CHECK (length(trim(name)) > 0)
);

CREATE INDEX nodes_type_idx ON nodes (type);
CREATE INDEX nodes_name_idx ON nodes (name);
CREATE INDEX nodes_props_gin ON nodes USING gin (props);

-- ---------------------------------------------------------------------------
-- edges: directed typed relations between nodes
-- ---------------------------------------------------------------------------
CREATE TABLE edges (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    src_id      UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    dst_id      UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    type        TEXT NOT NULL,
    props       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT edges_type_nonempty CHECK (length(trim(type)) > 0),
    CONSTRAINT edges_no_self_loop CHECK (src_id <> dst_id),
    CONSTRAINT edges_unique_triple UNIQUE (src_id, dst_id, type)
);

CREATE INDEX edges_src_idx ON edges (src_id);
CREATE INDEX edges_dst_idx ON edges (dst_id);
CREATE INDEX edges_type_idx ON edges (type);
CREATE INDEX edges_src_type_idx ON edges (src_id, type);
CREATE INDEX edges_dst_type_idx ON edges (dst_id, type);

-- ---------------------------------------------------------------------------
-- chunks: text units with pgvector embeddings (dim 2048)
-- ---------------------------------------------------------------------------
CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id     UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    embedding   vector(2048),
    props       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chunks_text_nonempty CHECK (length(trim(text)) > 0)
);

CREATE INDEX chunks_node_idx ON chunks (node_id);
CREATE INDEX chunks_props_gin ON chunks USING gin (props);

-- Approximate nearest neighbor (cosine). Tune m / ef_construction for corpus size.
CREATE INDEX chunks_embedding_hnsw
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Optional: updated_at trigger helper
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER nodes_set_updated_at
    BEFORE UPDATE ON nodes
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER chunks_set_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
