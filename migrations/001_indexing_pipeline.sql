-- One-shot migration for existing Docker volumes (init.sql only runs on first boot).
-- Safe to re-run: uses IF NOT EXISTS.

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT NOT NULL,
    text        TEXT NOT NULL,
    source_uri  TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    node_id     UUID REFERENCES nodes(id) ON DELETE SET NULL,
    error       TEXT,
    props       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT documents_title_nonempty CHECK (length(trim(title)) > 0),
    CONSTRAINT documents_text_nonempty CHECK (length(trim(text)) > 0),
    CONSTRAINT documents_status_valid CHECK (
        status IN (
            'pending', 'chunking', 'extracting', 'resolving',
            'building_communities', 'ready', 'failed'
        )
    )
);

CREATE INDEX IF NOT EXISTS documents_status_idx ON documents (status);
CREATE INDEX IF NOT EXISTS documents_node_idx ON documents (node_id);
CREATE INDEX IF NOT EXISTS documents_props_gin ON documents USING gin (props);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL DEFAULT 'pending',
    status       TEXT NOT NULL DEFAULT 'pending',
    progress     JSONB NOT NULL DEFAULT '{}'::jsonb,
    error        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ingest_jobs_status_valid CHECK (
        status IN ('pending', 'running', 'completed', 'failed')
    )
);

CREATE INDEX IF NOT EXISTS ingest_jobs_document_idx ON ingest_jobs (document_id);
CREATE INDEX IF NOT EXISTS ingest_jobs_status_idx ON ingest_jobs (status);

CREATE TABLE IF NOT EXISTS communities (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label         TEXT NOT NULL,
    summary       TEXT,
    node_id       UUID REFERENCES nodes(id) ON DELETE SET NULL,
    member_count  INTEGER NOT NULL DEFAULT 0,
    props         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT communities_label_nonempty CHECK (length(trim(label)) > 0),
    CONSTRAINT communities_member_count_nonneg CHECK (member_count >= 0)
);

CREATE INDEX IF NOT EXISTS communities_node_idx ON communities (node_id);

CREATE TABLE IF NOT EXISTS community_members (
    community_id  UUID NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    node_id       UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    PRIMARY KEY (community_id, node_id)
);

CREATE INDEX IF NOT EXISTS community_members_node_idx ON community_members (node_id);

DROP TRIGGER IF EXISTS documents_set_updated_at ON documents;
CREATE TRIGGER documents_set_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS ingest_jobs_set_updated_at ON ingest_jobs;
CREATE TRIGGER ingest_jobs_set_updated_at
    BEFORE UPDATE ON ingest_jobs
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS communities_set_updated_at ON communities;
CREATE TRIGGER communities_set_updated_at
    BEFORE UPDATE ON communities
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
