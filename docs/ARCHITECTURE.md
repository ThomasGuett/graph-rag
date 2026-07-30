# GraphRAG System Architecture

Multi-purpose GraphRAG over **PostgreSQL + pgvector** (embedding dim **2000**), with a **Python** backend exposing a **CRUD REST API** and a retrieval/Q&A service that grounds an LLM in graph + vector context.

---

## 1. High-level system

```
+---------------------+        +----------------------+
|       User          |  Q&A   |        LLM           |
+----------+----------+ <----> +-----------+----------+
           ^                               ^
           |                               |
           v                               |
  +--------+---------+      context        |
  |  GraphRAG Layer  +---------------------+
  |  (API / Service) |
  +--------+---------+
           |
           v
+----------+----------------------------------------+
|                 Postgres                         |
|                                                  |
|  nodes          edges           chunks           |
|  -----          -----           ------           |
|  id, type,      src_id,         id, node_id,     |
|  name,          dst_id,         text, embedding  |
|  props          type            (pgvector 2000)  |
+--------------------------------------------------+
```

### Roles

| Layer | Responsibility |
|-------|----------------|
| **User** | CRUD on the knowledge graph; ask natural-language questions |
| **GraphRAG Layer** | REST API, ingestion helpers, hybrid retrieval (vector + graph walk), prompt assembly, LLM orchestration |
| **LLM** | Answer generation (and optional entity/relation extraction during ingest) |
| **Postgres** | Source of truth for graph topology + chunk embeddings |

The GraphRAG layer is the only component that talks to the database and to the LLM. Clients never hit pgvector or the model provider directly.

---

## 2. Design goals

1. **Multi-purpose graph** — nodes/edges are typed and schemaless via `props` JSONB; one store can serve docs, code, org charts, product catalogs, etc.
2. **Hybrid retrieval** — semantic search on chunks, then expand via edges to related nodes/chunks.
3. **Simple CRUD first** — predictable REST resources for nodes, edges, chunks before advanced agent features.
4. **Postgres-native** — no separate graph DB or vector DB; ACID, joins, and `pgvector` in one place.
5. **Pluggable LLM / embedder** — interfaces so OpenAI, local models, or other providers can be swapped.

---

## 3. Logical components (Python service)

```
graphrag/
├── api/                 # FastAPI (or similar) HTTP layer
│   ├── routes/
│   │   ├── nodes.py
│   │   ├── edges.py
│   │   ├── chunks.py
│   │   ├── search.py    # vector / hybrid search
│   │   └── qa.py        # GraphRAG Q&A endpoint
│   ├── deps.py          # DB session, settings, clients
│   └── schemas.py       # Pydantic request/response models
├── domain/              # Pure domain types & rules
│   ├── models.py
│   └── graph.py
├── services/
│   ├── node_service.py
│   ├── edge_service.py
│   ├── chunk_service.py
│   ├── embedding_service.py
│   ├── retrieval_service.py   # vector + graph expansion
│   └── qa_service.py          # context pack → LLM → answer
├── adapters/
│   ├── db/                    # SQLAlchemy / asyncpg
│   ├── llm/                   # chat completion client
│   └── embeddings/            # 2000-d embedding client
└── workers/                   # optional: async ingest / re-embed jobs
```

### Component boundaries

| Component | Owns | Does not own |
|-----------|------|--------------|
| **API routes** | HTTP contracts, validation, status codes | SQL, prompt templates |
| **Services** | Use cases (CRUD, retrieve, Q&A) | HTTP details, vendor SDKs |
| **Adapters** | Postgres, LLM, embedder I/O | Business orchestration |
| **Domain** | Node/edge/chunk semantics | Persistence format |

---

## 4. Data model (Postgres)

### Extensions

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- or use gen_random_uuid()
```

### Core tables

```sql
-- Knowledge-graph entities (documents, people, concepts, files, …)
CREATE TABLE nodes (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type        TEXT NOT NULL,              -- e.g. 'document', 'person', 'concept'
    name        TEXT NOT NULL,
    props       JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX nodes_type_idx ON nodes (type);
CREATE INDEX nodes_name_trgm_idx ON nodes USING gin (name gin_trgm_ops);  -- optional: pg_trgm
CREATE INDEX nodes_props_gin ON nodes USING gin (props);

-- Directed typed relations
CREATE TABLE edges (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    src_id      UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    dst_id      UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    type        TEXT NOT NULL,              -- e.g. 'mentions', 'part_of', 'depends_on'
    props       JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (src_id, dst_id, type)
);

CREATE INDEX edges_src_idx ON edges (src_id);
CREATE INDEX edges_dst_idx ON edges (dst_id);
CREATE INDEX edges_type_idx ON edges (type);

-- Text units attached to nodes, with fixed 2000-d embeddings
CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id     UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    embedding   vector(2000),               -- NULL until embedded
    props       JSONB NOT NULL DEFAULT '{}', -- e.g. { "section": "2.1", "ord": 3 }
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX chunks_node_idx ON chunks (node_id);

-- HNSW for ANN; cosine distance matches typical normalized embeddings
CREATE INDEX chunks_embedding_hnsw
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

### Schema notes

- **Embedding dim is fixed at 2000** — changing dim requires a migration (new column / rebuild index). Enforce in app config and DB check if desired.
- **`props` JSONB** keeps the graph multi-purpose without proliferating columns.
- **Chunks belong to nodes** — a document node can have many chunks; a concept node might have a single definitional chunk.
- **Edges are directed**; undirected relations can be modeled as two edges or queried symmetrically in the service layer.
- Indexing tables: `documents` (raw text + status), `ingest_jobs`, `communities`, `community_members`. Community detection is **flat connected components** (Leiden later).

### Entity relationship

```
nodes 1──* chunks
nodes 1──* edges (as src)
nodes 1──* edges (as dst)
```

---

## 5. REST API (CRUD + GraphRAG)

Base path: `/api/v1`. JSON in/out. Auth is pluggable (API key / JWT); omit for local/dev.

### Nodes

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/nodes` | Create node `{ type, name, props? }` |
| `GET` | `/nodes` | List / filter (`type`, `q` name search, pagination) |
| `GET` | `/nodes/{id}` | Get node |
| `PATCH` | `/nodes/{id}` | Update name / props / type |
| `DELETE` | `/nodes/{id}` | Delete node (cascades edges + chunks) |
| `GET` | `/nodes/{id}/neighbors` | Adjacent nodes via edges (`direction`, `type`, `depth`) |

### Edges

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/edges` | Create `{ src_id, dst_id, type, props? }` |
| `GET` | `/edges` | Filter by `src_id`, `dst_id`, `type` |
| `GET` | `/edges/{id}` | Get edge |
| `PATCH` | `/edges/{id}` | Update type / props |
| `DELETE` | `/edges/{id}` | Delete edge |

### Chunks

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chunks` | Create `{ node_id, text, props? }` → enqueue / sync embed |
| `POST` | `/chunks/batch` | Bulk create + embed |
| `GET` | `/chunks` | Filter by `node_id` |
| `GET` | `/chunks/{id}` | Get chunk (omit embedding by default) |
| `PATCH` | `/chunks/{id}` | Update text → re-embed |
| `DELETE` | `/chunks/{id}` | Delete chunk |

### Search & Q&A

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search` | Retrieval with `mode=auto\|local\|global\|hybrid` (no LLM) |
| `POST` | `/qa` | GraphRAG answer; global uses community map-reduce |

#### `POST /search` body

```json
{
  "query": "How does billing relate to accounts?",
  "mode": "auto",
  "top_k": 8,
  "node_types": ["document", "concept"],
  "expand_hops": 1,
  "edge_types": ["mentions", "part_of"]
}
```

Response: ranked chunks + linked nodes + optional expanded neighbor subgraph + `mode_used`.

#### `POST /qa` body

```json
{
  "question": "…",
  "mode": "auto",
  "top_k": 8,
  "expand_hops": 1,
  "include_sources": true
}
```

Response: `{ "answer": "…", "sources": […], "subgraph": { "nodes": [], "edges": [] }, "mode_used": "local" }`.

---

## 6. GraphRAG retrieval pipeline

`POST /search` and `POST /qa` accept `mode: auto|local|global|hybrid` (default `auto`).

### Hybrid (vector + hop expand)

```
Question
   │
   ▼
┌─────────────────┐
│  Embed query    │  same model family → vector(2000)
└────────┬────────┘
         ▼
┌─────────────────┐
│  ANN on chunks  │  ORDER BY embedding <=> :q LIMIT top_k
└────────┬────────┘
         ▼
┌─────────────────┐
│  Resolve nodes  │  chunk.node_id → nodes (+ props)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Graph expand   │  BFS on edges up to expand_hops
│                 │  collect neighbor nodes + their chunks
└────────┬────────┘
         ▼
┌─────────────────┐
│  Context pack   │  dedupe, rank, truncate to token budget
└────────┬────────┘
         ▼
┌─────────────────┐
│  LLM generate   │  system prompt + question + structured context
└────────┬────────┘
         ▼
      Answer + citations
```

### Local (entity-first)

1. Resolve seed entities (exact/alias on `normalized_name`/`aliases`, else ANN on `entity_description` chunks).
2. Expand along typed edges (excluding `mentions`).
3. Collect entity description chunks + document evidence via reverse `mentions` / `props.chunk_id`.
4. Pack + single LLM answer (same as hybrid QA).

### Global (community map-reduce)

1. ANN over community summary chunks (`node.type=community` / `kind=community_summary`).
2. **Map:** LLM partial answer per top community summary.
3. **Reduce:** LLM synthesizes the final answer from partials.

### Auto routing

- Strong entity match and non-thematic question → `local`
- Communities exist and (thematic keywords or no strong entities) → `global`
- Else → `hybrid`

### Ranking / packing

1. Seed chunks ranked by cosine similarity (hybrid) or entity seed score × hop decay (local).
2. Expanded chunks get a decayed score (e.g. `score * 0.5^hop`).
3. Cap total characters/tokens.
4. Serialize context as labeled blocks: `[node:Name|type|chunk:id] chunk text …`.

### Why hybrid matters

Pure vector search finds similar text; the **graph** surfaces related entities that may not share lexical/semantic overlap (e.g. `Invoice --billed_to--> Account` when the question only mentions “billing”).

---

## 7. Ingestion patterns (multi-purpose)

The core model stays the same; adapters vary by domain.

| Domain | Nodes | Edges | Chunks |
|--------|-------|-------|--------|
| Documents | `document`, `section`, `concept` | `contains`, `mentions` | Section/paragraph text |
| Code | `repo`, `file`, `symbol` | `imports`, `calls`, `defines` | Docstrings / chunks of source |
| Org / CRM | `person`, `team`, `account` | `reports_to`, `owns` | Bio / notes |
| Products | `product`, `feature`, `ticket` | `has_feature`, `related_to` | Specs / descriptions |

**Ingest flow (`POST /documents`):**

1. Create `documents` row + `document`-typed node + `ingest_jobs` row.
2. Split text → create/embed chunks on the document node.
3. LLM-extract entities/relations per chunk.
4. Resolve entities by normalized `(type, name)`; write nodes, `mentions` edges, typed edges, description chunks.
5. Rebuild flat communities (connected components) + LLM summaries as embedded `community` chunks.

---

## 8. Adapter contracts

### Embedding

```python
class EmbeddingClient(Protocol):
    dim: int  # must be 2000

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...
```

Fail fast if returned vectors are not length 2000.

### LLM

```python
class LLMClient(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
    ) -> str:
        ...
```

Q&A service builds `user` from question + packed context; does not hardcode a vendor.

### Database

- Prefer **async** (`asyncpg` / SQLAlchemy async) for API concurrency.
- Use parameterized queries; never interpolate embeddings as raw strings without proper vector binding.
- Transactions around multi-table CRUD (e.g. create node + initial chunks).

---

## 9. Configuration

Environment / settings object (example):

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Postgres DSN |
| `EMBEDDING_DIM` | `2000` (asserted at startup) |
| `EMBEDDING_PROVIDER` / API keys | Embedder config |
| `LLM_PROVIDER` / model name / keys | Chat model |
| `RETRIEVAL_TOP_K` | Default `8` |
| `EXPAND_HOPS` | Default `1` |
| `CONTEXT_TOKEN_BUDGET` | Truncation limit |
| `HNSW_EF_SEARCH` | Query-time ANN recall/latency |

---

## 10. Request lifecycle examples

### Create + embed chunk

```
Client POST /chunks
  → ChunkService.create
  → DB insert (embedding NULL)
  → EmbeddingClient.embed([text])
  → DB update embedding
  → 201 { id, node_id, text, props }
```

### Q&A

```
Client POST /qa { question }
  → QAService.ask
  → EmbeddingClient.embed([question])
  → RetrievalService.hybrid_search
  → pack context
  → LLMClient.complete
  → 200 { answer, sources, subgraph? }
```

---

## 11. Non-functional considerations

| Concern | Approach |
|---------|----------|
| **Latency** | HNSW index; cache embedder for identical queries; keep expand_hops small (1–2) |
| **Scale** | Partition by `type` or tenant later; partial indexes; connection pool |
| **Consistency** | FK cascades; re-embed on text update; optional job table for async backfill |
| **Security** | Parameterized SQL; secrets in env; optional row-level tenant_id in props or column |
| **Observability** | Structured logs: query, top_k, hop counts, LLM token usage, ANN latency |
| **Testing** | Unit tests for packing/ranking; integration tests with Postgres+pgvector in Docker |

---

## 12. Suggested implementation phases

1. **Schema + CRUD** — migrations, nodes/edges/chunks REST, no LLM.
2. **Embeddings + `/search`** — embedder adapter, HNSW, vector search API.
3. **Graph expand** — neighbors + hybrid retrieval packing.
4. **`/qa`** — LLM adapter + citation response.
5. **Ingest pipeline** — `POST /documents`: chunk → embed → LLM extract → resolve → flat communities.
6. **Query modes** — `local` / `global` / `hybrid` / `auto` on `/search` and `/qa` (entity-first + community map-reduce).
7. **Hardening** — auth, tenants, metrics, eval harness for retrieval quality.

**Follow-ups:** Leiden hierarchical communities / multi-level community reports.

---

## 13. Out of scope for v1 (intentional)

- Separate graph database (Neo4j, etc.)
- Multi-modal embeddings
- Real-time collaborative editing of the graph UI
- Fine-tuned models (use off-the-shelf embed + chat)

These can sit on the same schema later without redesigning the three-table core.
