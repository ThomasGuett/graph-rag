# graph-rag

Multi-purpose **GraphRAG** system: PostgreSQL + pgvector (embeddings dim **2048**), Python backend, CRUD REST API, hybrid vector + graph retrieval for LLM Q&A.

## Architecture

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the full design:

- System diagram (User ↔ GraphRAG layer ↔ Postgres / LLM)
- Data model: `nodes`, `edges`, `chunks` (`vector(2048)`)
- Python service layout and adapter boundaries
- Hybrid retrieval + Q&A pipeline
- Implementation phases

Supporting artifacts:

| Doc | Contents |
|-----|----------|
| [docs/schema.sql](docs/schema.sql) | Postgres DDL (pgvector HNSW) |
| [docs/API.md](docs/API.md) | REST CRUD + `/search` + `/qa` sketch |

## Core data model (summary)

```
nodes (id, type, name, props)
edges (src_id, dst_id, type)
chunks (id, node_id, text, embedding vector(2048))
```

## Status

Architecture / planning. Implementation follows the phased plan in the architecture doc.
