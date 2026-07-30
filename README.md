# graph-rag

Multi-purpose **GraphRAG** system: PostgreSQL + pgvector (embeddings dim **2000**), Python/FastAPI backend, CRUD REST API, hybrid vector + graph retrieval for LLM Q&A.

LLMs and embeddings are accessed through a configurable **OpenAI-compatible** HTTP API (local or cloud).

## Quick start

```bash
cp .env.example .env
# Edit .env: OPENAI_API_BASE, OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL, Postgres creds

docker compose up --build -d
```

- API: http://localhost:8080/docs  
- Health: http://localhost:8080/api/v1/health  
- pgAdmin: http://localhost:8091  
- nginx: http://localhost:8000  

## Configuration (`.env` next to `docker-compose.yml`)

| Variable | Purpose |
|----------|---------|
| `POSTGRES_*` | Database name/user/password/host/port |
| `OPENAI_API_BASE` | OpenAI-compatible base URL (e.g. `http://host.docker.internal:11434/v1` or cloud) |
| `OPENAI_API_KEY` | API key (any non-empty value for many local servers) |
| `OPENAI_TIMEOUT_SECONDS` | HTTP timeout for LLM/embedding calls |
| `LLM_MODEL` | Chat model id |
| `EMBEDDING_MODEL` | Embedding model id (vectors longer than 2000 are truncated + L2-renormalized) |
| `EMBEDDING_DIM` | Must be `2000` |
| `EMBEDDING_BATCH_SIZE` | Max texts per embedding request |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Document chunking windows |
| `EXTRACTION_CONCURRENCY` | Parallel LLM extractions per ingest job |
| `COMMUNITY_MIN_SIZE` | Min entities per community (default 3) |
| `RETRIEVAL_TOP_K` / `EXPAND_HOPS` / `CONTEXT_TOKEN_BUDGET` | Retrieval defaults |

## API overview

Base path: `/api/v1`

- CRUD: `/nodes`, `/edges`, `/chunks` (+ `/chunks/batch`)
- Ingest: `POST /documents` (chunk → extract → resolve → communities), `GET /documents`, `POST /documents/{id}/reindex`
- Communities: `GET /communities`, `GET /communities/{id}`, `POST /communities/rebuild`
- `POST /search` — hybrid retrieval (no LLM)
- `POST /qa` — retrieve → context pack → LLM answer
- `GET /health`

Existing volumes: apply `migrations/001_indexing_pipeline.sql` once if you upgraded from a pre-ingest schema.

## Local development (without Docker for the API)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Start Postgres via compose, then:
export POSTGRES_HOST=localhost
uvicorn graphrag.main:app --reload --port 8080
```

```bash
pytest
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/API.md](docs/API.md), and [docs/schema.sql](docs/schema.sql).
