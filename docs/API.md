# REST API Sketch

Base URL: `/api/v1`  
Content-Type: `application/json`

This sketch mirrors the architecture in [ARCHITECTURE.md](./ARCHITECTURE.md). Status codes follow usual REST practice (`201` create, `204` delete, `404` missing, `409` unique conflict, `422` validation).

---

## Nodes

### `POST /nodes`

```json
{
  "type": "document",
  "name": "Billing Guide",
  "props": { "external_id": "doc-42", "source": "confluence" }
}
```

### `GET /nodes?type=document&q=billing&limit=50&offset=0`

### `GET /nodes/{id}`

### `PATCH /nodes/{id}`

```json
{ "name": "Billing Guide v2", "props": { "version": 2 } }
```

### `DELETE /nodes/{id}`

Cascades to related edges and chunks.

### `GET /nodes/{id}/neighbors?direction=both&type=mentions&depth=1`

Returns `{ "nodes": [...], "edges": [...] }` for the local subgraph.

---

## Edges

### `POST /edges`

```json
{
  "src_id": "…",
  "dst_id": "…",
  "type": "mentions",
  "props": { "weight": 1.0 }
}
```

### `GET /edges?src_id=…&dst_id=…&type=mentions`

### `GET /edges/{id}` · `PATCH /edges/{id}` · `DELETE /edges/{id}`

---

## Chunks

### `POST /chunks`

```json
{
  "node_id": "…",
  "text": "Accounts are billed monthly on the anniversary date.",
  "props": { "ord": 0, "section": "overview" }
}
```

Server embeds synchronously (or accepts `?async=true` later) and stores `vector(2000)`.

### `POST /chunks/batch`

```json
{
  "chunks": [
    { "node_id": "…", "text": "…", "props": { "ord": 0 } },
    { "node_id": "…", "text": "…", "props": { "ord": 1 } }
  ]
}
```

### `GET /chunks?node_id=…` · `GET /chunks/{id}` · `PATCH /chunks/{id}` · `DELETE /chunks/{id}`

By default, responses **omit** the raw embedding vector; use `?include_embedding=true` when needed.

---

## Documents (indexing pipeline)

### `POST /documents`

```json
{
  "title": "Billing Guide",
  "text": "Accounts are billed monthly...\n\nDr. Smith leads oncology at Boston General.",
  "source_uri": "https://example.com/billing",
  "props": { "external_id": "doc-42" }
}
```

Creates a `document` node, an `ingest_jobs` row, and runs the pipeline in-process:

1. chunk + embed  
2. LLM entity/relationship extraction  
3. entity resolution + graph write (`mentions` + typed edges)  
4. flat connected-component communities + summaries  

Response: `{ "document": {...}, "job": { "id", "stage", "status", "progress" } }`.

### `GET /documents` · `GET /documents/{id}`

Includes `status`, `error`, and `counts` (`chunks`, `mentions`).

### `POST /documents/{id}/reindex`

Clears document chunks + `mentions` edges, then re-runs the pipeline.

---

## Communities

Flat connected components over entity nodes (excludes `document` / `community` scaffolding). No Leiden hierarchy yet.

### `GET /communities`

### `GET /communities/{id}`

Includes `members` (nodes) and `summary`.

### `POST /communities/rebuild`

Full recompute of communities + LLM summaries + embedded summary chunks.

---

## Search (retrieval only)

### `POST /search`

`mode` selects the retrieval strategy (`auto` | `local` | `global` | `hybrid`, default `auto`).

```json
{
  "query": "When are accounts billed?",
  "mode": "auto",
  "top_k": 8,
  "node_types": ["document", "concept"],
  "expand_hops": 1,
  "edge_types": ["mentions", "part_of"]
}
```

```json
{
  "hits": [
    {
      "chunk_id": "…",
      "node_id": "…",
      "node_name": "Billing Guide",
      "node_type": "document",
      "text": "…",
      "score": 0.87,
      "hop": 0
    }
  ],
  "subgraph": {
    "nodes": [],
    "edges": []
  },
  "mode_used": "hybrid"
}
```

Modes:

- **`hybrid`** — ANN on chunk embeddings, then hop-expand on the graph (original path).
- **`local`** — entity-first: resolve query entities → expand typed edges → collect entity + document evidence via `mentions`.
- **`global`** — rank flat community summary chunks only (no map-reduce; that happens in `/qa`).
- **`auto`** — heuristic: strong entity match → local; thematic / no entities + communities exist → global; else hybrid.

---

## Q&A (GraphRAG)

### `POST /qa`

```json
{
  "question": "When are accounts billed?",
  "mode": "auto",
  "top_k": 8,
  "expand_hops": 1,
  "include_sources": true
}
```

```json
{
  "answer": "Accounts are billed monthly on the anniversary date.",
  "sources": [
    {
      "chunk_id": "…",
      "node_id": "…",
      "node_name": "Billing Guide",
      "excerpt": "Accounts are billed monthly…"
    }
  ],
  "subgraph": {
    "nodes": [],
    "edges": []
  },
  "mode_used": "local"
}
```

For `mode=global`, the service map-reduces over top community summaries (per-community partial answers → final synthesis).

---

## Health

### `GET /health`

```json
{ "status": "ok", "embedding_dim": 2000, "db": true }
```
