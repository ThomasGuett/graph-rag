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

Server embeds synchronously (or accepts `?async=true` later) and stores `vector(2048)`.

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

## Search (retrieval only)

### `POST /search`

```json
{
  "query": "When are accounts billed?",
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
  }
}
```

---

## Q&A (GraphRAG)

### `POST /qa`

```json
{
  "question": "When are accounts billed?",
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
  }
}
```

---

## Health

### `GET /health`

```json
{ "status": "ok", "embedding_dim": 2048, "db": true }
```
