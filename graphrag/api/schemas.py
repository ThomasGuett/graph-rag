from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class NodeCreate(BaseModel):
    type: str = Field(min_length=1)
    name: str = Field(min_length=1)
    props: dict = Field(default_factory=dict)


class NodeUpdate(BaseModel):
    type: str | None = Field(default=None, min_length=1)
    name: str | None = Field(default=None, min_length=1)
    props: dict | None = None


class NodeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    type: str
    name: str
    props: dict
    created_at: datetime
    updated_at: datetime


class EdgeCreate(BaseModel):
    src_id: UUID
    dst_id: UUID
    type: str = Field(min_length=1)
    props: dict = Field(default_factory=dict)


class EdgeUpdate(BaseModel):
    type: str | None = Field(default=None, min_length=1)
    props: dict | None = None


class EdgeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    src_id: UUID
    dst_id: UUID
    type: str
    props: dict
    created_at: datetime


class ChunkCreate(BaseModel):
    node_id: UUID
    text: str = Field(min_length=1)
    props: dict = Field(default_factory=dict)


class ChunkBatchCreate(BaseModel):
    chunks: list[ChunkCreate] = Field(min_length=1)


class ChunkUpdate(BaseModel):
    text: str | None = Field(default=None, min_length=1)
    props: dict | None = None


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    node_id: UUID
    text: str
    props: dict
    created_at: datetime
    updated_at: datetime
    embedding: list[float] | None = None


class SubgraphOut(BaseModel):
    nodes: list[NodeOut] = Field(default_factory=list)
    edges: list[EdgeOut] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)
    node_types: list[str] | None = None
    expand_hops: int | None = Field(default=None, ge=0, le=5)
    edge_types: list[str] | None = None


class SearchHit(BaseModel):
    chunk_id: UUID
    node_id: UUID
    node_name: str
    node_type: str
    text: str
    score: float
    hop: int = 0


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    subgraph: SubgraphOut


class QARequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)
    expand_hops: int | None = Field(default=None, ge=0, le=5)
    node_types: list[str] | None = None
    edge_types: list[str] | None = None
    include_sources: bool = True


class QASource(BaseModel):
    chunk_id: UUID
    node_id: UUID
    node_name: str
    excerpt: str


class QAResponse(BaseModel):
    answer: str
    sources: list[QASource] = Field(default_factory=list)
    subgraph: SubgraphOut | None = None


class HealthOut(BaseModel):
    status: str
    embedding_dim: int
    db: bool
    llm_model: str
    embedding_model: str


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_uri: str | None = None
    props: dict = Field(default_factory=dict)


class IngestJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    stage: str
    status: str
    progress: dict
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    text: str
    source_uri: str | None = None
    status: str
    node_id: UUID | None = None
    error: str | None = None
    props: dict
    created_at: datetime
    updated_at: datetime
    counts: dict[str, int] = Field(default_factory=dict)
    job: IngestJobOut | None = None


class DocumentCreateResponse(BaseModel):
    document: DocumentOut
    job: IngestJobOut


class CommunityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    label: str
    summary: str | None = None
    node_id: UUID | None = None
    member_count: int
    props: dict
    created_at: datetime
    updated_at: datetime


class CommunityDetailOut(CommunityOut):
    members: list[NodeOut] = Field(default_factory=list)


class CommunityRebuildOut(BaseModel):
    communities: list[CommunityOut]
