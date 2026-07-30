"""Shared application errors."""


class GraphRAGError(Exception):
    """Base application error."""


class NotFoundError(GraphRAGError):
    pass


class ConflictError(GraphRAGError):
    pass


class ValidationAppError(GraphRAGError):
    pass


class UpstreamModelError(GraphRAGError):
    """LLM or embedding provider failure."""


class EmbeddingDimensionError(GraphRAGError):
    """Embedding vector length does not match configured dimension."""
