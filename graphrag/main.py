from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from graphrag.adapters.db.session import dispose_engine
from graphrag.api.routes import api_router
from graphrag.config import get_settings
from graphrag.exceptions import (
    ConflictError,
    EmbeddingDimensionError,
    GraphRAGError,
    NotFoundError,
    UpstreamModelError,
    ValidationAppError,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    if settings.embedding_dim != 2000:
        raise RuntimeError("EMBEDDING_DIM must be 2000")
    # pgvector 0.5 + SQLAlchemy uses text bind format; do not register asyncpg binary codecs.
    yield
    await dispose_engine()


app = FastAPI(
    title="GraphRAG API",
    version="0.1.0",
    description="Multi-purpose GraphRAG over Postgres + pgvector",
    lifespan=lifespan,
)
app.include_router(api_router)


@app.exception_handler(NotFoundError)
async def not_found_handler(_request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ConflictError)
async def conflict_handler(_request: Request, exc: ConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(ValidationAppError)
async def validation_handler(_request: Request, exc: ValidationAppError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(EmbeddingDimensionError)
async def embedding_dim_handler(_request: Request, exc: EmbeddingDimensionError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(UpstreamModelError)
async def upstream_handler(_request: Request, exc: UpstreamModelError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(GraphRAGError)
async def app_error_handler(_request: Request, exc: GraphRAGError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "graphrag", "docs": "/docs", "health": "/api/v1/health"}
