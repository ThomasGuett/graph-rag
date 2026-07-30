from contextlib import asynccontextmanager

from fastapi import FastAPI
from pgvector.asyncpg import register_vector
from sqlalchemy import event

from graphrag.adapters.db.session import engine
from graphrag.api.routes import api_router
from graphrag.config import get_settings


@event.listens_for(engine.sync_engine, "connect")
def _register_vector(dbapi_connection, _connection_record) -> None:
    dbapi_connection.run_async(register_vector)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    if settings.embedding_dim != 2048:
        raise RuntimeError("EMBEDDING_DIM must be 2048")
    yield
    await engine.dispose()


app = FastAPI(
    title="GraphRAG API",
    version="0.1.0",
    description="Multi-purpose GraphRAG over Postgres + pgvector",
    lifespan=lifespan,
)
app.include_router(api_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "graphrag", "docs": "/docs", "health": "/api/v1/health"}
