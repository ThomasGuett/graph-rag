from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env next to docker-compose.yml (repo root), not CWD
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    postgres_db: str = "graphrag"
    postgres_user: str = "graphrag"
    postgres_password: str = "graphrag"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str | None = None

    # OpenAI-compatible API (native default; Docker Compose sets host.docker.internal via .env)
    openai_api_base: str = "http://localhost:11434/v1"
    openai_api_key: str = "sk-local"
    openai_timeout_seconds: float = Field(default=60.0, gt=0)
    openai_max_retries: int = Field(default=2, ge=0, le=8)

    # Models
    llm_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 2000
    embedding_batch_size: int = Field(default=64, ge=1, le=2000)

    # Retrieval
    retrieval_top_k: int = Field(default=8, ge=1, le=100)
    expand_hops: int = Field(default=1, ge=0, le=5)
    context_token_budget: int = Field(default=4000, ge=100)
    hnsw_ef_search: int = Field(default=64, ge=1)
    global_map_top_k: int = Field(default=6, ge=1, le=50)
    local_entity_top_k: int = Field(default=5, ge=1, le=50)
    auto_entity_score_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    global_map_concurrency: int = Field(default=4, ge=1, le=32)

    # Indexing / ingest
    chunk_size: int = Field(default=1200, ge=100, le=16000)
    chunk_overlap: int = Field(default=200, ge=0, le=4000)
    extraction_concurrency: int = Field(default=4, ge=1, le=32)
    community_min_size: int = Field(default=3, ge=2, le=1000)

    @field_validator("embedding_dim")
    @classmethod
    def _dim_must_be_2000(cls, v: int) -> int:
        if v != 2000:
            raise ValueError("EMBEDDING_DIM must be 2000 for this schema")
        return v

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
