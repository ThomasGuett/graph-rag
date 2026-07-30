import pytest

from graphrag.config import Settings


def test_embedding_dim_must_be_2000():
    with pytest.raises(ValueError):
        Settings(embedding_dim=768, _env_file=None)


def test_sqlalchemy_url_from_parts(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    settings = Settings(
        postgres_user="u",
        postgres_password="p",
        postgres_host="db",
        postgres_port=5432,
        postgres_db="g",
        embedding_dim=2000,
        _env_file=None,
    )
    assert settings.sqlalchemy_url == "postgresql+asyncpg://u:p@db:5432/g"
