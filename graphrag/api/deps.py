from collections.abc import AsyncGenerator
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from graphrag.adapters.db.session import get_session
from graphrag.adapters.embeddings.openai_compatible import OpenAICompatibleEmbeddings
from graphrag.adapters.llm.openai_compatible import OpenAICompatibleLLM
from graphrag.config import Settings, get_settings
from graphrag.services.chunk_service import ChunkService
from graphrag.services.community_service import CommunityService
from graphrag.services.edge_service import EdgeService
from graphrag.services.embedding_service import EmbeddingService
from graphrag.services.ingest_service import IngestService
from graphrag.services.node_service import NodeService
from graphrag.services.qa_service import QAService
from graphrag.services.retrieval_service import RetrievalService


def settings_dep() -> Settings:
    return get_settings()


@lru_cache
def get_embedding_client() -> OpenAICompatibleEmbeddings:
    return OpenAICompatibleEmbeddings(get_settings())


@lru_cache
def get_llm_client() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(get_settings())


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_embedding_client(), get_settings())


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session():
        yield session


def get_node_service(session: AsyncSession = Depends(get_db_session)) -> NodeService:
    return NodeService(session)


def get_edge_service(session: AsyncSession = Depends(get_db_session)) -> EdgeService:
    return EdgeService(session)


def get_chunk_service(
    session: AsyncSession = Depends(get_db_session),
) -> ChunkService:
    return ChunkService(session, get_embedding_service())


def get_retrieval_service(
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(settings_dep),
) -> RetrievalService:
    return RetrievalService(session, get_embedding_service(), settings)


def get_qa_service(
    retrieval: RetrievalService = Depends(get_retrieval_service),
    settings: Settings = Depends(settings_dep),
) -> QAService:
    return QAService(retrieval, get_llm_client(), settings)


def get_ingest_service(
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(settings_dep),
) -> IngestService:
    return IngestService(session, get_embedding_service(), get_llm_client(), settings)


def get_community_service(
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(settings_dep),
) -> CommunityService:
    return CommunityService(session, get_llm_client(), get_embedding_service(), settings)
