from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from graphrag.adapters.embeddings.openai_compatible import OpenAICompatibleEmbeddings
from graphrag.adapters.llm.openai_compatible import OpenAICompatibleLLM
from graphrag.api.deps import get_embedding_client, get_llm_client, get_db_session, settings_dep
from graphrag.api.schemas import HealthOut
from graphrag.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
async def health(
    settings: Settings = Depends(settings_dep),
    session: AsyncSession = Depends(get_db_session),
    llm: OpenAICompatibleLLM = Depends(get_llm_client),
    embeddings: OpenAICompatibleEmbeddings = Depends(get_embedding_client),
) -> HealthOut:
    db_ok = False
    try:
        await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    embeddings_ok = False
    try:
        vectors = await embeddings.embed(["health-check"], max_retries=0)
        embeddings_ok = (
            len(vectors) == 1 and len(vectors[0]) == settings.embedding_dim
        )
    except Exception:
        embeddings_ok = False

    llm_ok = False
    try:
        await llm.complete(
            system="Reply with exactly: OK",
            user="ping",
            temperature=0,
            max_retries=0,
        )
        llm_ok = True
    except Exception:
        llm_ok = False

    status = "ok" if db_ok and embeddings_ok and llm_ok else "degraded"
    return HealthOut(
        status=status,
        embedding_dim=settings.embedding_dim,
        db=db_ok,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        llm_ok=llm_ok,
        embeddings_ok=embeddings_ok,
    )
