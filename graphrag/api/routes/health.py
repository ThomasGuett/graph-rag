from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from graphrag.api.deps import get_db_session, settings_dep
from graphrag.api.schemas import HealthOut
from graphrag.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
async def health(
    settings: Settings = Depends(settings_dep),
    session: AsyncSession = Depends(get_db_session),
) -> HealthOut:
    db_ok = False
    try:
        await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return HealthOut(
        status="ok" if db_ok else "degraded",
        embedding_dim=settings.embedding_dim,
        db=db_ok,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )
