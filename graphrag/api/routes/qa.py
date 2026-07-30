from fastapi import APIRouter, Depends, HTTPException

from graphrag.api.deps import get_qa_service
from graphrag.api.schemas import QARequest, QAResponse
from graphrag.services.qa_service import QAService

router = APIRouter(tags=["qa"])


@router.post("/qa", response_model=QAResponse)
async def qa(
    body: QARequest,
    service: QAService = Depends(get_qa_service),
) -> QAResponse:
    try:
        return await service.ask(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
