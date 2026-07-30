from fastapi import APIRouter, Depends

from graphrag.api.deps import get_qa_service
from graphrag.api.schemas import QARequest, QAResponse
from graphrag.services.qa_service import QAService

router = APIRouter(tags=["qa"])


@router.post("/qa", response_model=QAResponse)
async def qa(
    body: QARequest,
    service: QAService = Depends(get_qa_service),
) -> QAResponse:
    return await service.ask(body)
