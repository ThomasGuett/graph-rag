from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from graphrag.api.deps import get_community_service
from graphrag.api.schemas import CommunityDetailOut, CommunityOut, CommunityRebuildOut, NodeOut
from graphrag.services.community_service import CommunityService

router = APIRouter(prefix="/communities", tags=["communities"])


@router.get("", response_model=list[CommunityOut])
async def list_communities(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: CommunityService = Depends(get_community_service),
) -> list[CommunityOut]:
    communities = await service.list(limit=limit, offset=offset)
    return [CommunityOut.model_validate(c) for c in communities]


@router.post("/rebuild", response_model=CommunityRebuildOut, status_code=status.HTTP_200_OK)
async def rebuild_communities(
    service: CommunityService = Depends(get_community_service),
) -> CommunityRebuildOut:
    communities = await service.rebuild()
    return CommunityRebuildOut(
        communities=[CommunityOut.model_validate(c) for c in communities]
    )


@router.get("/{community_id}", response_model=CommunityDetailOut)
async def get_community(
    community_id: UUID,
    service: CommunityService = Depends(get_community_service),
) -> CommunityDetailOut:
    community = await service.get(community_id)
    if not community:
        raise HTTPException(status_code=404, detail="community not found")
    members = await service.members(community_id)
    detail = CommunityDetailOut.model_validate(community)
    detail.members = [NodeOut.model_validate(m) for m in members]
    return detail
