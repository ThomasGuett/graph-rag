from fastapi import APIRouter

from graphrag.api.routes import chunks, communities, documents, edges, health, nodes, qa, search

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(nodes.router)
api_router.include_router(edges.router)
api_router.include_router(chunks.router)
api_router.include_router(documents.router)
api_router.include_router(communities.router)
api_router.include_router(search.router)
api_router.include_router(qa.router)
api_router.include_router(health.router)
