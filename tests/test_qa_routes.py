"""Route-level /qa behavior for generation vs retrieval failures."""

from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from graphrag.api.deps import get_qa_service
from graphrag.api.schemas import QAResponse, QASource, SubgraphOut
from graphrag.exceptions import UpstreamModelError
from graphrag.main import app
from graphrag.services.qa_service import GENERATION_FAILED_ANSWER


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_qa_generation_failure_returns_200_partial(client: TestClient):
    async def _ask(_request):
        return QAResponse(
            answer=GENERATION_FAILED_ANSWER,
            sources=[
                QASource(
                    chunk_id=uuid4(),
                    node_id=uuid4(),
                    node_name="Doc",
                    excerpt="evidence",
                    score=0.9,
                )
            ],
            subgraph=SubgraphOut(),
            mode_used="hybrid",
            confidence=0.9,
            generation_error="llm request failed: boom",
        )

    mock = AsyncMock()
    mock.ask = _ask
    app.dependency_overrides[get_qa_service] = lambda: mock

    response = client.post("/api/v1/qa", json={"question": "When billed?", "mode": "hybrid"})
    assert response.status_code == 200
    body = response.json()
    assert body["generation_error"]
    assert body["sources"]
    assert body["answer"] == GENERATION_FAILED_ANSWER


def test_qa_retrieval_upstream_error_still_502(client: TestClient):
    async def _ask(_request):
        raise UpstreamModelError("embedding request failed: down")

    mock = AsyncMock()
    mock.ask = _ask
    app.dependency_overrides[get_qa_service] = lambda: mock

    response = client.post("/api/v1/qa", json={"question": "When billed?", "mode": "hybrid"})
    assert response.status_code == 502
    assert "embedding request failed" in response.json()["detail"]
