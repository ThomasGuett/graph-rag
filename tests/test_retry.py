"""Tests for transient OpenAI retry helper."""

import httpx
import pytest
from openai import APITimeoutError

from graphrag.adapters.retry import with_retries
from graphrag.exceptions import UpstreamModelError


def _timeout() -> APITimeoutError:
    return APITimeoutError(httpx.Request("GET", "http://example.test"))


@pytest.mark.asyncio
async def test_with_retries_succeeds_after_transient_failure():
    calls = {"n": 0}

    async def _op():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _timeout()
        return "ok"

    result = await with_retries(
        _op, max_retries=2, label="test", detail_prefix="failed"
    )
    assert result == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_with_retries_raises_upstream_after_exhaustion():
    async def _op():
        raise _timeout()

    with pytest.raises(UpstreamModelError, match="failed"):
        await with_retries(_op, max_retries=1, label="test", detail_prefix="failed")
