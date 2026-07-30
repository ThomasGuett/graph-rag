"""Retry helpers for OpenAI-compatible upstream calls."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAIError

from graphrag.exceptions import UpstreamModelError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_transient_openai_error(exc: BaseException) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 429
    return False


async def with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    label: str,
    detail_prefix: str,
) -> T:
    """Run ``operation`` with bounded retries on transient OpenAI errors."""
    attempts = max(1, int(max_retries) + 1)
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except OpenAIError as exc:
            last_exc = exc
            transient = is_transient_openai_error(exc)
            if not transient or attempt >= attempts:
                logger.warning(
                    "%s failed after %s/%s attempts: %s",
                    label,
                    attempt,
                    attempts,
                    exc,
                )
                raise UpstreamModelError(f"{detail_prefix}: {exc}") from exc
            delay = min(2.0, 0.25 * (2 ** (attempt - 1)))
            logger.info(
                "%s transient failure (%s/%s); retrying in %.2fs: %s",
                label,
                attempt,
                attempts,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise UpstreamModelError(f"{detail_prefix}: {last_exc}") from last_exc
