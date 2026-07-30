"""LLM entity / relationship extraction from text chunks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from graphrag.adapters.llm.base import LLMClient
from graphrag.exceptions import UpstreamModelError, ValidationAppError

_EXTRACT_SYSTEM = """You extract knowledge-graph entities and relationships from text.
Return ONLY valid JSON with this shape:
{
  "entities": [{"name": string, "type": string, "description": string}],
  "relationships": [{"source": string, "target": string, "type": string, "description": string}]
}
Rules:
- Use concise entity names; types are lowercase snake_case (person, org, place, concept, ...).
- Relationship type is lowercase snake_case (leads, part_of, mentions, related_to, ...).
- source/target must match entity names from entities when possible.
- Prefer fewer high-quality extractions over noisy ones.
- No markdown fences, no commentary."""


@dataclass(slots=True)
class ExtractedEntity:
    name: str
    type: str
    description: str = ""


@dataclass(slots=True)
class ExtractedRelationship:
    source: str
    target: str
    type: str
    description: str = ""


@dataclass(slots=True)
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)


class ExtractionService:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def extract_from_text(self, text: str) -> ExtractionResult:
        user = f"Extract entities and relationships from:\n\n{text}"
        raw = await self._llm.complete(system=_EXTRACT_SYSTEM, user=user, temperature=0.1)
        return parse_extraction_json(raw)


def parse_extraction_json(raw: str) -> ExtractionResult:
    payload = _load_json_object(raw)
    entities_raw = payload.get("entities") or []
    rels_raw = payload.get("relationships") or []
    if not isinstance(entities_raw, list) or not isinstance(rels_raw, list):
        raise ValidationAppError("extraction JSON must contain entities and relationships lists")

    entities: list[ExtractedEntity] = []
    for item in entities_raw:
        if not isinstance(item, dict):
            continue
        name = _clean_name(item.get("name"))
        etype = _clean_type(item.get("type") or "concept")
        if not name:
            continue
        entities.append(
            ExtractedEntity(
                name=name,
                type=etype,
                description=str(item.get("description") or "").strip(),
            )
        )

    relationships: list[ExtractedRelationship] = []
    for item in rels_raw:
        if not isinstance(item, dict):
            continue
        source = _clean_name(item.get("source"))
        target = _clean_name(item.get("target"))
        rtype = _clean_type(item.get("type") or "related_to")
        if not source or not target or source.lower() == target.lower():
            continue
        relationships.append(
            ExtractedRelationship(
                source=source,
                target=target,
                type=rtype,
                description=str(item.get("description") or "").strip(),
            )
        )
    return ExtractionResult(entities=entities, relationships=relationships)


def _load_json_object(raw: str) -> dict:
    text = raw.strip()
    if not text:
        raise UpstreamModelError("empty extraction response")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to salvage first {...} block.
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValidationAppError("extraction response is not valid JSON") from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValidationAppError("extraction response is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ValidationAppError("extraction JSON root must be an object")
    return data


def _clean_name(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _clean_type(value: object) -> str:
    text = str(value or "concept").strip().lower().replace(" ", "_").replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text or "concept"
