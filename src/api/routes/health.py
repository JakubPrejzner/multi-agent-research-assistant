"""Health check endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health and dependency status."""
    settings = get_settings()
    deps: dict[str, str] = {}

    # Redis check
    if settings.use_fakeredis:
        deps["redis"] = "ok (fakeredis)"
    else:
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(settings.redis_url)
            await r.ping()
            await r.aclose()
            deps["redis"] = "ok"
        except Exception as e:
            deps["redis"] = f"error: {e}"

    # ChromaDB check
    if settings.chroma_use_local:
        deps["chromadb"] = "ok (local)"
    else:
        try:
            import chromadb

            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
            client.heartbeat()
            deps["chromadb"] = "ok"
        except Exception as e:
            deps["chromadb"] = f"error: {e}"

    # LLM check
    deps["llm_model"] = settings.default_model
    if settings.openai_api_key:
        deps["openai_key"] = "configured"
    if settings.anthropic_api_key:
        deps["anthropic_key"] = "configured"

    overall = "healthy" if all(
        v.startswith("ok") or v == "configured" or "key" in k or k == "llm_model"
        for k, v in deps.items()
    ) else "degraded"

    return HealthResponse(
        status=overall,
        version="0.1.0",
        dependencies=deps,
    )
