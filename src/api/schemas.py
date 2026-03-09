"""Request/response schemas for the API layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.models.domain import ReportFormat, ResearchDepth, ResearchStatus


class ResearchRequest(BaseModel):
    """POST /research request body."""

    query: str = Field(min_length=3, max_length=1000, description="Research question")
    depth: ResearchDepth = Field(default=ResearchDepth.STANDARD)
    model: str = Field(default="", description="LLM model override (empty = default)")


class ResearchResponse(BaseModel):
    """Response for research task creation."""

    task_id: str
    status: ResearchStatus
    message: str


class ResearchStatusResponse(BaseModel):
    """GET /research/{task_id} response."""

    task_id: str
    status: ResearchStatus
    current_phase: str = ""
    progress: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    partial_results: dict[str, Any] = Field(default_factory=dict)


class ReportResponse(BaseModel):
    """GET /research/{task_id}/report response."""

    task_id: str
    format: ReportFormat
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str
    version: str
    dependencies: dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """RFC 7807 problem details."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str = ""
    instance: str = ""
