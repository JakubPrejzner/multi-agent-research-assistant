"""State schema for the LangGraph research workflow."""

from __future__ import annotations

from typing import Any, TypedDict

from src.models.domain import (
    AnalysisResult,
    CritiqueResult,
    Report,
    ResearchPlan,
    SearchResult,
)


class ResearchState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    # Input
    query: str
    depth: str
    model: str

    # Workflow state
    current_phase: str
    status: str

    # Agent outputs
    research_plan: ResearchPlan | None
    plan_approved: bool
    search_results: list[SearchResult]
    rag_context: str
    analysis: AnalysisResult | None
    draft_report: Report | None
    critique: CritiqueResult | None
    final_report: Report | None

    # Control flow
    revision_count: int
    max_revisions: int

    # Metadata
    errors: list[str]
    phase_timings: dict[str, float]
    token_usage: dict[str, Any]
    task_id: str

    # Callbacks
    status_callback: Any
