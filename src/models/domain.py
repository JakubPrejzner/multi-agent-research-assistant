"""Domain models for the multi-agent research assistant."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ResearchDepth(StrEnum):
    """Controls the breadth and depth of research."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"

    @property
    def max_subtasks(self) -> int:
        return {self.QUICK: 3, self.STANDARD: 5, self.DEEP: 7}[self]

    @property
    def max_results_per_task(self) -> int:
        return {self.QUICK: 3, self.STANDARD: 5, self.DEEP: 10}[self]


class ResearchStatus(StrEnum):
    """Lifecycle states for a research task."""

    PENDING = "pending"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    WRITING = "writing"
    CRITIQUING = "critiquing"
    REVISING = "revising"
    COMPLETED = "completed"
    FAILED = "failed"


class ReportFormat(StrEnum):
    """Supported output formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


class SourceReliability(StrEnum):
    """Assessed reliability of a source."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class SubTask(BaseModel):
    """A single research sub-task within a plan."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    query: str
    priority: int = Field(ge=1, le=10, description="1 = highest priority")
    depends_on: list[str] = Field(default_factory=list)
    rationale: str = ""

    model_config = {"frozen": True}


class ResearchPlan(BaseModel):
    """Decomposed research plan produced by the PlannerAgent."""

    original_query: str
    subtasks: list[SubTask]
    reasoning: str = ""
    estimated_complexity: str = "medium"

    @property
    def task_count(self) -> int:
        return len(self.subtasks)


class SearchResult(BaseModel):
    """A single search result with extracted content."""

    url: str
    title: str
    snippet: str
    content: str = ""
    source: str = "web"
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reliability: SourceReliability = SourceReliability.UNKNOWN


class Claim(BaseModel):
    """An extracted claim with source attribution and confidence."""

    statement: str
    sources: list[str] = Field(description="URLs supporting this claim")
    confidence: str = Field(description="high, medium, or low")
    category: str = ""


class Contradiction(BaseModel):
    """A detected contradiction between sources."""

    claim_a: str
    claim_b: str
    source_a: str
    source_b: str
    explanation: str = ""


class AnalysisResult(BaseModel):
    """Output of the AnalystAgent."""

    claims: list[Claim] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    key_themes: list[str] = Field(default_factory=list)
    source_reliability: dict[str, SourceReliability] = Field(default_factory=dict)


class Report(BaseModel):
    """Structured research report."""

    title: str
    executive_summary: str
    key_findings: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    sources: list[dict[str, str]] = Field(default_factory=list)
    markdown: str = ""
    raw_data: dict[str, Any] = Field(default_factory=dict)


class RevisionSuggestion(BaseModel):
    """A single actionable revision from the CriticAgent."""

    section: str
    issue: str
    suggestion: str
    severity: str = "medium"


class CritiqueResult(BaseModel):
    """Output of the CriticAgent."""

    overall_score: float = Field(ge=0.0, le=1.0)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggestions: list[RevisionSuggestion] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    bias_flags: list[str] = Field(default_factory=list)

    @property
    def needs_revision(self) -> bool:
        return self.overall_score < 0.7


class TaskMetadata(BaseModel):
    """Timing, cost, and usage metadata for a research run."""

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    model: str = ""
    phase_timings: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    def mark_completed(self) -> None:
        self.completed_at = datetime.now(UTC)

    @property
    def duration_seconds(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


class ResearchResult(BaseModel):
    """Top-level container for a complete research run."""

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    query: str
    depth: ResearchDepth = ResearchDepth.STANDARD
    status: ResearchStatus = ResearchStatus.PENDING
    plan: ResearchPlan | None = None
    search_results: list[SearchResult] = Field(default_factory=list)
    analysis: AnalysisResult | None = None
    report: Report | None = None
    critique: CritiqueResult | None = None
    metadata: TaskMetadata = Field(default_factory=TaskMetadata)

    def set_status(self, status: ResearchStatus) -> None:
        self.status = status
