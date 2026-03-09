"""Tests for domain models."""

from __future__ import annotations

from src.models.domain import (
    AnalysisResult,
    Claim,
    CritiqueResult,
    ResearchDepth,
    ResearchPlan,
    ResearchResult,
    ResearchStatus,
    SearchResult,
    SubTask,
    TaskMetadata,
)


class TestResearchDepth:
    def test_max_subtasks(self) -> None:
        assert ResearchDepth.QUICK.max_subtasks == 3
        assert ResearchDepth.STANDARD.max_subtasks == 5
        assert ResearchDepth.DEEP.max_subtasks == 7

    def test_max_results_per_task(self) -> None:
        assert ResearchDepth.QUICK.max_results_per_task == 3
        assert ResearchDepth.STANDARD.max_results_per_task == 5
        assert ResearchDepth.DEEP.max_results_per_task == 10


class TestSubTask:
    def test_creation(self) -> None:
        task = SubTask(query="test query", priority=1)
        assert task.query == "test query"
        assert task.priority == 1
        assert len(task.id) == 8
        assert task.depends_on == []

    def test_frozen(self) -> None:
        task = SubTask(query="test", priority=1)
        try:
            task.query = "modified"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except Exception:
            pass


class TestResearchPlan:
    def test_task_count(self, sample_plan: ResearchPlan) -> None:
        assert sample_plan.task_count == 3

    def test_creation(self) -> None:
        plan = ResearchPlan(
            original_query="test",
            subtasks=[SubTask(query="sub", priority=1)],
        )
        assert plan.task_count == 1


class TestSearchResult:
    def test_defaults(self) -> None:
        result = SearchResult(url="https://example.com", title="Test", snippet="test")
        assert result.content == ""
        assert result.relevance_score == 0.0
        assert result.retrieved_at is not None


class TestClaim:
    def test_creation(self) -> None:
        claim = Claim(
            statement="Test claim",
            sources=["https://example.com"],
            confidence="high",
        )
        assert claim.statement == "Test claim"
        assert len(claim.sources) == 1


class TestAnalysisResult:
    def test_empty(self) -> None:
        result = AnalysisResult()
        assert result.claims == []
        assert result.contradictions == []
        assert result.gaps == []

    def test_with_data(self, sample_analysis: AnalysisResult) -> None:
        assert len(sample_analysis.claims) == 2
        assert len(sample_analysis.key_themes) == 3


class TestCritiqueResult:
    def test_needs_revision_true(self) -> None:
        critique = CritiqueResult(overall_score=0.5)
        assert critique.needs_revision is True

    def test_needs_revision_false(self) -> None:
        critique = CritiqueResult(overall_score=0.8)
        assert critique.needs_revision is False

    def test_boundary(self) -> None:
        critique = CritiqueResult(overall_score=0.7)
        assert critique.needs_revision is False


class TestTaskMetadata:
    def test_duration_none_when_incomplete(self) -> None:
        meta = TaskMetadata()
        assert meta.duration_seconds is None

    def test_mark_completed(self) -> None:
        meta = TaskMetadata()
        meta.mark_completed()
        assert meta.completed_at is not None
        assert meta.duration_seconds is not None
        assert meta.duration_seconds >= 0


class TestResearchResult:
    def test_defaults(self) -> None:
        result = ResearchResult(query="test")
        assert result.status == ResearchStatus.PENDING
        assert result.plan is None
        assert result.report is None

    def test_set_status(self) -> None:
        result = ResearchResult(query="test")
        result.set_status(ResearchStatus.PLANNING)
        assert result.status == ResearchStatus.PLANNING
