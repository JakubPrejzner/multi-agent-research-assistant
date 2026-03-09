"""Integration tests for the full research orchestration graph."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.orchestrator.callbacks import StatusEmitter
from src.orchestrator.graph import build_research_graph, run_research


def _make_plan_response() -> str:
    return json.dumps({
        "original_query": "test query",
        "subtasks": [
            {"id": "t1", "query": "test sub-query", "priority": 1, "rationale": "Core"},
        ],
        "reasoning": "Simple test",
        "estimated_complexity": "low",
    })


def _make_analysis_response() -> str:
    return json.dumps({
        "claims": [
            {
                "statement": "Test claim",
                "sources": ["https://example.com"],
                "confidence": "high",
                "category": "test",
            }
        ],
        "contradictions": [],
        "gaps": ["missing detail"],
        "key_themes": ["testing"],
    })


def _make_report_response() -> str:
    return json.dumps({
        "title": "Test Report",
        "executive_summary": "A summary of test findings.",
        "key_findings": ["Finding 1 [Source 1]"],
        "contradictions": [],
        "open_questions": ["Is this a good test?"],
        "sources": [{"url": "https://example.com", "title": "Test", "reliability": "high"}],
        "markdown": "# Test Report\n\nA summary.",
    })


def _make_critique_response(score: float = 0.85) -> str:
    return json.dumps({
        "overall_score": score,
        "strengths": ["Well-structured"],
        "weaknesses": ["Lacks depth"],
        "suggestions": [],
        "unsupported_claims": [],
        "bias_flags": [],
    })


class TestBuildGraph:
    def test_graph_compiles(self) -> None:
        graph = build_research_graph()
        compiled = graph.compile()
        assert compiled is not None


class TestRunResearch:
    @pytest.mark.integration
    async def test_full_flow_good_score(self) -> None:
        """Test complete research flow where critique passes on first attempt."""
        mock_llm = AsyncMock()
        mock_llm.complete_with_system = AsyncMock(
            side_effect=[
                _make_plan_response(),
                _make_analysis_response(),
                _make_report_response(),
                _make_critique_response(0.85),
            ]
        )
        mock_llm.usage = type("U", (), {
            "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "total_cost_usd": 0.0,
            "call_count": 0, "to_dict": lambda self: {},
        })()

        mock_search_results = [
            type("SR", (), {
                "url": "https://example.com",
                "title": "Test",
                "snippet": "Test snippet",
                "content": "Test content for the search result",
                "relevance_score": 0.9,
                "reliability": "unknown",
            })()
        ]

        with patch("src.orchestrator.graph.LLMClient", return_value=mock_llm), \
             patch("src.orchestrator.graph.SearchAgent") as mock_search_cls, \
             patch("src.orchestrator.graph.chunk_documents", return_value=[]), \
             patch("src.orchestrator.graph.VectorStore"), \
             patch("src.orchestrator.graph.HybridRetriever"):
            mock_searcher = AsyncMock()
            mock_searcher.run = AsyncMock(return_value=mock_search_results)
            mock_search_cls.return_value = mock_searcher

            emitter = StatusEmitter()
            result = await run_research(
                query="test query",
                depth="quick",
                task_id="test-001",
                emitter=emitter,
            )

        assert result["status"] == "completed"
        assert result["final_report"] is not None
        assert result["current_phase"] == "completed"
        assert len(emitter.history) > 0

    @pytest.mark.integration
    async def test_flow_with_revision(self) -> None:
        """Test flow where critique score triggers one revision."""
        mock_llm = AsyncMock()
        mock_llm.complete_with_system = AsyncMock(
            side_effect=[
                _make_plan_response(),
                _make_analysis_response(),
                _make_report_response(),
                _make_critique_response(0.5),   # triggers revision
                _make_report_response(),         # revised report
                _make_critique_response(0.85),   # passes
            ]
        )
        mock_llm.usage = type("U", (), {
            "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "total_cost_usd": 0.0,
            "call_count": 0, "to_dict": lambda self: {},
        })()

        with patch("src.orchestrator.graph.LLMClient", return_value=mock_llm), \
             patch("src.orchestrator.graph.SearchAgent") as mock_search_cls, \
             patch("src.orchestrator.graph.chunk_documents", return_value=[]), \
             patch("src.orchestrator.graph.VectorStore"), \
             patch("src.orchestrator.graph.HybridRetriever"):
            mock_searcher = AsyncMock()
            mock_searcher.run = AsyncMock(return_value=[])
            mock_search_cls.return_value = mock_searcher

            result = await run_research(
                query="revision test",
                depth="quick",
                task_id="test-002",
            )

        assert result["status"] == "completed"
        assert result.get("revision_count", 0) >= 1


class TestStatusEmitter:
    async def test_emits_events(self) -> None:
        emitter = StatusEmitter()
        queue = emitter.subscribe()

        await emitter.emit_phase_start("testing", "running tests")
        event = queue.get_nowait()

        assert event["type"] == "phase_start"
        assert event["data"]["phase"] == "testing"

    async def test_multiple_listeners(self) -> None:
        emitter = StatusEmitter()
        q1 = emitter.subscribe()
        q2 = emitter.subscribe()

        await emitter.emit("test_event", {"key": "value"})

        assert not q1.empty()
        assert not q2.empty()

    async def test_unsubscribe(self) -> None:
        emitter = StatusEmitter()
        queue = emitter.subscribe()
        emitter.unsubscribe(queue)

        await emitter.emit("test_event")
        assert queue.empty()
