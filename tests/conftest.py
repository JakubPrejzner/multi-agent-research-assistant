"""Shared test fixtures."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.config import Settings, reset_settings
from src.models.domain import (
    AnalysisResult,
    Claim,
    CritiqueResult,
    Report,
    ResearchPlan,
    RevisionSuggestion,
    SearchResult,
    SubTask,
)


@pytest.fixture(autouse=True)
def _reset_settings() -> None:
    """Reset settings singleton between tests."""
    reset_settings()


@pytest.fixture()
def test_settings() -> Settings:
    """Settings configured for testing."""
    return Settings(
        use_fakeredis=True,
        chroma_use_local=True,
        chroma_persist_dir="./test_data/chroma",
        default_model="gpt-3.5-turbo",
        openai_api_key="test-key",
    )


@pytest.fixture()
def sample_plan() -> ResearchPlan:
    return ResearchPlan(
        original_query="What is quantum computing?",
        subtasks=[
            SubTask(
                id="t1",
                query="quantum computing fundamentals",
                priority=1,
                rationale="Core concepts",
            ),
            SubTask(
                id="t2",
                query="quantum computing applications",
                priority=2,
                depends_on=["t1"],
                rationale="Practical uses",
            ),
            SubTask(
                id="t3",
                query="quantum computing challenges",
                priority=3,
                rationale="Current limitations",
            ),
        ],
        reasoning="Decomposed into fundamentals, applications, and challenges.",
    )


@pytest.fixture()
def sample_search_results() -> list[SearchResult]:
    return [
        SearchResult(
            url="https://example.com/quantum-1",
            title="Quantum Computing Basics",
            snippet="Quantum computers use qubits...",
            content="Quantum computing leverages quantum mechanical phenomena like "
            "superposition and entanglement to process information.",
            relevance_score=0.9,
        ),
        SearchResult(
            url="https://example.com/quantum-2",
            title="Applications of Quantum Computing",
            snippet="Drug discovery, cryptography...",
            content="Quantum computing has potential applications in drug discovery, "
            "cryptography, optimization, and materials science.",
            relevance_score=0.85,
        ),
    ]


@pytest.fixture()
def sample_analysis() -> AnalysisResult:
    return AnalysisResult(
        claims=[
            Claim(
                statement="Quantum computers use qubits instead of classical bits",
                sources=["https://example.com/quantum-1"],
                confidence="high",
                category="fundamentals",
            ),
            Claim(
                statement="Quantum computing can break RSA encryption",
                sources=["https://example.com/quantum-2"],
                confidence="medium",
                category="applications",
            ),
        ],
        gaps=["Timeline for practical quantum advantage"],
        key_themes=["superposition", "entanglement", "quantum advantage"],
    )


@pytest.fixture()
def sample_report() -> Report:
    return Report(
        title="Quantum Computing: Current State and Future Prospects",
        executive_summary="Quantum computing represents a paradigm shift in computation...",
        key_findings=[
            "Quantum computers use qubits that can exist in superposition [1]",
            "Applications span cryptography, drug discovery, and optimization [2]",
        ],
        contradictions=["Timelines for quantum advantage vary significantly"],
        open_questions=["When will quantum computers surpass classical ones?"],
        sources=[
            {"url": "https://example.com/quantum-1", "title": "Quantum Computing Basics"},
            {"url": "https://example.com/quantum-2", "title": "Applications of QC"},
        ],
        markdown="# Quantum Computing Report\n\n...",
    )


@pytest.fixture()
def sample_critique() -> CritiqueResult:
    return CritiqueResult(
        overall_score=0.75,
        strengths=["Well-structured", "Good source diversity"],
        weaknesses=["Missing recent developments"],
        suggestions=[
            RevisionSuggestion(
                section="Key Findings",
                issue="Lacks specificity on timelines",
                suggestion="Add concrete timeline estimates from industry leaders",
            ),
        ],
    )


@pytest.fixture()
def mock_llm_response() -> str:
    """Default mock LLM response as JSON."""
    return json.dumps({
        "original_query": "test query",
        "subtasks": [
            {
                "id": "t1",
                "query": "sub-query 1",
                "priority": 1,
                "rationale": "First step",
            }
        ],
        "reasoning": "Simple decomposition",
    })


def create_mock_llm(responses: list[str] | None = None) -> AsyncMock:
    """Create a mock LLM client that returns predefined responses."""
    mock = AsyncMock()
    if responses:
        mock.complete.side_effect = responses
        mock.complete_with_system.side_effect = responses
    else:
        mock.complete.return_value = '{"result": "mock response"}'
        mock.complete_with_system.return_value = '{"result": "mock response"}'

    mock.usage = type("Usage", (), {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "call_count": 0,
        "to_dict": lambda self: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "call_count": 0,
        },
    })()
    return mock
