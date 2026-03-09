"""Tests for AnalystAgent."""

from __future__ import annotations

import json

import pytest

from src.agents.analyst import AnalystAgent
from src.models.domain import SearchResult, SourceReliability
from tests.conftest import create_mock_llm


class TestAnalystAgent:
    @pytest.fixture()
    def analyst_response(self) -> str:
        return json.dumps({
            "claims": [
                {
                    "statement": "Quantum computers use qubits",
                    "sources": ["https://example.com/q1"],
                    "confidence": "high",
                    "category": "fundamentals",
                },
                {
                    "statement": "Quantum advantage achieved in specific tasks",
                    "sources": ["https://example.com/q2"],
                    "confidence": "medium",
                    "category": "applications",
                },
            ],
            "contradictions": [
                {
                    "claim_a": "Quantum computers will replace classical by 2030",
                    "claim_b": "Classical computers will remain dominant for decades",
                    "source_a": "https://example.com/q1",
                    "source_b": "https://example.com/q2",
                    "explanation": "Different timeline predictions",
                }
            ],
            "gaps": ["Error correction progress"],
            "key_themes": ["qubits", "quantum advantage"],
        })

    async def test_analyzes_results(
        self,
        analyst_response: str,
        sample_search_results: list[SearchResult],
    ) -> None:
        mock_llm = create_mock_llm([analyst_response])
        analyst = AnalystAgent(llm=mock_llm)
        result = await analyst.run(search_results=sample_search_results)

        assert len(result.claims) == 2
        assert result.claims[0].confidence == "high"
        assert len(result.contradictions) == 1
        assert len(result.gaps) == 1
        assert len(result.key_themes) == 2

    async def test_empty_results(self) -> None:
        analyst = AnalystAgent(llm=create_mock_llm())
        result = await analyst.run(search_results=[])

        assert result.claims == []
        assert result.contradictions == []

    async def test_source_reliability_map(
        self, analyst_response: str
    ) -> None:
        results = [
            SearchResult(
                url="https://arxiv.org/paper1",
                title="Academic Paper",
                snippet="test",
                reliability=SourceReliability.HIGH,
            ),
            SearchResult(
                url="https://blog.example.com",
                title="Blog Post",
                snippet="test",
                reliability=SourceReliability.UNKNOWN,
            ),
        ]
        mock_llm = create_mock_llm([analyst_response])
        analyst = AnalystAgent(llm=mock_llm)
        result = await analyst.run(search_results=results)

        assert result.source_reliability["https://arxiv.org/paper1"] == SourceReliability.HIGH
        assert result.source_reliability["https://blog.example.com"] == SourceReliability.UNKNOWN

    async def test_name_property(self) -> None:
        analyst = AnalystAgent(llm=create_mock_llm())
        assert analyst.name == "analyst"
