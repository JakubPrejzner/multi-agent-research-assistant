"""Tests for CriticAgent."""

from __future__ import annotations

import json

import pytest

from src.agents.critic import CriticAgent
from src.models.domain import Report
from tests.conftest import create_mock_llm


class TestCriticAgent:
    @pytest.fixture()
    def critic_response_good(self) -> str:
        return json.dumps({
            "overall_score": 0.82,
            "strengths": ["Well-structured", "Good citations"],
            "weaknesses": ["Could include more recent data"],
            "suggestions": [
                {
                    "section": "Key Findings",
                    "issue": "Missing 2024 developments",
                    "suggestion": "Add recent breakthroughs",
                    "severity": "medium",
                }
            ],
            "unsupported_claims": [],
            "bias_flags": [],
        })

    @pytest.fixture()
    def critic_response_poor(self) -> str:
        return json.dumps({
            "overall_score": 0.45,
            "strengths": ["Covers basic concepts"],
            "weaknesses": ["Many unsupported claims", "Biased framing"],
            "suggestions": [
                {
                    "section": "Executive Summary",
                    "issue": "Overly optimistic tone",
                    "suggestion": "Balance with challenges",
                    "severity": "high",
                }
            ],
            "unsupported_claims": ["QC will revolutionize all industries by 2025"],
            "bias_flags": ["Tech-optimism bias"],
        })

    async def test_good_critique(
        self,
        critic_response_good: str,
        sample_report: Report,
    ) -> None:
        mock_llm = create_mock_llm([critic_response_good])
        critic = CriticAgent(llm=mock_llm)
        result = await critic.run(report=sample_report)

        assert result.overall_score == 0.82
        assert not result.needs_revision
        assert len(result.strengths) == 2
        assert len(result.suggestions) == 1

    async def test_poor_critique_triggers_revision(
        self,
        critic_response_poor: str,
        sample_report: Report,
    ) -> None:
        mock_llm = create_mock_llm([critic_response_poor])
        critic = CriticAgent(llm=mock_llm)
        result = await critic.run(report=sample_report)

        assert result.overall_score == 0.45
        assert result.needs_revision
        assert len(result.unsupported_claims) == 1
        assert len(result.bias_flags) == 1

    async def test_name_property(self) -> None:
        critic = CriticAgent(llm=create_mock_llm())
        assert critic.name == "critic"
