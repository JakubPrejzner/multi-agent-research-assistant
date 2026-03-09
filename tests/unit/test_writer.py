"""Tests for WriterAgent."""

from __future__ import annotations

import json

import pytest

from src.agents.writer import WriterAgent
from src.models.domain import (
    AnalysisResult,
    CritiqueResult,
    Report,
    RevisionSuggestion,
)
from tests.conftest import create_mock_llm


class TestWriterAgent:
    @pytest.fixture()
    def writer_response(self) -> str:
        return json.dumps(
            {
                "title": "Quantum Computing: State of the Art",
                "executive_summary": "Quantum computing represents a paradigm shift...",
                "key_findings": [
                    "Quantum computers use qubits for parallel computation [Source 1]",
                    "Current quantum advantage is limited to specific problems [Source 2]",
                ],
                "contradictions": ["Timeline predictions vary widely"],
                "open_questions": ["When will fault-tolerant QC be achieved?"],
                "sources": [
                    {"url": "https://example.com/q1", "title": "QC Basics", "reliability": "high"},
                    {"url": "https://example.com/q2", "title": "QC Apps", "reliability": "medium"},
                ],
                "markdown": "# Quantum Computing Report\n\nExecutive summary...",
            }
        )

    async def test_writes_report(
        self,
        writer_response: str,
        sample_analysis: AnalysisResult,
    ) -> None:
        mock_llm = create_mock_llm([writer_response])
        writer = WriterAgent(llm=mock_llm)
        report = await writer.run(analysis=sample_analysis, query="quantum computing")

        assert report.title == "Quantum Computing: State of the Art"
        assert len(report.key_findings) == 2
        assert len(report.sources) == 2
        assert report.markdown.startswith("# Quantum Computing Report")

    async def test_revises_report(self, writer_response: str) -> None:
        original = Report(
            title="Draft",
            executive_summary="Draft summary",
            key_findings=["Finding 1"],
            markdown="# Draft",
        )
        critique = CritiqueResult(
            overall_score=0.5,
            weaknesses=["Lacks detail"],
            suggestions=[
                RevisionSuggestion(
                    section="Key Findings",
                    issue="Too vague",
                    suggestion="Add specific data points",
                )
            ],
        )

        mock_llm = create_mock_llm([writer_response])
        writer = WriterAgent(llm=mock_llm)
        revised = await writer.revise(original, critique)

        assert revised.title != ""
        assert len(revised.key_findings) > 0

    async def test_name_property(self) -> None:
        writer = WriterAgent(llm=create_mock_llm())
        assert writer.name == "writer"
