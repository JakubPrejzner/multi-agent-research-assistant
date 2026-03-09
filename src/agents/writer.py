"""WriterAgent: synthesizes analysis into structured reports."""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import AgentBase, build_prompt_with_context
from src.llm import LLMClient
from src.models.domain import AnalysisResult, CritiqueResult, Report

logger = logging.getLogger(__name__)

WRITER_SYSTEM = build_prompt_with_context(
    system_role="research report writer who produces clear, well-sourced reports",
    instructions="""\
Write a structured research report based on the analysis provided.

Requirements:
- Executive summary (2-3 paragraphs, accessible to non-experts).
- Key findings as a numbered list with inline citations [Source N].
- Contradictions & open questions section.
- Sources section with reliability assessment.

Respond with ONLY valid JSON:
{
  "title": "<descriptive report title>",
  "executive_summary": "<2-3 paragraph summary>",
  "key_findings": ["<finding with [Source N] citation>"],
  "contradictions": ["<contradiction description>"],
  "open_questions": ["<unresolved question>"],
  "sources": [
    {"url": "<url>", "title": "<title>", "reliability": "high|medium|low|unknown"}
  ],
  "markdown": "<full report in markdown format>"
}""",
)

REVISION_SYSTEM = build_prompt_with_context(
    system_role="research report editor revising based on critique feedback",
    instructions="""\
Revise the report based on the critique feedback provided.
Address each suggestion while maintaining the report's structure and flow.

Respond with the same JSON schema as the original report.""",
)


class WriterAgent(AgentBase):
    """Synthesizes analysis results into structured reports."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        super().__init__(llm)

    @property
    def name(self) -> str:
        return "writer"

    def _format_analysis(self, analysis: AnalysisResult, query: str) -> str:
        """Format analysis for the LLM."""
        parts = [f"Research Query: {query}\n"]

        if analysis.claims:
            parts.append("## Claims")
            for c in analysis.claims:
                sources = ", ".join(c.sources)
                parts.append(
                    f"- [{c.confidence}] {c.statement} (Sources: {sources})"
                )

        if analysis.contradictions:
            parts.append("\n## Contradictions")
            for ct in analysis.contradictions:
                parts.append(f"- {ct.claim_a} vs {ct.claim_b}: {ct.explanation}")

        if analysis.gaps:
            parts.append("\n## Gaps")
            for g in analysis.gaps:
                parts.append(f"- {g}")

        if analysis.key_themes:
            parts.append(f"\n## Key Themes: {', '.join(analysis.key_themes)}")

        return "\n".join(parts)

    def _parse_report(self, data: dict[str, Any]) -> Report:
        """Parse LLM JSON response into a Report model."""
        sources = [
            {
                "url": str(s.get("url", "")),
                "title": str(s.get("title", "")),
                "reliability": str(s.get("reliability", "unknown")),
            }
            for s in data.get("sources", [])
        ]

        return Report(
            title=str(data.get("title", "Research Report")),
            executive_summary=str(data.get("executive_summary", "")),
            key_findings=[str(f) for f in data.get("key_findings", [])],
            contradictions=[str(c) for c in data.get("contradictions", [])],
            open_questions=[str(q) for q in data.get("open_questions", [])],
            sources=sources,
            markdown=str(data.get("markdown", "")),
        )

    async def run(self, **kwargs: Any) -> Report:
        """Write a research report from analysis results.

        Args:
            **kwargs: Must include 'analysis' (AnalysisResult) and 'query' (str).

        Returns:
            A structured Report.
        """
        analysis: AnalysisResult = kwargs["analysis"]
        query: str = kwargs["query"]

        formatted = self._format_analysis(analysis, query)
        data = await self._llm_json_call(WRITER_SYSTEM, formatted)
        report = self._parse_report(data)

        logger.info("Report written: %s (%d findings)", report.title, len(report.key_findings))
        return report

    async def revise(self, report: Report, critique: CritiqueResult) -> Report:
        """Revise a report based on critique feedback.

        Args:
            report: The current draft report.
            critique: Critique with revision suggestions.

        Returns:
            Revised Report.
        """
        critique_text = [
            f"Score: {critique.overall_score}",
            f"Weaknesses: {', '.join(critique.weaknesses)}",
        ]
        for s in critique.suggestions:
            critique_text.append(f"- [{s.section}] {s.issue}: {s.suggestion}")
        if critique.unsupported_claims:
            critique_text.append(f"Unsupported: {', '.join(critique.unsupported_claims)}")
        if critique.bias_flags:
            critique_text.append(f"Bias: {', '.join(critique.bias_flags)}")

        user_msg = (
            f"Current report:\n{report.markdown}\n\n"
            f"Critique feedback:\n" + "\n".join(critique_text)
        )

        data = await self._llm_json_call(REVISION_SYSTEM, user_msg)
        revised = self._parse_report(data)

        logger.info("Report revised: %s", revised.title)
        return revised
