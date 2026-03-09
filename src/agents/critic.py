"""CriticAgent: reviews reports for quality, bias, and unsupported claims."""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import AgentBase, build_prompt_with_context
from src.llm import LLMClient
from src.models.domain import CritiqueResult, Report, RevisionSuggestion

logger = logging.getLogger(__name__)

CRITIC_SYSTEM = build_prompt_with_context(
    system_role="critical research reviewer who evaluates report quality and accuracy",
    instructions="""\
Review the research report and evaluate it on these criteria:
1. Are all claims supported by cited sources?
2. Is the analysis logically consistent?
3. Are there missing perspectives or viewpoints?
4. Is there detectable bias in framing or source selection?
5. Is the report well-structured and clear?

Scoring guide:
- 0.9-1.0: Excellent, publishable quality
- 0.7-0.89: Good, minor improvements needed
- 0.5-0.69: Adequate, needs revision
- Below 0.5: Poor, major revision required

Respond with ONLY valid JSON:
{
  "overall_score": <0.0-1.0>,
  "strengths": ["<strength1>"],
  "weaknesses": ["<weakness1>"],
  "suggestions": [
    {
      "section": "<section name>",
      "issue": "<what's wrong>",
      "suggestion": "<how to fix>",
      "severity": "high|medium|low"
    }
  ],
  "unsupported_claims": ["<claim without adequate sourcing>"],
  "bias_flags": ["<detected bias>"]
}""",
)


class CriticAgent(AgentBase):
    """Reviews research reports for quality and accuracy."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        super().__init__(llm)

    @property
    def name(self) -> str:
        return "critic"

    async def run(self, **kwargs: Any) -> CritiqueResult:
        """Critique a research report.

        Args:
            **kwargs: Must include 'report' (Report).

        Returns:
            A CritiqueResult with score and suggestions.
        """
        report: Report = kwargs["report"]

        findings = "\n".join(f"- {f}" for f in report.key_findings)
        contras = "\n".join(f"- {c}" for c in report.contradictions)
        questions = "\n".join(f"- {q}" for q in report.open_questions)
        sources = "\n".join(
            f"- {s.get('title', 'Unknown')} ({s.get('url', '')})" for s in report.sources
        )
        user_msg = (
            f"Title: {report.title}\n\n"
            f"Executive Summary:\n{report.executive_summary}\n\n"
            f"Key Findings:\n{findings}\n\n"
            f"Contradictions:\n{contras}\n\n"
            f"Open Questions:\n{questions}\n\n"
            f"Sources:\n{sources}"
        )

        data = await self._llm_json_call(CRITIC_SYSTEM, user_msg)

        suggestions = [
            RevisionSuggestion(
                section=str(s.get("section", "")),
                issue=str(s.get("issue", "")),
                suggestion=str(s.get("suggestion", "")),
                severity=str(s.get("severity", "medium")),
            )
            for s in data.get("suggestions", [])
        ]

        result = CritiqueResult(
            overall_score=float(data.get("overall_score", 0.5)),
            strengths=[str(s) for s in data.get("strengths", [])],
            weaknesses=[str(w) for w in data.get("weaknesses", [])],
            suggestions=suggestions,
            unsupported_claims=[str(c) for c in data.get("unsupported_claims", [])],
            bias_flags=[str(b) for b in data.get("bias_flags", [])],
        )

        logger.info(
            "Critique score: %.2f (needs revision: %s)", result.overall_score, result.needs_revision
        )
        return result
