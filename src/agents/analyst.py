"""AnalystAgent: extracts claims, detects contradictions, and scores confidence."""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import AgentBase, build_prompt_with_context
from src.llm import LLMClient
from src.models.domain import (
    AnalysisResult,
    Claim,
    Contradiction,
    SearchResult,
    SourceReliability,
)

logger = logging.getLogger(__name__)

ANALYST_SYSTEM = build_prompt_with_context(
    system_role="research analyst specializing in claim extraction and source analysis",
    instructions="""\
Analyze the provided search results and extract structured findings.

For each piece of information:
1. Extract distinct claims with their supporting sources (URLs).
2. Assess confidence: "high" (3+ agreeing sources), "medium" (2 sources), "low" (single source).
3. Identify contradictions between sources.
4. Note gaps — important aspects not covered by any source.
5. Identify key themes across all results.

Respond with ONLY valid JSON:
{
  "claims": [
    {
      "statement": "<factual claim>",
      "sources": ["<url1>", "<url2>"],
      "confidence": "high|medium|low",
      "category": "<theme>"
    }
  ],
  "contradictions": [
    {
      "claim_a": "<first claim>",
      "claim_b": "<contradicting claim>",
      "source_a": "<url>",
      "source_b": "<url>",
      "explanation": "<why these contradict>"
    }
  ],
  "gaps": ["<missing aspect 1>", "<missing aspect 2>"],
  "key_themes": ["<theme1>", "<theme2>"]
}""",
)


class AnalystAgent(AgentBase):
    """Analyzes search results to extract claims and detect contradictions."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        super().__init__(llm)

    @property
    def name(self) -> str:
        return "analyst"

    def _format_sources(self, results: list[SearchResult]) -> str:
        """Format search results as context for the LLM."""
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            content = r.content or r.snippet
            parts.append(f"[Source {i}] {r.title}\nURL: {r.url}\nContent: {content[:3000]}\n")
        return "\n---\n".join(parts)

    def _build_reliability_map(self, results: list[SearchResult]) -> dict[str, SourceReliability]:
        return {r.url: r.reliability for r in results}

    async def run(self, **kwargs: Any) -> AnalysisResult:
        """Analyze search results.

        Args:
            **kwargs: Must include 'search_results' (list[SearchResult]).
                     Optionally 'rag_context' (str) for cross-reference.

        Returns:
            Structured AnalysisResult.
        """
        search_results: list[SearchResult] = kwargs["search_results"]
        rag_context: str = kwargs.get("rag_context", "")

        if not search_results:
            return AnalysisResult()

        formatted = self._format_sources(search_results)
        user_msg = f"Analyze these research sources:\n\n{formatted}"
        if rag_context:
            user_msg += f"\n\nAdditional cross-reference context:\n{rag_context}"

        data = await self._llm_json_call(ANALYST_SYSTEM, user_msg)

        claims = [
            Claim(
                statement=str(c["statement"]),
                sources=[str(s) for s in c.get("sources", [])],
                confidence=str(c.get("confidence", "low")),
                category=str(c.get("category", "")),
            )
            for c in data.get("claims", [])
        ]

        contradictions = [
            Contradiction(
                claim_a=str(ct["claim_a"]),
                claim_b=str(ct["claim_b"]),
                source_a=str(ct.get("source_a", "")),
                source_b=str(ct.get("source_b", "")),
                explanation=str(ct.get("explanation", "")),
            )
            for ct in data.get("contradictions", [])
        ]

        gaps = [str(g) for g in data.get("gaps", [])]
        key_themes = [str(t) for t in data.get("key_themes", [])]

        result = AnalysisResult(
            claims=claims,
            contradictions=contradictions,
            gaps=gaps,
            key_themes=key_themes,
            source_reliability=self._build_reliability_map(search_results),
        )

        logger.info(
            "Analysis complete: %d claims, %d contradictions, %d gaps",
            len(claims),
            len(contradictions),
            len(gaps),
        )
        return result
