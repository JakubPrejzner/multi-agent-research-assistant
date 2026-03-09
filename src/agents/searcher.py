"""SearchAgent: executes web searches with Tavily primary and DuckDuckGo fallback."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from duckduckgo_search import DDGS

from src.agents.base import AgentBase
from src.config import get_settings
from src.llm import LLMClient
from src.models.domain import ResearchDepth, SearchResult, SourceReliability

logger = logging.getLogger(__name__)


async def _search_tavily(
    query: str,
    max_results: int,
    api_key: str,
) -> list[dict[str, Any]]:
    """Search via Tavily API."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_raw_content": True,
                "search_depth": "advanced",
            },
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return list(data.get("results", []))


def _search_ddg(query: str, max_results: int) -> list[dict[str, Any]]:
    """Search via DuckDuckGo (no API key needed)."""
    results: list[dict[str, Any]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "url": r.get("href", ""),
                    "title": r.get("title", ""),
                    "content": r.get("body", ""),
                }
            )
    return results


def _assess_reliability(url: str) -> SourceReliability:
    """Heuristic source reliability based on domain."""
    high_trust = [
        "nature.com",
        "science.org",
        "ieee.org",
        "acm.org",
        "gov",
        "edu",
        "who.int",
        "nih.gov",
        "arxiv.org",
        "scholar.google",
    ]
    medium_trust = [
        "wikipedia.org",
        "reuters.com",
        "bbc.com",
        "nytimes.com",
        "theguardian.com",
        "washingtonpost.com",
        "apnews.com",
    ]
    url_lower = url.lower()
    for domain in high_trust:
        if domain in url_lower:
            return SourceReliability.HIGH
    for domain in medium_trust:
        if domain in url_lower:
            return SourceReliability.MEDIUM
    return SourceReliability.UNKNOWN


class SearchAgent(AgentBase):
    """Executes web searches with automatic fallback."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        depth: ResearchDepth = ResearchDepth.STANDARD,
    ) -> None:
        super().__init__(llm)
        self.depth = depth

    @property
    def name(self) -> str:
        return "searcher"

    async def _search_single(self, query: str, max_results: int) -> list[SearchResult]:
        """Run search with Tavily primary, DuckDuckGo fallback."""
        settings = get_settings()
        raw_results: list[dict[str, Any]] = []

        if settings.has_tavily:
            try:
                raw_results = await _search_tavily(query, max_results, settings.tavily_api_key)
                logger.info("Tavily returned %d results for: %s", len(raw_results), query)
            except Exception as e:
                logger.warning("Tavily search failed (%s), falling back to DuckDuckGo", e)

        if not raw_results:
            try:
                raw_results = _search_ddg(query, max_results)
                logger.info("DDG returned %d results for: %s", len(raw_results), query)
            except Exception as e:
                logger.error("DuckDuckGo search also failed: %s", e)
                return []

        seen_urls: set[str] = set()
        results: list[SearchResult] = []
        for r in raw_results:
            url = str(r.get("url", r.get("href", "")))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            results.append(
                SearchResult(
                    url=url,
                    title=str(r.get("title", "")),
                    snippet=str(r.get("content", r.get("snippet", "")))[:500],
                    content=str(r.get("raw_content", r.get("content", "")))[:5000],
                    source="tavily" if settings.has_tavily else "duckduckgo",
                    relevance_score=float(r.get("score", 0.5)),
                    reliability=_assess_reliability(url),
                )
            )

        return results

    async def run(self, **kwargs: Any) -> list[SearchResult]:
        """Search for all subtask queries.

        Args:
            **kwargs: Must include 'queries' (list[str]).

        Returns:
            Deduplicated list of SearchResult.
        """
        queries: list[str] = kwargs["queries"]
        max_per = self.depth.max_results_per_task
        all_results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for query in queries:
            results = await self._search_single(query, max_per)
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)

        logger.info("Total search results: %d (from %d queries)", len(all_results), len(queries))
        return all_results
