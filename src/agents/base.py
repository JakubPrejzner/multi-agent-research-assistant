"""Base agent protocol and shared utilities."""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

from src.llm import LLMClient

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseAgent(Protocol):
    """Protocol that all agents must satisfy."""

    @property
    def name(self) -> str: ...

    async def run(self, **kwargs: Any) -> Any: ...


def parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from an LLM response, handling markdown code fences."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        start = 1
        end = len(lines)
        for i in range(1, len(lines)):
            if lines[i].strip() == "```":
                end = i
                break
        cleaned = "\n".join(lines[start:end]).strip()

    result: dict[str, Any] = json.loads(cleaned)
    return result


def build_prompt_with_context(
    system_role: str,
    instructions: str,
    context: dict[str, str] | None = None,
) -> str:
    """Build a system prompt with optional context sections."""
    parts = [f"You are a {system_role}.", "", instructions]
    if context:
        parts.append("")
        for key, value in context.items():
            parts.append(f"## {key}")
            parts.append(value)
            parts.append("")
    return "\n".join(parts)


class AgentBase:
    """Shared base implementation for agents."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    @property
    def llm(self) -> LLMClient:
        return self._llm

    async def _llm_json_call(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call LLM and parse response as JSON."""
        raw = await self._llm.complete_with_system(system_prompt, user_message, **kwargs)
        return parse_json_response(raw)
