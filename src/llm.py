"""LiteLLM wrapper with retry logic, fallback models, and cost tracking."""

from __future__ import annotations

import logging
import time
from typing import Any

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


class LLMError(Exception):
    """Raised when all LLM call attempts fail."""


class TokenUsage:
    """Tracks cumulative token usage and estimated cost."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.call_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, usage: dict[str, Any], cost: float | None = None) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        if cost is not None:
            self.total_cost_usd += cost
        self.call_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "call_count": self.call_count,
        }


class LLMClient:
    """Unified LLM interface via LiteLLM with retry and fallback support."""

    def __init__(
        self,
        model: str | None = None,
        fallback_model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.default_model
        self.fallback_model = fallback_model or settings.fallback_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = settings.llm_timeout
        self.max_retries = settings.llm_max_retries
        self.usage = TokenUsage()

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_model(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Make a single LLM call with retry logic."""
        start = time.monotonic()
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            timeout=self.timeout,
        )
        elapsed = time.monotonic() - start

        # Extract usage
        usage_data = getattr(response, "usage", None)
        cost: float | None = None
        if usage_data:
            usage_dict = {
                "prompt_tokens": getattr(usage_data, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_data, "completion_tokens", 0),
            }
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = None
            self.usage.record(usage_dict, cost)

        content = str(response.choices[0].message.content or "")
        logger.debug(
            "LLM call: model=%s tokens=%d elapsed=%.2fs",
            model,
            self.usage.total_tokens,
            elapsed,
        )
        return content

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Call the LLM with automatic fallback on failure.

        Args:
            messages: Chat messages in OpenAI format.
            model: Override model for this call.
            **kwargs: Additional parameters (temperature, max_tokens).

        Returns:
            The assistant's response text.

        Raises:
            LLMError: If both primary and fallback models fail.
        """
        target_model = model or self.model
        try:
            return await self._call_model(target_model, messages, **kwargs)
        except Exception as primary_err:
            if target_model == self.fallback_model:
                raise LLMError(f"LLM call failed: {primary_err}") from primary_err

            logger.warning(
                "Primary model %s failed (%s), trying fallback %s",
                target_model,
                primary_err,
                self.fallback_model,
            )
            try:
                return await self._call_model(self.fallback_model, messages, **kwargs)
            except Exception as fallback_err:
                raise LLMError(
                    f"Both models failed. Primary ({target_model}): {primary_err}. "
                    f"Fallback ({self.fallback_model}): {fallback_err}"
                ) from fallback_err

    async def complete_with_system(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs: Any,
    ) -> str:
        """Convenience method: system + user message pair."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return await self.complete(messages, **kwargs)
