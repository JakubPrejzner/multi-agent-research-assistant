"""Status event emitters for real-time streaming."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class StatusEmitter:
    """Emits status events for WebSocket streaming and progress tracking."""

    def __init__(self) -> None:
        self._listeners: list[asyncio.Queue[dict[str, Any]]] = []
        self._history: list[dict[str, Any]] = []
        self._start_time = time.monotonic()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Create a new event listener queue."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._listeners.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove an event listener."""
        if queue in self._listeners:
            self._listeners.remove(queue)

    async def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Emit an event to all listeners."""
        event = {
            "type": event_type,
            "data": data or {},
            "timestamp": time.monotonic() - self._start_time,
        }
        self._history.append(event)
        logger.debug("Event: %s %s", event_type, data)

        for queue in self._listeners:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Event queue full, dropping event")

    async def emit_phase_start(self, phase: str, details: str = "") -> None:
        await self.emit("phase_start", {"phase": phase, "details": details})

    async def emit_phase_end(self, phase: str, duration: float = 0.0) -> None:
        await self.emit("phase_end", {"phase": phase, "duration": duration})

    async def emit_error(self, error: str, phase: str = "") -> None:
        await self.emit("error", {"error": error, "phase": phase})

    async def emit_complete(self, task_id: str) -> None:
        await self.emit("complete", {"task_id": task_id})

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)
