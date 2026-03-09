"""Research task endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

import markdown
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.api.middleware import limiter
from src.api.schemas import (
    ReportResponse,
    ResearchRequest,
    ResearchResponse,
    ResearchStatusResponse,
)
from src.models.domain import ReportFormat, ResearchStatus
from src.orchestrator.callbacks import StatusEmitter
from src.orchestrator.graph import run_research

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory task store (production would use Redis)
_tasks: dict[str, dict[str, Any]] = {}
_emitters: dict[str, StatusEmitter] = {}


def _get_task_store() -> dict[str, dict[str, Any]]:
    return _tasks


@router.post("/research", response_model=ResearchResponse, status_code=202)
async def create_research(
    request: ResearchRequest,
) -> ResearchResponse:
    """Start a new research task."""
    task_id = uuid.uuid4().hex[:12]
    emitter = StatusEmitter()

    _tasks[task_id] = {
        "task_id": task_id,
        "query": request.query,
        "depth": request.depth.value,
        "model": request.model,
        "status": ResearchStatus.PENDING,
        "result": None,
    }
    _emitters[task_id] = emitter

    asyncio.create_task(
        _run_research_task(task_id, request.query, request.depth.value, request.model, emitter)
    )

    return ResearchResponse(
        task_id=task_id,
        status=ResearchStatus.PENDING,
        message=f"Research task created. Poll GET /research/{task_id} for status.",
    )


async def _run_research_task(
    task_id: str,
    query: str,
    depth: str,
    model: str,
    emitter: StatusEmitter,
) -> None:
    """Background task runner."""
    try:
        _tasks[task_id]["status"] = ResearchStatus.PLANNING
        result = await run_research(
            query=query,
            depth=depth,
            model=model or None,
            task_id=task_id,
            emitter=emitter,
        )
        _tasks[task_id]["result"] = result
        _tasks[task_id]["status"] = ResearchStatus(result.get("status", "completed"))
    except Exception as e:
        logger.exception("Research task %s failed: %s", task_id, e)
        _tasks[task_id]["status"] = ResearchStatus.FAILED
        _tasks[task_id]["error"] = str(e)


@router.get("/research/{task_id}", response_model=ResearchStatusResponse)
async def get_research_status(task_id: str) -> ResearchStatusResponse | JSONResponse:
    """Poll research task status and partial results."""
    task = _tasks.get(task_id)
    if not task:
        return JSONResponse(
            status_code=404,
            content={
                "type": "about:blank",
                "title": "Not Found",
                "status": 404,
                "detail": f"Task {task_id} not found",
            },
        )

    result = task.get("result") or {}
    partial: dict[str, Any] = {}

    plan = result.get("research_plan")
    if plan:
        partial["plan"] = {
            "subtask_count": plan.task_count,
            "subtasks": [{"query": st.query, "priority": st.priority} for st in plan.subtasks],
        }

    search_results = result.get("search_results", [])
    if search_results:
        partial["search_result_count"] = len(search_results)

    analysis = result.get("analysis")
    if analysis:
        partial["claim_count"] = len(analysis.claims)
        partial["contradiction_count"] = len(analysis.contradictions)

    return ResearchStatusResponse(
        task_id=task_id,
        status=task["status"],
        current_phase=result.get("current_phase", ""),
        progress=result.get("phase_timings", {}),
        errors=result.get("errors", []),
        partial_results=partial,
    )


@router.get("/research/{task_id}/report", response_model=ReportResponse)
async def get_research_report(
    task_id: str,
    format: ReportFormat = ReportFormat.MARKDOWN,
) -> ReportResponse | JSONResponse:
    """Retrieve the final research report."""
    task = _tasks.get(task_id)
    if not task:
        return JSONResponse(
            status_code=404,
            content={
                "type": "about:blank",
                "title": "Not Found",
                "status": 404,
                "detail": f"Task {task_id} not found",
            },
        )

    result = task.get("result") or {}
    report = result.get("final_report")
    if not report:
        return JSONResponse(
            status_code=409,
            content={
                "type": "about:blank",
                "title": "Conflict",
                "status": 409,
                "detail": "Report not yet available. Current status: "
                + str(task["status"].value),
            },
        )

    if format == ReportFormat.JSON:
        content = json.dumps(report.model_dump(), indent=2, default=str)
    elif format == ReportFormat.HTML:
        content = markdown.markdown(report.markdown or report.executive_summary)
    else:
        content = report.markdown or _build_markdown(report)

    metadata: dict[str, Any] = {}
    timings = result.get("phase_timings")
    if timings:
        metadata["phase_timings"] = timings

    return ReportResponse(
        task_id=task_id,
        format=format,
        content=content,
        metadata=metadata,
    )


def _build_markdown(report: Any) -> str:
    """Build markdown from report fields when no pre-built markdown exists."""
    parts = [
        f"# {report.title}\n",
        f"## Executive Summary\n{report.executive_summary}\n",
    ]
    if report.key_findings:
        parts.append("## Key Findings")
        for i, f in enumerate(report.key_findings, 1):
            parts.append(f"{i}. {f}")
        parts.append("")
    if report.contradictions:
        parts.append("## Contradictions & Open Questions")
        for c in report.contradictions:
            parts.append(f"- {c}")
        parts.append("")
    if report.sources:
        parts.append("## Sources")
        for s in report.sources:
            parts.append(f"- [{s.get('title', 'Source')}]({s.get('url', '')})")
        parts.append("")
    return "\n".join(parts)


@router.websocket("/research/{task_id}/stream")
async def research_stream(websocket: WebSocket, task_id: str) -> None:
    """WebSocket for real-time research status streaming."""
    await websocket.accept()

    emitter = _emitters.get(task_id)
    if not emitter:
        await websocket.send_json({"error": f"Task {task_id} not found"})
        await websocket.close()
        return

    queue = emitter.subscribe()

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=60.0)
                await websocket.send_json(event)
                if event.get("type") == "complete":
                    break
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected for task %s", task_id)
    finally:
        emitter.unsubscribe(queue)
