"""Integration tests for the FastAPI layer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app
from src.api.routes.research import _tasks
from src.models.domain import (
    Report,
    ResearchStatus,
)


@pytest.fixture()
def app():  # type: ignore[no-untyped-def]
    return create_app()


@pytest.fixture()
async def client(app):  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_tasks() -> None:
    _tasks.clear()


class TestHealthEndpoint:
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert data["version"] == "0.1.0"
        assert "redis" in data["dependencies"] or "chromadb" in data["dependencies"]


class TestResearchEndpoints:
    async def test_create_research_returns_202(self, client: AsyncClient) -> None:
        with patch("src.api.routes.research.run_research", new_callable=AsyncMock):
            resp = await client.post(
                "/research",
                json={"query": "What is quantum computing?", "depth": "quick"},
            )
        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "pending"

    async def test_create_research_validation(self, client: AsyncClient) -> None:
        resp = await client.post("/research", json={"query": "ab"})
        assert resp.status_code == 422

    async def test_get_status_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/research/nonexistent")
        assert resp.status_code == 404

    async def test_get_status_exists(self, client: AsyncClient) -> None:
        _tasks["test-123"] = {
            "task_id": "test-123",
            "query": "test",
            "status": ResearchStatus.PLANNING,
            "result": {"current_phase": "planning", "phase_timings": {}, "errors": []},
        }

        resp = await client.get("/research/test-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "test-123"
        assert data["status"] == "planning"

    async def test_get_report_not_ready(self, client: AsyncClient) -> None:
        _tasks["test-456"] = {
            "task_id": "test-456",
            "query": "test",
            "status": ResearchStatus.RESEARCHING,
            "result": {},
        }

        resp = await client.get("/research/test-456/report")
        assert resp.status_code == 409

    async def test_get_report_markdown(self, client: AsyncClient) -> None:
        report = Report(
            title="Test Report",
            executive_summary="Summary here.",
            key_findings=["Finding 1"],
            sources=[],
            markdown="# Test Report\n\nSummary.",
        )
        _tasks["test-789"] = {
            "task_id": "test-789",
            "query": "test",
            "status": ResearchStatus.COMPLETED,
            "result": {"final_report": report, "phase_timings": {"planning": 1.5}},
        }

        resp = await client.get("/research/test-789/report?format=markdown")
        assert resp.status_code == 200
        data = resp.json()
        assert data["format"] == "markdown"
        assert "# Test Report" in data["content"]

    async def test_get_report_json_format(self, client: AsyncClient) -> None:
        report = Report(
            title="Test Report",
            executive_summary="Summary.",
            key_findings=["Finding"],
            markdown="# Test",
        )
        _tasks["test-json"] = {
            "task_id": "test-json",
            "query": "test",
            "status": ResearchStatus.COMPLETED,
            "result": {"final_report": report},
        }

        resp = await client.get("/research/test-json/report?format=json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["format"] == "json"
        content = json.loads(data["content"])
        assert content["title"] == "Test Report"

    async def test_get_report_html_format(self, client: AsyncClient) -> None:
        report = Report(
            title="Test Report",
            executive_summary="Summary.",
            markdown="# Test Report\n\nSummary paragraph.",
        )
        _tasks["test-html"] = {
            "task_id": "test-html",
            "query": "test",
            "status": ResearchStatus.COMPLETED,
            "result": {"final_report": report},
        }

        resp = await client.get("/research/test-html/report?format=html")
        assert resp.status_code == 200
        data = resp.json()
        assert data["format"] == "html"
        assert "<h1>" in data["content"]
