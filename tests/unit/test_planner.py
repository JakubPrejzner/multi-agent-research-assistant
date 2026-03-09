"""Tests for PlannerAgent."""

from __future__ import annotations

import json

import pytest

from src.agents.planner import PlannerAgent
from src.models.domain import ResearchDepth
from tests.conftest import create_mock_llm


class TestPlannerAgent:
    @pytest.fixture()
    def planner_response(self) -> str:
        return json.dumps({
            "original_query": "quantum computing impact",
            "subtasks": [
                {
                    "id": "t1",
                    "query": "quantum computing fundamentals explained",
                    "priority": 1,
                    "depends_on": [],
                    "rationale": "Need to understand basics first",
                },
                {
                    "id": "t2",
                    "query": "quantum computing practical applications 2024",
                    "priority": 2,
                    "depends_on": ["t1"],
                    "rationale": "Practical uses require understanding fundamentals",
                },
                {
                    "id": "t3",
                    "query": "quantum computing industry challenges",
                    "priority": 3,
                    "depends_on": [],
                    "rationale": "Independent assessment of hurdles",
                },
            ],
            "reasoning": "Split into fundamentals, applications, and challenges",
            "estimated_complexity": "medium",
        })

    async def test_creates_plan(self, planner_response: str) -> None:
        mock_llm = create_mock_llm([planner_response])
        planner = PlannerAgent(llm=mock_llm, depth=ResearchDepth.STANDARD)
        plan = await planner.run(query="quantum computing impact")

        assert plan.original_query == "quantum computing impact"
        assert plan.task_count == 3
        assert plan.subtasks[0].id == "t1"
        assert plan.subtasks[1].depends_on == ["t1"]

    async def test_respects_depth_limit(self, planner_response: str) -> None:
        mock_llm = create_mock_llm([planner_response])
        planner = PlannerAgent(llm=mock_llm, depth=ResearchDepth.QUICK)
        plan = await planner.run(query="test")

        assert plan.task_count <= ResearchDepth.QUICK.max_subtasks

    async def test_fallback_on_empty_subtasks(self) -> None:
        response = json.dumps({
            "original_query": "test",
            "subtasks": [],
            "reasoning": "empty",
        })
        mock_llm = create_mock_llm([response])
        planner = PlannerAgent(llm=mock_llm)
        plan = await planner.run(query="test query")

        assert plan.task_count == 1
        assert plan.subtasks[0].query == "test query"

    async def test_name_property(self) -> None:
        planner = PlannerAgent(llm=create_mock_llm())
        assert planner.name == "planner"
