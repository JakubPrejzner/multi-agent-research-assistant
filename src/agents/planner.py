"""PlannerAgent: decomposes user queries into research sub-tasks."""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import AgentBase, build_prompt_with_context
from src.llm import LLMClient
from src.models.domain import ResearchDepth, ResearchPlan, SubTask

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = build_prompt_with_context(
    system_role="research planning specialist",
    instructions="""\
Given a user's research query, decompose it into focused sub-tasks.

Rules:
- Create between 3 and {max_tasks} sub-tasks depending on complexity.
- Each sub-task should be a specific, searchable question.
- Assign priority 1 (highest) to 10 (lowest).
- Set depends_on if a sub-task requires results from another.
- Provide rationale for each sub-task.

Respond with ONLY valid JSON matching this schema:
{{
  "original_query": "<the user's query>",
  "subtasks": [
    {{
      "id": "<short unique id>",
      "query": "<specific search query>",
      "priority": <1-10>,
      "depends_on": ["<id of dependency>"],
      "rationale": "<why this sub-task>"
    }}
  ],
  "reasoning": "<overall decomposition strategy>",
  "estimated_complexity": "<low|medium|high>"
}}""",
)


class PlannerAgent(AgentBase):
    """Decomposes a research query into prioritized sub-tasks."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        depth: ResearchDepth = ResearchDepth.STANDARD,
    ) -> None:
        super().__init__(llm)
        self.depth = depth

    @property
    def name(self) -> str:
        return "planner"

    async def run(self, **kwargs: Any) -> ResearchPlan:
        """Create a research plan from a user query.

        Args:
            **kwargs: Must include 'query' (str).

        Returns:
            A structured ResearchPlan.
        """
        query: str = kwargs["query"]
        max_tasks = self.depth.max_subtasks

        system = PLANNER_SYSTEM.replace("{max_tasks}", str(max_tasks))
        data = await self._llm_json_call(system, query)

        subtasks = [
            SubTask(
                id=str(st.get("id", f"t{i}")),
                query=str(st["query"]),
                priority=int(st.get("priority", i + 1)),
                depends_on=[str(d) for d in st.get("depends_on", [])],
                rationale=str(st.get("rationale", "")),
            )
            for i, st in enumerate(data.get("subtasks", []))
        ]

        if not subtasks:
            subtasks = [
                SubTask(id="t1", query=query, priority=1, rationale="Direct search"),
            ]

        plan = ResearchPlan(
            original_query=query,
            subtasks=subtasks[:max_tasks],
            reasoning=str(data.get("reasoning", "")),
            estimated_complexity=str(data.get("estimated_complexity", "medium")),
        )

        logger.info("Created plan with %d subtasks for: %s", plan.task_count, query)
        return plan
