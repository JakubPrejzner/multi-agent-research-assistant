"""LangGraph StateGraph definition for the research workflow."""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.analyst import AnalystAgent
from src.agents.critic import CriticAgent
from src.agents.planner import PlannerAgent
from src.agents.searcher import SearchAgent
from src.agents.writer import WriterAgent
from src.config import get_settings
from src.llm import LLMClient
from src.models.domain import ResearchDepth
from src.orchestrator.callbacks import StatusEmitter
from src.orchestrator.state import ResearchState
from src.rag.chunker import chunk_documents
from src.rag.retriever import HybridRetriever
from src.rag.store import VectorStore

logger = logging.getLogger(__name__)


async def planning_node(state: ResearchState) -> dict[str, Any]:
    """Decompose the query into sub-tasks."""
    emitter: StatusEmitter | None = state.get("status_callback")
    if emitter:
        await emitter.emit_phase_start("planning", "Decomposing research query")

    start = time.monotonic()
    depth = ResearchDepth(state.get("depth", "standard"))
    model = state.get("model")
    llm = LLMClient(model=model) if model else LLMClient()

    planner = PlannerAgent(llm=llm, depth=depth)

    try:
        plan = await planner.run(query=state["query"])
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"Planning failed: {e}")
        return {"errors": errors, "status": "failed", "current_phase": "planning"}

    elapsed = time.monotonic() - start
    timings = dict(state.get("phase_timings", {}))
    timings["planning"] = elapsed

    if emitter:
        await emitter.emit_phase_end("planning", elapsed)

    return {
        "research_plan": plan,
        "current_phase": "planning",
        "status": "awaiting_approval",
        "phase_timings": timings,
    }


async def search_node(state: ResearchState) -> dict[str, Any]:
    """Execute web searches for all sub-tasks."""
    emitter: StatusEmitter | None = state.get("status_callback")
    if emitter:
        await emitter.emit_phase_start("searching", "Executing web searches")

    start = time.monotonic()
    plan = state.get("research_plan")
    if not plan:
        return {"errors": [*state.get("errors", []), "No plan available"], "status": "failed"}

    depth = ResearchDepth(state.get("depth", "standard"))
    model = state.get("model")
    llm = LLMClient(model=model) if model else LLMClient()

    searcher = SearchAgent(llm=llm, depth=depth)
    queries = [st.query for st in plan.subtasks]

    try:
        results = await searcher.run(queries=queries)
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"Search failed: {e}")
        return {"errors": errors, "status": "failed", "current_phase": "searching"}

    # Index results in RAG store
    rag_context = ""
    if results:
        task_id = state.get("task_id", "default")
        try:
            docs = [
                {"content": r.content or r.snippet, "url": r.url, "title": r.title} for r in results
            ]
            chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=64)

            store = VectorStore(f"research_{task_id}")
            chunk_texts = [c["content"] for c in chunks]
            chunk_metas = [{"url": c["url"], "title": c["title"]} for c in chunks]

            retriever = HybridRetriever(store, documents=chunk_texts, metadatas=chunk_metas)
            retriever.add_documents(chunk_texts, metadatas=chunk_metas)

            cross_ref = retriever.retrieve(state["query"], n_results=5)
            rag_context = "\n\n".join(r["content"] for r in cross_ref)
        except Exception as e:
            logger.warning("RAG indexing failed (non-fatal): %s", e)

    elapsed = time.monotonic() - start
    timings = dict(state.get("phase_timings", {}))
    timings["searching"] = elapsed

    if emitter:
        await emitter.emit_phase_end("searching", elapsed)

    return {
        "search_results": results,
        "rag_context": rag_context,
        "current_phase": "researching",
        "status": "analyzing",
        "phase_timings": timings,
    }


async def analysis_node(state: ResearchState) -> dict[str, Any]:
    """Analyze search results and extract claims."""
    emitter: StatusEmitter | None = state.get("status_callback")
    if emitter:
        await emitter.emit_phase_start(
            "analyzing", "Extracting claims and detecting contradictions"
        )

    start = time.monotonic()
    model = state.get("model")
    llm = LLMClient(model=model) if model else LLMClient()
    analyst = AnalystAgent(llm=llm)

    results = state.get("search_results", [])
    rag_context = state.get("rag_context", "")

    try:
        analysis = await analyst.run(search_results=results, rag_context=rag_context)
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"Analysis failed: {e}")
        return {"errors": errors, "status": "failed", "current_phase": "analyzing"}

    elapsed = time.monotonic() - start
    timings = dict(state.get("phase_timings", {}))
    timings["analyzing"] = elapsed

    if emitter:
        await emitter.emit_phase_end("analyzing", elapsed)

    return {
        "analysis": analysis,
        "current_phase": "analyzing",
        "status": "writing",
        "phase_timings": timings,
    }


async def writing_node(state: ResearchState) -> dict[str, Any]:
    """Write or revise the research report."""
    emitter: StatusEmitter | None = state.get("status_callback")
    revision_count = state.get("revision_count", 0)
    is_revision = revision_count > 0

    phase_name = "revising" if is_revision else "writing"
    if emitter:
        await emitter.emit_phase_start(
            phase_name, f"{'Revising' if is_revision else 'Writing'} report"
        )

    start = time.monotonic()
    model = state.get("model")
    llm = LLMClient(model=model) if model else LLMClient()
    writer = WriterAgent(llm=llm)

    try:
        if is_revision and state.get("draft_report") and state.get("critique"):
            report = await writer.revise(state["draft_report"], state["critique"])  # type: ignore[arg-type]
        else:
            analysis = state.get("analysis")
            if not analysis:
                return {"errors": [*state.get("errors", []), "No analysis"], "status": "failed"}
            report = await writer.run(analysis=analysis, query=state["query"])
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"Writing failed: {e}")
        return {"errors": errors, "status": "failed", "current_phase": phase_name}

    elapsed = time.monotonic() - start
    timings = dict(state.get("phase_timings", {}))
    timings[phase_name] = elapsed

    if emitter:
        await emitter.emit_phase_end(phase_name, elapsed)

    return {
        "draft_report": report,
        "current_phase": phase_name,
        "status": "critiquing",
        "phase_timings": timings,
    }


async def critique_node(state: ResearchState) -> dict[str, Any]:
    """Critique the draft report."""
    emitter: StatusEmitter | None = state.get("status_callback")
    if emitter:
        await emitter.emit_phase_start("critiquing", "Reviewing report quality")

    start = time.monotonic()
    model = state.get("model")
    llm = LLMClient(model=model) if model else LLMClient()
    critic = CriticAgent(llm=llm)

    report = state.get("draft_report")
    if not report:
        return {"errors": [*state.get("errors", []), "No draft report"], "status": "failed"}

    try:
        critique = await critic.run(report=report)
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"Critique failed: {e}")
        return {"errors": errors, "status": "failed", "current_phase": "critiquing"}

    elapsed = time.monotonic() - start
    timings = dict(state.get("phase_timings", {}))
    timings["critiquing"] = elapsed

    revision_count = state.get("revision_count", 0) + 1

    if emitter:
        await emitter.emit_phase_end("critiquing", elapsed)

    return {
        "critique": critique,
        "revision_count": revision_count,
        "current_phase": "critiquing",
        "phase_timings": timings,
    }


def should_revise(state: ResearchState) -> str:
    """Conditional edge: revise if critique score < threshold and under revision limit."""
    critique = state.get("critique")
    revision_count = state.get("revision_count", 0)
    settings = get_settings()
    max_revisions = state.get("max_revisions", settings.max_revision_cycles)

    if state.get("status") == "failed":
        return "finalize"

    if critique and critique.needs_revision and revision_count < max_revisions:
        logger.info(
            "Revision needed (score=%.2f, revision %d/%d)",
            critique.overall_score,
            revision_count,
            max_revisions,
        )
        return "revise"

    return "finalize"


async def finalize_node(state: ResearchState) -> dict[str, Any]:
    """Finalize the report."""
    emitter: StatusEmitter | None = state.get("status_callback")
    if emitter:
        await emitter.emit_complete(state.get("task_id", ""))

    return {
        "final_report": state.get("draft_report"),
        "status": "completed",
        "current_phase": "completed",
    }


def build_research_graph() -> StateGraph:  # type: ignore[type-arg]
    """Construct the LangGraph StateGraph for the research workflow."""
    graph = StateGraph(ResearchState)

    graph.add_node("plan", planning_node)
    graph.add_node("search", search_node)
    graph.add_node("analyze", analysis_node)
    graph.add_node("write", writing_node)
    graph.add_node("critique", critique_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "analyze")
    graph.add_edge("analyze", "write")
    graph.add_edge("write", "critique")

    graph.add_conditional_edges(
        "critique",
        should_revise,
        {
            "revise": "write",
            "finalize": "finalize",
        },
    )

    graph.add_edge("finalize", END)

    return graph


async def run_research(
    query: str,
    depth: str = "standard",
    model: str | None = None,
    task_id: str = "",
    emitter: StatusEmitter | None = None,
) -> ResearchState:
    """Execute the full research workflow.

    Args:
        query: The research question.
        depth: Research depth (quick, standard, deep).
        model: LLM model identifier.
        task_id: Unique task identifier.
        emitter: Optional status emitter for real-time events.

    Returns:
        Final research state with all results.
    """
    graph = build_research_graph()
    compiled = graph.compile()

    initial_state: ResearchState = {
        "query": query,
        "depth": depth,
        "model": model or get_settings().default_model,
        "task_id": task_id,
        "current_phase": "starting",
        "status": "pending",
        "plan_approved": True,
        "revision_count": 0,
        "max_revisions": get_settings().max_revision_cycles,
        "errors": [],
        "phase_timings": {},
        "token_usage": {},
        "search_results": [],
        "rag_context": "",
        "status_callback": emitter,
    }

    result: ResearchState = await compiled.ainvoke(initial_state)  # type: ignore[assignment,arg-type]
    return result
