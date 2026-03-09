"""Rich CLI for the research assistant."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.config import get_settings
from src.orchestrator.callbacks import StatusEmitter
from src.orchestrator.graph import run_research

app = typer.Typer(
    name="research",
    help="Multi-Agent Research Assistant CLI",
    no_args_is_help=True,
)
console = Console()

# In-memory store for CLI session
_cli_results: dict[str, dict[str, Any]] = {}


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


async def _execute_research(
    query: str,
    depth: str,
    model: str,
) -> dict[str, Any]:
    """Execute research with live progress display."""
    emitter = StatusEmitter()
    task_id = f"cli-{int(time.time())}"

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    )

    research_task = asyncio.create_task(
        run_research(
            query=query,
            depth=depth,
            model=model,
            task_id=task_id,
            emitter=emitter,
        )
    )

    queue = emitter.subscribe()
    progress_task = progress.add_task("Starting research...", total=None)

    with Live(progress, console=console, refresh_per_second=4):
        while not research_task.done():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
                event_type = event.get("type", "")
                data = event.get("data", {})

                if event_type == "phase_start":
                    phase = data.get("phase", "")
                    details = data.get("details", "")
                    progress.update(
                        progress_task,
                        description=f"[bold blue]{phase.title()}: {details}",
                    )
                elif event_type == "phase_end":
                    phase = data.get("phase", "")
                    duration = data.get("duration", 0)
                    console.print(
                        f"  [green]\u2713[/green] {phase.title()} completed ({duration:.1f}s)"
                    )
                elif event_type == "complete":
                    progress.update(progress_task, description="[bold green]Complete!")
                elif event_type == "error":
                    console.print(f"  [red]\u2717[/red] Error: {data.get('error', '')}")
            except TimeoutError:
                continue

    result: dict[str, Any] = dict(research_task.result())
    _cli_results[task_id] = result
    return result


@app.command()
def research(
    query: str = typer.Argument(help="Research question to investigate"),
    depth: str = typer.Option("standard", help="Research depth: quick, standard, deep"),
    model: str = typer.Option("", help="LLM model override"),
) -> None:
    """Run a research query with live progress display."""
    if depth not in ("quick", "standard", "deep"):
        console.print("[red]Error: depth must be quick, standard, or deep[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Query:[/bold] {query}\n[bold]Depth:[/bold] {depth}\n[bold]Model:[/bold] {model or 'default'}",
            title="Research Assistant",
            border_style="blue",
        )
    )

    settings = get_settings()
    effective_model = model or settings.default_model

    result = _run_async(_execute_research(query, depth, effective_model))

    status = result.get("status", "unknown")
    if status == "completed":
        report = result.get("final_report")
        if report:
            console.print()
            console.print(
                Panel(
                    Markdown(report.markdown or report.executive_summary),
                    title=report.title,
                    border_style="green",
                )
            )

            timings = result.get("phase_timings", {})
            if timings:
                table = Table(title="Phase Timings")
                table.add_column("Phase", style="cyan")
                table.add_column("Duration", style="green")
                for phase, duration in timings.items():
                    table.add_row(phase, f"{duration:.2f}s")
                console.print(table)
        else:
            console.print("[yellow]Research completed but no report generated.[/yellow]")
    else:
        errors = result.get("errors", [])
        console.print(f"[red]Research failed with status: {status}[/red]")
        for err in errors:
            console.print(f"  [red]- {err}[/red]")
        raise typer.Exit(1)


@app.command(name="list")
def list_tasks() -> None:
    """List past research tasks from this session."""
    if not _cli_results:
        console.print("[dim]No research tasks in this session.[/dim]")
        return

    table = Table(title="Research Tasks")
    table.add_column("Task ID", style="cyan")
    table.add_column("Query", max_width=50)
    table.add_column("Status", style="green")

    for task_id, result in _cli_results.items():
        query = str(result.get("query", ""))[:50]
        status = str(result.get("status", "unknown"))
        table.add_row(task_id, query, status)

    console.print(table)


@app.command()
def show(task_id: str = typer.Argument(help="Task ID to display")) -> None:
    """Display a research report in the terminal."""
    result = _cli_results.get(task_id)
    if not result:
        console.print(f"[red]Task {task_id} not found in this session.[/red]")
        raise typer.Exit(1)

    report = result.get("final_report")
    if not report:
        console.print("[yellow]No report available for this task.[/yellow]")
        raise typer.Exit(1)

    console.print(
        Panel(
            Markdown(report.markdown or report.executive_summary),
            title=report.title,
            border_style="green",
        )
    )

    if report.key_findings:
        console.print("\n[bold]Key Findings:[/bold]")
        for i, finding in enumerate(report.key_findings, 1):
            console.print(f"  {i}. {finding}")

    if report.sources:
        console.print("\n[bold]Sources:[/bold]")
        for src in report.sources:
            title = src.get("title", "Unknown")
            url = src.get("url", "")
            console.print(f"  - {title}: {url}")


@app.command()
def export(
    task_id: str = typer.Argument(help="Task ID to export"),
    format: str = typer.Option("markdown", help="Export format: markdown, json, html"),
    output: str = typer.Option("", help="Output file path (default: stdout)"),
) -> None:
    """Export a research report to a file."""
    result = _cli_results.get(task_id)
    if not result:
        console.print(f"[red]Task {task_id} not found.[/red]")
        raise typer.Exit(1)

    report = result.get("final_report")
    if not report:
        console.print("[yellow]No report available.[/yellow]")
        raise typer.Exit(1)

    if format == "json":
        content = json.dumps(report.model_dump(), indent=2, default=str)
    elif format == "html":
        import markdown as md_lib

        html = md_lib.markdown(report.markdown or report.executive_summary)
        content = f"<!DOCTYPE html><html><head><title>{report.title}</title></head><body>{html}</body></html>"
    else:
        content = report.markdown or report.executive_summary

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[green]Report exported to {output}[/green]")
    else:
        console.print(content)


if __name__ == "__main__":
    app()
