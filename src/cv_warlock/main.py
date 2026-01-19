"""CLI entry point for CV Warlock."""

import time
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv

# Load environment variables from .env.local (including LangSmith config)
# Path: main.py -> cv_warlock/ -> src/ -> project root
load_dotenv(Path(__file__).parent.parent.parent / ".env.local")

import typer  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

from cv_warlock.graph.workflow import run_cv_tailoring  # noqa: E402
from cv_warlock.output.markdown import format_match_analysis, save_markdown  # noqa: E402


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


app = typer.Typer(
    name="cv-warlock",
    help="CV Warlock - AI-powered CV tailoring for job applications",
    add_completion=False,
)
console = Console()


def read_file(path: Path) -> str:
    """Read file content as text."""
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(1)
    return path.read_text(encoding="utf-8")


@app.command()
def tailor(
    cv: Annotated[Path, typer.Argument(help="Path to your CV (txt or md)")],
    job: Annotated[Path, typer.Argument(help="Path to job specification")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")] = Path(
        "tailored_cv.md"
    ),
    provider: Annotated[
        Literal["openai", "anthropic", "google"],
        typer.Option("--provider", "-p", help="LLM provider (model auto-selected)"),
    ] = "anthropic",
    lookback_years: Annotated[
        int | None,
        typer.Option(
            "--lookback-years",
            "-l",
            help="Only tailor jobs ending within N years (default: 4)",
        ),
    ] = None,
    no_rlm: Annotated[
        bool,
        typer.Option(
            "--no-rlm",
            help="Disable RLM (Recursive Language Model) for large document handling",
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed progress")
    ] = False,
) -> None:
    """Tailor your CV to match a specific job posting."""
    console.print(
        Panel.fit(
            "[bold blue]CV Warlock[/bold blue] - Tailoring your CV",
            border_style="blue",
        )
    )

    # Read input files
    raw_cv = read_file(cv)
    raw_job_spec = read_file(job)

    if verbose:
        console.print(f"[dim]CV:[/dim] {cv}")
        console.print(f"[dim]Job:[/dim] {job}")
        console.print(f"[dim]Provider:[/dim] {provider} (model auto-selected)")
        console.print(f"[dim]Lookback:[/dim] {lookback_years or 4} years")
        console.print(f"[dim]RLM Mode:[/dim] {'Disabled' if no_rlm else 'Enabled'}")
        console.print()

    # Step descriptions for display
    step_labels = {
        "validate_inputs": "Initializing",
        "extract_cv": "Extracting CV",
        "extract_job": "Analyzing job",
        "analyze_match": "Matching profile",
        "create_plan": "Creating plan",
        "tailor_skills": "Tailoring skills",
        "tailor_experiences": "Tailoring experiences",
        "tailor_summary": "Crafting summary",
        "assemble_cv": "Assembling CV",
    }
    step_timings: dict[str, float] = {}
    last_elapsed: float = 0.0
    start_time = time.time()

    def progress_callback(step_name: str, _description: str, elapsed: float) -> None:
        nonlocal last_elapsed

        # Calculate this step's duration from elapsed time
        step_time = elapsed - last_elapsed
        last_elapsed = elapsed

        # Record and display
        step_timings[step_name] = step_time
        label = step_labels.get(step_name, step_name)
        console.print(f"  [green]OK[/green] {label:<25} [dim][{format_time(step_time)}][/dim]")

    # Run the tailoring workflow with progress callback
    console.print()  # Blank line before progress
    try:
        result = run_cv_tailoring(
            raw_cv=raw_cv,
            raw_job_spec=raw_job_spec,
            provider=provider,
            lookback_years=lookback_years,
            use_rlm=not no_rlm,
            progress_callback=progress_callback,
        )

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    # Show total time
    total_time = time.time() - start_time
    console.print(f"\n[bold]Total time:[/bold] {format_time(total_time)}")

    # Show RLM metadata if used
    if not no_rlm and result.get("rlm_metadata"):
        rlm_meta = result["rlm_metadata"]
        if rlm_meta.get("used"):
            console.print(
                f"[dim]RLM: {rlm_meta['total_iterations']} iterations, "
                f"{rlm_meta['sub_call_count']} sub-calls, "
                f"{rlm_meta['execution_time_seconds']:.1f}s[/dim]"
            )

    # Check for errors
    if result.get("errors"):
        console.print("[red]Errors occurred:[/red]")
        for error in result["errors"]:
            console.print(f"  - {error}")
        raise typer.Exit(1)

    # Save output
    if result.get("tailored_cv"):
        save_markdown(result["tailored_cv"], output)
        console.print(f"\n[green]Tailored CV saved to:[/green] {output}")

        # Show match analysis
        if result.get("match_analysis"):
            match = result["match_analysis"]
            score = match["relevance_score"]
            score_color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"

            # Check if this is a hybrid score with breakdown
            if match.get("score_breakdown"):
                breakdown = match["score_breakdown"]
                knockout = match.get("knockout_triggered", False)

                console.print(
                    f"\n[bold]Match Score:[/bold] [{score_color}]{score:.0%}[/{score_color}]"
                )

                if knockout:
                    console.print(
                        f"[red]⚠ Knockout:[/red] {match.get('knockout_reason', 'Missing required skills')}"
                    )
                else:
                    console.print("[dim]Score Breakdown:[/dim]")
                    console.print(f"  Skills (exact):    {breakdown['exact_skill_match']:.0%}")
                    console.print(f"  Skills (semantic): {breakdown['semantic_skill_match']:.0%}")
                    console.print(f"  Doc similarity:    {breakdown['document_similarity']:.0%}")
                    console.print(f"  Experience fit:    {breakdown['experience_years_fit']:.0%}")
                    console.print(f"  Education:         {breakdown['education_match']:.0%}")
                    console.print(f"  Recency:           {breakdown['recency_score']:.0%}")

                    algo_score = match.get("algorithmic_score", 0)
                    llm_adj = match.get("llm_adjustment", 0)
                    if llm_adj != 0:
                        adj_sign = "+" if llm_adj > 0 else ""
                        console.print(
                            f"  [dim]Algorithmic: {algo_score:.0%}, LLM adjust: {adj_sign}{llm_adj:.0%}[/dim]"
                        )
            else:
                # Simple LLM-only score
                console.print(
                    f"\n[bold]Match Score:[/bold] [{score_color}]{score:.0%}[/{score_color}]"
                )
    else:
        console.print("[red]No tailored CV was generated.[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    cv: Annotated[Path, typer.Argument(help="Path to your CV")],
    job: Annotated[Path, typer.Argument(help="Path to job specification")],
    provider: Annotated[
        Literal["openai", "anthropic", "google"],
        typer.Option("--provider", "-p", help="LLM provider (model auto-selected)"),
    ] = "anthropic",
    no_rlm: Annotated[
        bool,
        typer.Option(
            "--no-rlm",
            help="Disable RLM (Recursive Language Model) for large document handling",
        ),
    ] = False,
) -> None:
    """Analyze CV-job fit without generating a tailored CV."""
    console.print(
        Panel.fit(
            "[bold blue]CV Warlock[/bold blue] - Analyzing CV-Job Fit",
            border_style="blue",
        )
    )

    # Read input files
    raw_cv = read_file(cv)
    raw_job_spec = read_file(job)

    # Step descriptions for display
    step_labels = {
        "validate_inputs": "Initializing",
        "extract_cv": "Extracting CV",
        "extract_job": "Analyzing job",
        "analyze_match": "Matching profile",
        "create_plan": "Creating plan",
        "tailor_skills": "Tailoring skills",
        "tailor_experiences": "Tailoring experiences",
        "tailor_summary": "Crafting summary",
        "assemble_cv": "Assembling CV",
    }
    step_timings: dict[str, float] = {}
    last_elapsed: float = 0.0
    start_time = time.time()

    def progress_callback(step_name: str, _description: str, elapsed: float) -> None:
        nonlocal last_elapsed

        # Calculate this step's duration from elapsed time
        step_time = elapsed - last_elapsed
        last_elapsed = elapsed

        # Record and display
        step_timings[step_name] = step_time
        label = step_labels.get(step_name, step_name)
        console.print(f"  [green]OK[/green] {label:<25} [dim][{format_time(step_time)}][/dim]")

    # Run analysis with progress callback
    console.print()  # Blank line before progress
    try:
        result = run_cv_tailoring(
            raw_cv=raw_cv,
            raw_job_spec=raw_job_spec,
            provider=provider,
            use_rlm=not no_rlm,
            progress_callback=progress_callback,
        )

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    # Show total time
    total_time = time.time() - start_time
    console.print(f"\n[bold]Total time:[/bold] {format_time(total_time)}")

    # Show RLM metadata if used
    if not no_rlm and result.get("rlm_metadata"):
        rlm_meta = result["rlm_metadata"]
        if rlm_meta.get("used"):
            console.print(
                f"[dim]RLM: {rlm_meta['total_iterations']} iterations, "
                f"{rlm_meta['sub_call_count']} sub-calls, "
                f"{rlm_meta['execution_time_seconds']:.1f}s[/dim]"
            )

    # Check for errors
    if result.get("errors"):
        console.print("[red]Errors occurred:[/red]")
        for error in result["errors"]:
            console.print(f"  - {error}")
        raise typer.Exit(1)

    # Display match analysis
    if result.get("match_analysis"):
        console.print()
        analysis_text = format_match_analysis(result["match_analysis"])
        console.print(Panel(analysis_text, title="Match Analysis", border_style="blue"))
    else:
        console.print("[yellow]No match analysis available.[/yellow]")


@app.command()
def version() -> None:
    """Show version information."""
    from cv_warlock import __version__

    console.print(f"CV Warlock v{__version__}")


if __name__ == "__main__":
    app()
