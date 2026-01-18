"""CLI entry point for CV Warlock."""

from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv

# Load environment variables from .env.local (including LangSmith config)
# Path: main.py -> cv_warlock/ -> src/ -> project root
load_dotenv(Path(__file__).parent.parent.parent / ".env.local")

import typer  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import Progress, SpinnerColumn, TextColumn  # noqa: E402

from cv_warlock.graph.workflow import run_cv_tailoring  # noqa: E402
from cv_warlock.output.markdown import format_match_analysis, save_markdown  # noqa: E402

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
        Literal["openai", "anthropic"],
        typer.Option("--provider", "-p", help="LLM provider to use"),
    ] = "openai",
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model name")] = None,
    lookback_years: Annotated[
        int | None,
        typer.Option(
            "--lookback-years",
            "-l",
            help="Only tailor jobs ending within N years (default: 4)",
        ),
    ] = None,
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
        console.print(f"[dim]Provider:[/dim] {provider}")
        console.print(f"[dim]Model:[/dim] {model or 'default'}")
        console.print(f"[dim]Lookback:[/dim] {lookback_years or 4} years")
        console.print()

    # Run the tailoring workflow
    # Use ASCII spinner for Windows compatibility
    with Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=None)

        try:
            result = run_cv_tailoring(
                raw_cv=raw_cv,
                raw_job_spec=raw_job_spec,
                provider=provider,
                model=model,
                lookback_years=lookback_years,
            )
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from e

        progress.update(task, completed=True)

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
        Literal["openai", "anthropic"],
        typer.Option("--provider", "-p", help="LLM provider to use"),
    ] = "openai",
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model name")] = None,
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

    # Run analysis (the full workflow, but we'll just show analysis)
    # Use ASCII spinner for Windows compatibility
    with Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing...", total=None)

        try:
            result = run_cv_tailoring(
                raw_cv=raw_cv,
                raw_job_spec=raw_job_spec,
                provider=provider,
                model=model,
            )
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from e

        progress.update(task, completed=True)

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
