"""Main LangGraph workflow assembly."""

from typing import Literal

from langgraph.graph import StateGraph, START, END

from cv_warlock.config import get_settings
from cv_warlock.llm.base import get_llm_provider
from cv_warlock.models.state import CVWarlockState
from cv_warlock.graph.nodes import create_nodes
from cv_warlock.graph.edges import (
    should_continue_after_validation,
    should_continue_after_extraction,
)


def create_cv_warlock_graph(
    provider: Literal["openai", "anthropic"] | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> StateGraph:
    """Create and compile the CV tailoring workflow graph.

    Args:
        provider: LLM provider to use (openai or anthropic).
        model: Model name to use.
        api_key: API key for the provider.

    Returns:
        Compiled StateGraph.
    """
    settings = get_settings()

    # Use provided values or fall back to settings
    provider = provider or settings.provider
    model = model or settings.model

    # Get API key from settings if not provided
    if api_key is None:
        if provider == "openai":
            api_key = settings.openai_api_key
        else:
            api_key = settings.anthropic_api_key

    # Create LLM provider and nodes
    llm_provider = get_llm_provider(provider, model, api_key)
    nodes = create_nodes(llm_provider)

    # Build the graph
    workflow = StateGraph(CVWarlockState)

    # Add all nodes
    workflow.add_node("validate_inputs", nodes["validate_inputs"])
    workflow.add_node("extract_cv", nodes["extract_cv"])
    workflow.add_node("extract_job", nodes["extract_job"])
    workflow.add_node("analyze_match", nodes["analyze_match"])
    workflow.add_node("create_plan", nodes["create_plan"])
    workflow.add_node("tailor_summary", nodes["tailor_summary"])
    workflow.add_node("tailor_experiences", nodes["tailor_experiences"])
    workflow.add_node("tailor_skills", nodes["tailor_skills"])
    workflow.add_node("assemble_cv", nodes["assemble_cv"])

    # Define edges
    workflow.add_edge(START, "validate_inputs")

    # Conditional edge after validation
    workflow.add_conditional_edges(
        "validate_inputs",
        should_continue_after_validation,
        {
            "continue": "extract_cv",
            "error": END,
        },
    )

    # Sequential extraction
    workflow.add_edge("extract_cv", "extract_job")

    # Conditional edge after extraction
    workflow.add_conditional_edges(
        "extract_job",
        should_continue_after_extraction,
        {
            "continue": "analyze_match",
            "error": END,
        },
    )

    # Sequential tailoring pipeline
    workflow.add_edge("analyze_match", "create_plan")
    workflow.add_edge("create_plan", "tailor_summary")
    workflow.add_edge("tailor_summary", "tailor_experiences")
    workflow.add_edge("tailor_experiences", "tailor_skills")
    workflow.add_edge("tailor_skills", "assemble_cv")
    workflow.add_edge("assemble_cv", END)

    return workflow.compile()


def run_cv_tailoring(
    raw_cv: str,
    raw_job_spec: str,
    provider: Literal["openai", "anthropic"] | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> CVWarlockState:
    """Run the CV tailoring workflow.

    Args:
        raw_cv: Raw CV text.
        raw_job_spec: Raw job specification text.
        provider: LLM provider to use.
        model: Model name to use.
        api_key: API key for the provider.

    Returns:
        Final workflow state with tailored CV.
    """
    graph = create_cv_warlock_graph(provider, model, api_key)

    initial_state: CVWarlockState = {
        "raw_cv": raw_cv,
        "raw_job_spec": raw_job_spec,
        "cv_data": None,
        "job_requirements": None,
        "match_analysis": None,
        "tailoring_plan": None,
        "tailored_summary": None,
        "tailored_experiences": None,
        "tailored_skills": None,
        "tailored_cv": None,
        "messages": [],
        "current_step": "start",
        "errors": [],
    }

    result = graph.invoke(initial_state)
    return result
