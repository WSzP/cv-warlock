"""Main LangGraph workflow assembly.

Supports chain-of-thought reasoning mode for higher quality generation.
When CoT is enabled, generation is slower (3-4x more LLM calls) but produces
significantly better tailored CVs.
"""

import time
from typing import Callable, Literal

from langgraph.graph import END, START, StateGraph

from cv_warlock.config import get_settings
from cv_warlock.graph.edges import (
    should_continue_after_extraction,
    should_continue_after_validation,
)
from cv_warlock.graph.nodes import create_nodes
from cv_warlock.llm.base import get_llm_provider
from cv_warlock.models.state import CVWarlockState


def create_cv_warlock_graph(
    provider: Literal["openai", "anthropic", "google"] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    use_cot: bool = True,
    temperature: float | None = None,
) -> StateGraph:
    """Create and compile the CV tailoring workflow graph.

    Args:
        provider: LLM provider to use (openai, anthropic, or google).
        model: Model name to use.
        api_key: API key for the provider.
        use_cot: Whether to use chain-of-thought reasoning for generation.
                 Default True for higher quality, False for faster generation.
        temperature: Model temperature (0.0-1.0). If None, uses settings default.

    Returns:
        Compiled StateGraph.
    """
    settings = get_settings()

    # Use provided values or fall back to settings
    provider = provider or settings.provider
    model = model or settings.model
    temperature = temperature if temperature is not None else settings.temperature

    # Get API key from settings if not provided
    if api_key is None:
        if provider == "openai":
            api_key = settings.openai_api_key
        elif provider == "google":
            api_key = settings.google_api_key
        else:
            api_key = settings.anthropic_api_key

    # Create LLM provider and nodes
    llm_provider = get_llm_provider(provider, model, api_key, temperature=temperature)
    nodes = create_nodes(llm_provider, use_cot=use_cot)

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

    # Sequential tailoring pipeline (new order: skills → experiences → summary)
    workflow.add_edge("analyze_match", "create_plan")
    workflow.add_edge("create_plan", "tailor_skills")  # Skills FIRST
    workflow.add_edge("tailor_skills", "tailor_experiences")  # Experiences second
    workflow.add_edge("tailor_experiences", "tailor_summary")  # Summary LAST
    workflow.add_edge("tailor_summary", "assemble_cv")
    workflow.add_edge("assemble_cv", END)

    return workflow.compile()


def run_cv_tailoring(
    raw_cv: str,
    raw_job_spec: str,
    provider: Literal["openai", "anthropic", "google"] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    progress_callback: Callable[[str, str, float], None] | None = None,
    assume_all_tech_skills: bool = True,
    use_cot: bool = True,
    temperature: float | None = None,
    lookback_years: int | None = None,
) -> CVWarlockState:
    """Run the CV tailoring workflow.

    Args:
        raw_cv: Raw CV text.
        raw_job_spec: Raw job specification text.
        provider: LLM provider to use.
        model: Model name to use.
        api_key: API key for the provider.
        progress_callback: Optional callback function(step_name, description, elapsed_seconds)
                          for progress updates with timing.
        assume_all_tech_skills: If True, assumes user has all tech skills from job spec.
        use_cot: If True, uses chain-of-thought reasoning for higher quality (slower).
                 If False, uses direct generation (faster but lower quality).
        temperature: Model temperature (0.0-1.0). If None, uses settings default.
        lookback_years: Only tailor jobs ending within this many years. If None, uses
                       settings default (4 years).

    Returns:
        Final workflow state with tailored CV.
    """
    graph = create_cv_warlock_graph(
        provider, model, api_key, use_cot=use_cot, temperature=temperature
    )

    # Step descriptions for progress updates
    # New order: skills → experiences → summary
    # Experience tailoring runs in parallel for significant time savings
    if use_cot:
        step_descriptions = {
            "validate_inputs": "Validating inputs...",
            "extract_cv": "Extracting CV structure...",
            "extract_job": "Analyzing job requirements...",
            "analyze_match": "Matching your profile to requirements...",
            "create_plan": "Creating tailoring strategy...",
            "tailor_skills": "Adding job skills to CV (reasoning → generating)...",
            "tailor_experiences": "Tailoring recent experiences in parallel...",
            "tailor_summary": "Crafting summary from tailored content...",
            "assemble_cv": "Assembling final CV...",
        }
    else:
        step_descriptions = {
            "validate_inputs": "Validating inputs...",
            "extract_cv": "Extracting CV structure...",
            "extract_job": "Analyzing job requirements...",
            "analyze_match": "Matching your profile to requirements...",
            "create_plan": "Creating tailoring strategy...",
            "tailor_skills": "Adding job skills to CV...",
            "tailor_experiences": "Tailoring recent work experiences...",
            "tailor_summary": "Crafting professional summary...",
            "assemble_cv": "Assembling final CV...",
        }

    initial_state: CVWarlockState = {
        "raw_cv": raw_cv,
        "raw_job_spec": raw_job_spec,
        "assume_all_tech_skills": assume_all_tech_skills,
        "use_cot": use_cot,
        "lookback_years": lookback_years,
        "cv_data": None,
        "job_requirements": None,
        "match_analysis": None,
        "tailoring_plan": None,
        "tailored_summary": None,
        "tailored_experiences": None,
        "tailored_skills": None,
        "tailored_cv": None,
        "summary_reasoning_result": None,
        "experience_reasoning_results": None,
        "skills_reasoning_result": None,
        "generation_context": None,
        "total_refinement_iterations": 0,
        "quality_scores": None,
        "step_timings": [],
        "current_step_start": None,
        "total_generation_time": None,
        "messages": [],
        "current_step": "start",
        "current_step_description": "Initializing...",
        "errors": [],
    }

    start_time = time.time()

    if progress_callback:
        # Use streaming to get node-by-node updates with timing
        final_state = dict(initial_state)
        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                elapsed = time.time() - start_time
                if node_name in step_descriptions:
                    # Get description from output if available, otherwise use default
                    desc = (
                        node_output.get("current_step_description")
                        if isinstance(node_output, dict)
                        else None
                    ) or step_descriptions.get(node_name, f"Running {node_name}...")
                    progress_callback(node_name, desc, elapsed)
                # Merge node output into state
                if isinstance(node_output, dict):
                    final_state.update(node_output)

        # Final timing
        final_state["total_generation_time"] = time.time() - start_time
        return final_state
    else:
        result = graph.invoke(initial_state)
        result["total_generation_time"] = time.time() - start_time
        return result
