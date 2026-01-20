"""Main LangGraph workflow assembly.

Supports chain-of-thought reasoning mode for higher quality generation.
When CoT is enabled, generation is slower (3-4x more LLM calls) but produces
significantly better tailored CVs.

Supports RLM (Recursive Language Model) mode for handling arbitrarily long
CVs and job specs through code-based context exploration and sub-model calls.
"""

import time
from collections.abc import Callable
from typing import Literal

from langgraph.graph import END, START, StateGraph

from cv_warlock.config import get_settings
from cv_warlock.graph.edges import (
    should_continue_after_extraction,
    should_continue_after_validation,
)
from cv_warlock.graph.nodes import create_nodes
from cv_warlock.llm.base import get_llm_provider
from cv_warlock.models.state import CVWarlockState


def _get_default_model_for_provider(provider: str) -> str:
    """Get the default/recommended model for a given provider.

    Returns the best balanced model for general use (non-RLM mode).
    """
    if provider == "anthropic":
        return "claude-sonnet-4-5-20250929"
    elif provider == "openai":
        return "gpt-5.2"
    elif provider == "google":
        return "gemini-3-flash-preview"
    else:
        return "claude-sonnet-4-5-20250929"


def _get_strong_model_for_provider(provider: str) -> str:
    """Get the strongest/most capable model for a given provider.

    Used for RLM root orchestration where maximum capability is needed.
    Note: For Anthropic, uses Sonnet (not Opus) per project policy.
    """
    if provider == "anthropic":
        return "claude-sonnet-4-5-20250929"
    elif provider == "openai":
        return "gpt-5.2"
    elif provider == "google":
        return "gemini-3-pro-preview"
    else:
        return "claude-sonnet-4-5-20250929"


def _get_fast_model_for_provider(provider: str) -> str:
    """Get the fast/efficient model for a given provider.

    Used for RLM sub-calls where a faster model is preferred.
    """
    if provider == "anthropic":
        return "claude-haiku-4-5-20251001"
    elif provider == "openai":
        return "gpt-5-mini"
    elif provider == "google":
        return "gemini-3-flash-preview"
    else:
        return "claude-haiku-4-5-20251001"


def create_cv_warlock_graph(
    provider: Literal["openai", "anthropic", "google"] | None = None,
    api_key: str | None = None,
    use_cot: bool = True,
    use_rlm: bool = False,
    on_step_start: Callable[[str, str], None] | None = None,
) -> StateGraph:
    """Create and compile the CV tailoring workflow graph.

    Args:
        provider: LLM provider to use (openai, anthropic, or google).
        api_key: API key for the provider.
        use_cot: Whether to use chain-of-thought reasoning for generation.
                 Default True for higher quality, False for faster generation.
        use_rlm: Whether to use RLM for large context handling.
                 Default False. When True, uses recursive orchestration
                 for documents exceeding the size threshold.
        on_step_start: Optional callback(step_name, description) fired when a step starts.

    Returns:
        Compiled StateGraph.

    Note:
        Model selection is automatic based on provider (Dual-Model Strategy):
        - Default model: Best balanced model for the provider
        - RLM sub-calls: Fastest model for the provider
    """
    settings = get_settings()

    # Use provided provider or fall back to settings
    provider = provider or settings.provider
    # Auto-select model based on provider (Dual-Model Strategy)
    model = _get_default_model_for_provider(provider)

    # Get API key from settings if not provided
    if api_key is None:
        if provider == "openai":
            api_key = settings.openai_api_key
        elif provider == "google":
            api_key = settings.google_api_key
        else:
            api_key = settings.anthropic_api_key

    # Create nodes - use RLM nodes if enabled
    if use_rlm:
        from cv_warlock.graph.rlm_nodes import create_rlm_nodes
        from cv_warlock.rlm.models import RLMConfig

        # Dual-Model Strategy for RLM:
        # - Root orchestration: strongest model (Opus, GPT-5.2, Gemini Pro)
        # - Sub-calls: fastest model (Haiku, GPT-5-mini, Gemini Flash)
        root_model = _get_strong_model_for_provider(provider)
        sub_model = _get_fast_model_for_provider(provider)

        # Create root provider with strongest model for orchestration
        llm_provider = get_llm_provider(provider, root_model, api_key)

        rlm_config = RLMConfig(
            root_provider=provider,
            root_model=root_model,
            sub_provider=provider,
            sub_model=sub_model,
            max_iterations=settings.rlm_max_iterations,
            max_sub_calls=settings.rlm_max_sub_calls,
            timeout_seconds=settings.rlm_timeout_seconds,
            size_threshold=settings.rlm_size_threshold,
            sandbox_mode=settings.rlm_sandbox_mode,
        )

        # Create sub-provider with fastest model for sub-calls
        sub_provider_instance = get_llm_provider(provider, sub_model, api_key)

        nodes = create_rlm_nodes(
            root_provider=llm_provider,
            sub_provider=sub_provider_instance,
            config=rlm_config,
            use_cot=use_cot,
            on_step_start=on_step_start,
        )
    else:
        # Non-RLM mode: Dual-Model Strategy
        # - Extraction/analysis: balanced model (Sonnet, GPT-5.2, Gemini Flash) for quality
        # - Tailoring: fast model (Haiku, GPT-5-mini, Gemini Flash) for speed
        llm_provider = get_llm_provider(provider, model, api_key)
        fast_model = _get_fast_model_for_provider(provider)
        tailor_provider = get_llm_provider(provider, fast_model, api_key)
        nodes = create_nodes(
            llm_provider,
            use_cot=use_cot,
            on_step_start=on_step_start,
            tailor_provider=tailor_provider,
        )

    # Build the graph
    workflow = StateGraph(CVWarlockState)

    # Add all nodes
    workflow.add_node("validate_inputs", nodes["validate_inputs"])
    workflow.add_node("extract_all", nodes["extract_all"])  # Parallel CV + job extraction
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
            "continue": "extract_all",  # Single parallel extraction node
            "error": END,
        },
    )

    # Conditional edge after extraction
    workflow.add_conditional_edges(
        "extract_all",
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
    api_key: str | None = None,
    progress_callback: Callable[[str, str, float], None] | None = None,
    assume_all_tech_skills: bool = True,
    use_cot: bool = True,
    use_rlm: bool = False,
    lookback_years: int | None = None,
) -> CVWarlockState:
    """Run the CV tailoring workflow.

    Args:
        raw_cv: Raw CV text.
        raw_job_spec: Raw job specification text.
        provider: LLM provider to use. Model is auto-selected based on provider.
        api_key: API key for the provider.
        progress_callback: Optional callback function(step_name, description, elapsed_seconds)
                          for progress updates with timing.
        assume_all_tech_skills: If True, assumes user has all tech skills from job spec.
        use_cot: If True, uses chain-of-thought reasoning for higher quality (slower).
                 If False, uses direct generation (faster but lower quality).
        use_rlm: If True, uses RLM for large context handling.
                 Enables recursive orchestration for long documents.
        lookback_years: Only tailor jobs ending within this many years. If None, uses
                       settings default (4 years).

    Returns:
        Final workflow state with tailored CV.
    """
    # Create callback adapter for immediate step start notifications
    on_step_start = None
    if progress_callback:
        start_tracking = {}

        def _on_step_start(step_name: str, description: str):
            start_tracking[step_name] = time.time()
            # Pass 0.0 elapsed time for start event
            progress_callback(step_name, description, 0.0)

        on_step_start = _on_step_start

    graph = create_cv_warlock_graph(
        provider,
        api_key,
        use_cot=use_cot,
        use_rlm=use_rlm,
        on_step_start=on_step_start,
    )

    # Step descriptions for progress updates
    # New order: skills → experiences → summary
    # Experience tailoring runs in parallel for significant time savings
    if use_cot:
        step_descriptions = {
            "validate_inputs": "Initializing workflow...",
            "extract_all": "Extracting CV + job in parallel...",
            "analyze_match": "Matching your profile to requirements...",
            "create_plan": "Creating tailoring strategy...",
            "tailor_skills": "Adding job skills to CV (reasoning → generating)...",
            "tailor_experiences": "Tailoring recent experiences in parallel...",
            "tailor_summary": "Crafting summary from tailored content...",
            "assemble_cv": "Assembling final CV...",
        }
    else:
        step_descriptions = {
            "validate_inputs": "Initializing workflow...",
            "extract_all": "Extracting CV + job in parallel...",
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
        "use_rlm": use_rlm,
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
        "rlm_metadata": None,
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
