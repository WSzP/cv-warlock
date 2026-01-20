"""LangGraph node definitions for CV tailoring workflow.

Supports chain-of-thought reasoning mode for higher quality generation.
When CoT is enabled, generation is slower (3-4x more LLM calls) but produces
significantly better tailored CVs.
"""

import copy
import time
from collections.abc import Callable

from cv_warlock.config import get_settings
from cv_warlock.extractors.cv_extractor import CVExtractor
from cv_warlock.extractors.job_extractor import JobExtractor
from cv_warlock.llm.base import LLMProvider
from cv_warlock.models.reasoning import GenerationContext
from cv_warlock.models.state import CVWarlockState, StepTiming
from cv_warlock.processors.matcher import MatchAnalyzer
from cv_warlock.processors.tailor import CVTailor

# Step descriptions for UI display
# Optimized pipeline: parallel skills + experiences → summary
STEP_DESCRIPTIONS = {
    "validate_inputs": "Initializing workflow...",
    "extract_cv": "Extracting CV data...",
    "extract_job": "Analyzing job requirements...",
    "extract_all": "Extracting CV and job data in parallel...",
    "analyze_match": "Analyzing CV-job match...",
    "create_plan": "Creating tailoring strategy...",
    "tailor_skills": "Adding job skills to CV (reasoning → generating)...",
    "tailor_experiences": "Tailoring recent experiences in parallel...",
    "tailor_skills_and_experiences": "Tailoring skills + experiences in parallel...",
    "tailor_summary": "Crafting summary from tailored content...",
    "assemble_cv": "Assembling final CV...",
}

STEP_DESCRIPTIONS_FAST = {
    "validate_inputs": "Initializing workflow...",
    "extract_cv": "Extracting CV data...",
    "extract_job": "Analyzing job requirements...",
    "extract_all": "Extracting CV and job data in parallel...",
    "analyze_match": "Analyzing CV-job match...",
    "create_plan": "Creating tailoring strategy...",
    "tailor_skills": "Adding job skills to CV...",
    "tailor_experiences": "Tailoring recent work experiences...",
    "tailor_skills_and_experiences": "Tailoring skills + experiences in parallel...",
    "tailor_summary": "Crafting professional summary...",
    "assemble_cv": "Assembling final CV...",
}


def _start_step(
    state: CVWarlockState,
    step_name: str,
    use_cot: bool = True,
    on_step_start: Callable[[str, str], None] | None = None,
) -> dict:
    """Record step start time and description."""
    descriptions = STEP_DESCRIPTIONS if use_cot else STEP_DESCRIPTIONS_FAST
    description = descriptions.get(step_name, f"Running {step_name}...")

    # Trigger callback if provided (for immediate UI updates)
    if on_step_start:
        try:
            on_step_start(step_name, description)
        except Exception:
            # Don't let UI callback errors break the workflow
            pass

    return {
        "current_step": step_name,
        "current_step_description": description,
        "current_step_start": time.time(),
    }


def _end_step(state: CVWarlockState, step_name: str, updates: dict) -> dict:
    """Record step end time and compute duration."""
    end_time = time.time()
    start_time = state.get("current_step_start")

    timing = StepTiming(
        step_name=step_name,
        start_time=start_time or end_time,
        end_time=end_time,
        duration_seconds=end_time - start_time if start_time else 0,
    )

    existing_timings = state.get("step_timings", [])
    updates["step_timings"] = existing_timings + [timing]
    # We don't clear current_step_start here so it persists until next step starts
    # updates["current_step_start"] = None

    return updates


def create_nodes(
    llm_provider: LLMProvider,
    use_cot: bool = True,
    on_step_start: Callable[[str, str], None] | None = None,
    tailor_provider: LLMProvider | None = None,
    extraction_provider: LLMProvider | None = None,
) -> dict:
    """Create all workflow nodes with the given LLM provider.

    Args:
        llm_provider: The LLM provider to use for analysis (match scoring, planning).
        use_cot: Whether to use chain-of-thought reasoning for generation.
                 Default True for higher quality, False for faster generation.
        on_step_start: Optional callback(step_name, description) fired when a step starts.
        tailor_provider: Optional faster provider for tailoring steps (skills, experiences,
                        summary). If not provided, uses llm_provider for all steps.
        extraction_provider: Optional fast provider for extraction (CV + job parsing).
                            Extraction is pattern-matching, not reasoning, so a fast model
                            like Haiku is ideal. If not provided, uses tailor_provider.

    Returns:
        dict: Dictionary of node functions.
    """
    # Use fast provider for tailoring if provided, otherwise use main provider
    fast_provider = tailor_provider or llm_provider
    # Use extraction provider if provided, otherwise use fast_provider (Haiku)
    # Extraction is pattern matching - doesn't need Sonnet-level reasoning
    extract_provider = extraction_provider or fast_provider

    cv_extractor = CVExtractor(extract_provider)
    job_extractor = JobExtractor(extract_provider)
    match_analyzer = MatchAnalyzer(llm_provider)
    cv_tailor = CVTailor(fast_provider, use_cot=use_cot)

    def validate_inputs(state: CVWarlockState) -> dict:
        """Validate input documents exist and are non-empty."""
        step_name = "validate_inputs"
        result = _start_step(state, step_name, use_cot, on_step_start)
        errors = []

        if not state.get("raw_cv") or not state["raw_cv"].strip():
            errors.append("CV document is empty or missing")

        if not state.get("raw_job_spec") or not state["raw_job_spec"].strip():
            errors.append("Job specification is empty or missing")

        result["errors"] = state.get("errors", []) + errors

        # Initialize timing list if needed
        if "step_timings" not in state:
            result["step_timings"] = []

        # Initialize total refinement counter
        result["total_refinement_iterations"] = 0

        return _end_step(state, step_name, result)

    def extract_cv(state: CVWarlockState) -> dict:
        """Extract structured data from CV."""
        step_name = "extract_cv"
        result = _start_step(state, step_name, use_cot, on_step_start)

        try:
            cv_data = cv_extractor.extract(state["raw_cv"])
            result["cv_data"] = cv_data
        except Exception as e:
            result["errors"] = state.get("errors", []) + [f"CV extraction failed: {e!s}"]

        return _end_step(state, step_name, result)

    def extract_job(state: CVWarlockState) -> dict:
        """Extract structured data from job specification."""
        step_name = "extract_job"
        result = _start_step(state, step_name, use_cot, on_step_start)

        try:
            job_requirements = job_extractor.extract(state["raw_job_spec"])
            result["job_requirements"] = job_requirements
        except Exception as e:
            result["errors"] = state.get("errors", []) + [f"Job extraction failed: {e!s}"]

        return _end_step(state, step_name, result)

    def extract_all(state: CVWarlockState) -> dict:
        """Extract CV and job data in parallel for faster processing."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        step_name = "extract_all"
        result = _start_step(state, step_name, use_cot, on_step_start)
        errors: list[str] = []

        def extract_cv_task():
            return cv_extractor.extract(state["raw_cv"])

        def extract_job_task():
            return job_extractor.extract(state["raw_job_spec"])

        cv_data = None
        job_requirements = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            cv_future = executor.submit(extract_cv_task)
            job_future = executor.submit(extract_job_task)

            for future in as_completed([cv_future, job_future]):
                try:
                    if future == cv_future:
                        cv_data = future.result()
                    else:
                        job_requirements = future.result()
                except Exception as e:
                    if future == cv_future:
                        errors.append(f"CV extraction failed: {e!s}")
                    else:
                        errors.append(f"Job extraction failed: {e!s}")

        if cv_data:
            result["cv_data"] = cv_data
        if job_requirements:
            result["job_requirements"] = job_requirements
        if errors:
            result["errors"] = state.get("errors", []) + errors

        return _end_step(state, step_name, result)

    def analyze_match(state: CVWarlockState) -> dict:
        """Analyze match between CV and job requirements.

        Uses hybrid scoring combining algorithmic sub-scores (exact string matching,
        experience years, education, recency) with LLM qualitative assessment.

        OPTIMIZATION: Uses score_with_plan() to get both match analysis and tailoring
        plan in a single LLM call, saving ~10-20 seconds.

        If assume_all_tech_skills is True, augments CV skills with all required
        technical skills from the job spec before analysis. The augmented CV
        is persisted in state so downstream nodes use the enhanced skill list.
        """
        step_name = "analyze_match"
        result = _start_step(state, step_name, use_cot, on_step_start)

        if state.get("errors"):
            return _end_step(state, step_name, result)

        try:
            cv_data = state["cv_data"]
            job_requirements = state["job_requirements"]

            # If assume_all_tech_skills is enabled, augment CV skills with job requirements
            # OPTIMIZATION: Use lightweight skill augmentation instead of deep copy
            if state.get("assume_all_tech_skills", True):
                existing_skills = {s.lower() for s in cv_data.skills}
                skills_to_add = [
                    skill
                    for skill in job_requirements.required_skills
                    + job_requirements.preferred_skills
                    if skill.lower() not in existing_skills
                ]
                if skills_to_add:
                    # Only copy if we need to add skills
                    cv_data = copy.deepcopy(cv_data)
                    cv_data.skills.extend(skills_to_add)
                    result["cv_data"] = cv_data

            # OPTIMIZATION: Use combined scoring + plan in single LLM call
            from cv_warlock.scoring.hybrid import HybridScorer

            hybrid_scorer = HybridScorer(llm_provider)
            match_analysis, tailoring_plan = hybrid_scorer.score_with_plan(
                cv_data, job_requirements
            )

            result["match_analysis"] = match_analysis
            result["tailoring_plan"] = tailoring_plan  # Pre-compute for create_plan node
        except Exception as e:
            result["errors"] = state.get("errors", []) + [f"Match analysis failed: {e!s}"]

        return _end_step(state, step_name, result)

    def create_plan(state: CVWarlockState) -> dict:
        """Create tailoring plan based on match analysis.

        OPTIMIZATION: If tailoring_plan was already computed by analyze_match
        (via score_with_plan), this node just passes it through.
        """
        step_name = "create_plan"
        result = _start_step(state, step_name, use_cot, on_step_start)

        if state.get("errors"):
            return _end_step(state, step_name, result)

        try:
            # Check if plan was already computed by analyze_match (optimization)
            if state.get("tailoring_plan") is not None:
                result["tailoring_plan"] = state["tailoring_plan"]
            else:
                # Fallback to separate LLM call if not pre-computed
                tailoring_plan = match_analyzer.create_tailoring_plan(
                    state["cv_data"],
                    state["job_requirements"],
                    state["match_analysis"],
                )
                result["tailoring_plan"] = tailoring_plan
        except Exception as e:
            result["errors"] = state.get("errors", []) + [f"Tailoring plan failed: {e!s}"]

        return _end_step(state, step_name, result)

    def tailor_summary(state: CVWarlockState) -> dict:
        """Tailor the professional summary (LAST in pipeline - has full context)."""
        step_name = "tailor_summary"
        result = _start_step(state, step_name, use_cot, on_step_start)

        if state.get("errors"):
            return _end_step(state, step_name, result)

        # Summary is LAST: has access to tailored skills and experiences
        tailored_skills = (
            state.get("tailored_skills", [""])[0] if state.get("tailored_skills") else ""
        )

        try:
            if use_cot:
                # Full CoT with reasoning output
                # Summary receives context from skills and experiences
                cot_result = cv_tailor.tailor_summary_with_cot(
                    state["cv_data"],
                    state["job_requirements"],
                    state["tailoring_plan"],
                    context=state.get("generation_context"),
                    tailored_skills_preview=tailored_skills,
                )

                # Update context (summary is last, so just for completeness)
                context = state.get("generation_context") or GenerationContext()
                context.established_identity = cot_result.reasoning.hook_strategy
                context.key_metric_used = cot_result.reasoning.strongest_metric

                result["tailored_summary"] = cot_result.final_summary
                result["summary_reasoning_result"] = cot_result
                result["generation_context"] = context
                result["total_refinement_iterations"] = (
                    state.get("total_refinement_iterations", 0) + cot_result.refinement_count
                )
                result["quality_scores"] = {
                    **(state.get("quality_scores") or {}),
                    "summary": cot_result.critique.quality_level.value,
                }
            else:
                # Direct generation (faster)
                tailored = cv_tailor.tailor_summary(
                    state["cv_data"],
                    state["job_requirements"],
                    state["tailoring_plan"],
                    tailored_skills_preview=tailored_skills,
                )
                result["tailored_summary"] = tailored
        except Exception as e:
            result["errors"] = state.get("errors", []) + [f"Summary tailoring failed: {e!s}"]

        return _end_step(state, step_name, result)

    def tailor_experiences(state: CVWarlockState) -> dict:
        """Tailor experience entries with lookback filtering (SECOND in pipeline)."""
        step_name = "tailor_experiences"
        result = _start_step(state, step_name, use_cot, on_step_start)

        if state.get("errors"):
            return _end_step(state, step_name, result)

        # Get lookback_years from state or settings default
        lookback_years = state.get("lookback_years") or get_settings().lookback_years

        try:
            if use_cot:
                # Full CoT with reasoning outputs and lookback filtering
                tailored_texts, exp_results, updated_context = (
                    cv_tailor.tailor_experiences_with_cot(
                        state["cv_data"],
                        state["job_requirements"],
                        state["tailoring_plan"],
                        context=state.get("generation_context"),
                        lookback_years=lookback_years,
                    )
                )

                result["tailored_experiences"] = tailored_texts
                result["experience_reasoning_results"] = exp_results
                result["generation_context"] = updated_context

                # Accumulate refinement counts and quality scores
                total_refinements = sum(r.refinement_count for r in exp_results)
                result["total_refinement_iterations"] = (
                    state.get("total_refinement_iterations", 0) + total_refinements
                )

                quality_scores = state.get("quality_scores") or {}
                for i, exp_result in enumerate(exp_results):
                    quality_scores[f"experience_{i}"] = exp_result.critique.quality_level.value
                result["quality_scores"] = quality_scores
            else:
                # Direct generation with lookback filtering
                tailored = cv_tailor.tailor_experiences(
                    state["cv_data"],
                    state["job_requirements"],
                    state["tailoring_plan"],
                    lookback_years=lookback_years,
                )
                result["tailored_experiences"] = tailored
        except Exception as e:
            result["errors"] = state.get("errors", []) + [f"Experience tailoring failed: {e!s}"]

        return _end_step(state, step_name, result)

    def tailor_skills(state: CVWarlockState) -> dict:
        """Tailor the skills section."""
        step_name = "tailor_skills"
        result = _start_step(state, step_name, use_cot, on_step_start)

        if state.get("errors"):
            return _end_step(state, step_name, result)

        try:
            if use_cot:
                # Full CoT with reasoning output
                cot_result = cv_tailor.tailor_skills_with_cot(
                    state["cv_data"],
                    state["job_requirements"],
                    context=None,
                )

                # Initialize context for downstream sections (summary)
                context = GenerationContext(
                    skills_demonstrated=cot_result.reasoning.required_skills_matched
                    + cot_result.reasoning.preferred_skills_matched,
                    primary_keywords_used=cot_result.reasoning.required_skills_matched[:5],
                )

                result["tailored_skills"] = [cot_result.final_skills]
                result["skills_reasoning_result"] = cot_result
                result["generation_context"] = context
                result["total_refinement_iterations"] = (
                    state.get("total_refinement_iterations", 0) + cot_result.refinement_count
                )
                result["quality_scores"] = {
                    **(state.get("quality_scores") or {}),
                    "skills": cot_result.critique.quality_level.value,
                }
            else:
                # Direct generation (faster)
                tailored = cv_tailor.tailor_skills(
                    state["cv_data"],
                    state["job_requirements"],
                )
                result["tailored_skills"] = [tailored]
                result["generation_context"] = GenerationContext()
        except Exception as e:
            result["errors"] = state.get("errors", []) + [f"Skills tailoring failed: {e!s}"]

        return _end_step(state, step_name, result)

    def tailor_skills_and_experiences(state: CVWarlockState) -> dict:
        """Tailor skills and experiences in PARALLEL for faster processing.

        OPTIMIZATION: Runs skills and experiences concurrently using ThreadPoolExecutor.
        This saves ~15-20% wall-clock time since both can make LLM calls simultaneously.

        Note: Experiences won't have skills context, but the impact is minimal
        (slight keyword overlap) compared to the performance gain.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        step_name = "tailor_skills_and_experiences"
        result = _start_step(state, step_name, use_cot, on_step_start)

        if state.get("errors"):
            return _end_step(state, step_name, result)

        errors: list[str] = []
        skills_result = None
        experiences_result = None
        experience_cot_results = None

        def tailor_skills_task():
            if use_cot:
                return cv_tailor.tailor_skills_with_cot(
                    state["cv_data"],
                    state["job_requirements"],
                    context=None,
                )
            else:
                return cv_tailor.tailor_skills(
                    state["cv_data"],
                    state["job_requirements"],
                )

        def tailor_experiences_task():
            if use_cot:
                return cv_tailor.tailor_experiences_with_cot(
                    state["cv_data"],
                    state["job_requirements"],
                    state["tailoring_plan"],
                    context=None,  # No context since running in parallel with skills
                    lookback_years=state.get("lookback_years"),
                )
            else:
                return cv_tailor.tailor_experiences(
                    state["cv_data"],
                    state["job_requirements"],
                    state["tailoring_plan"],
                    lookback_years=state.get("lookback_years"),
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            skills_future = executor.submit(tailor_skills_task)
            exp_future = executor.submit(tailor_experiences_task)

            for future in as_completed([skills_future, exp_future]):
                try:
                    if future == skills_future:
                        skills_result = future.result()
                    else:
                        experiences_result = future.result()
                except Exception as e:
                    if future == skills_future:
                        errors.append(f"Skills tailoring failed: {e!s}")
                    else:
                        errors.append(f"Experience tailoring failed: {e!s}")

        # Process skills result
        if skills_result:
            if use_cot:
                context = GenerationContext(
                    skills_demonstrated=skills_result.reasoning.required_skills_matched
                    + skills_result.reasoning.preferred_skills_matched,
                    primary_keywords_used=skills_result.reasoning.required_skills_matched[:5],
                )
                result["tailored_skills"] = [skills_result.final_skills]
                result["skills_reasoning_result"] = skills_result
                result["generation_context"] = context
                result["total_refinement_iterations"] = (
                    state.get("total_refinement_iterations", 0) + skills_result.refinement_count
                )
                result["quality_scores"] = {
                    **(state.get("quality_scores") or {}),
                    "skills": skills_result.critique.quality_level.value,
                }
            else:
                result["tailored_skills"] = [skills_result]
                result["generation_context"] = GenerationContext()

        # Process experiences result
        if experiences_result:
            if use_cot:
                tailored_texts, cot_results, exp_context = experiences_result
                result["tailored_experiences"] = tailored_texts
                result["experience_reasoning_results"] = cot_results

                # Merge experience context into existing context
                if result.get("generation_context"):
                    result["generation_context"].metrics_used = exp_context.metrics_used
                    result["generation_context"].skills_demonstrated.extend(
                        exp_context.skills_demonstrated
                    )
                    result["generation_context"].keyword_frequency = exp_context.keyword_frequency

                total_exp_refinements = sum(r.refinement_count for r in cot_results)
                result["total_refinement_iterations"] = (
                    result.get("total_refinement_iterations", 0) + total_exp_refinements
                )
                result["quality_scores"] = {
                    **(result.get("quality_scores") or {}),
                    "experiences": [r.critique.quality_level.value for r in cot_results],
                }
            else:
                result["tailored_experiences"] = experiences_result

        if errors:
            result["errors"] = state.get("errors", []) + errors

        return _end_step(state, step_name, result)

    def assemble_cv(state: CVWarlockState) -> dict:
        """Assemble the final tailored CV and compute total generation time."""
        step_name = "assemble_cv"
        result = _start_step(state, step_name, use_cot, on_step_start)

        if state.get("errors"):
            return _end_step(state, step_name, result)

        try:
            tailored_cv = cv_tailor.assemble_cv(
                state["cv_data"],
                state["tailored_summary"],
                state["tailored_experiences"],
                state["tailored_skills"][0] if state["tailored_skills"] else "",
            )
            result["tailored_cv"] = tailored_cv

            # Compute total generation time
            step_timings = state.get("step_timings", [])
            total_time = sum(t.get("duration_seconds", 0) or 0 for t in step_timings)
            # Add current step time estimate
            if state.get("current_step_start"):
                total_time += time.time() - state["current_step_start"]

            result["total_generation_time"] = total_time

            # Update description to show completion
            result["current_step_description"] = (
                f"CV generation complete! Total time: {total_time:.1f}s"
            )
        except Exception as e:
            result["errors"] = state.get("errors", []) + [f"CV assembly failed: {e!s}"]

        return _end_step(state, step_name, result)

    return {
        "validate_inputs": validate_inputs,
        "extract_cv": extract_cv,
        "extract_job": extract_job,
        "extract_all": extract_all,
        "analyze_match": analyze_match,
        "create_plan": create_plan,
        "tailor_summary": tailor_summary,
        "tailor_experiences": tailor_experiences,
        "tailor_skills": tailor_skills,
        "tailor_skills_and_experiences": tailor_skills_and_experiences,
        "assemble_cv": assemble_cv,
    }
