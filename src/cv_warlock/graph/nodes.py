"""LangGraph node definitions for CV tailoring workflow.

Supports chain-of-thought reasoning mode for higher quality generation.
When CoT is enabled, generation is slower (3-4x more LLM calls) but produces
significantly better tailored CVs.
"""

import copy
import time

from cv_warlock.config import get_settings
from cv_warlock.extractors.cv_extractor import CVExtractor
from cv_warlock.extractors.job_extractor import JobExtractor
from cv_warlock.llm.base import LLMProvider
from cv_warlock.models.reasoning import GenerationContext
from cv_warlock.models.state import CVWarlockState, StepTiming
from cv_warlock.processors.matcher import MatchAnalyzer
from cv_warlock.processors.tailor import CVTailor


# Step descriptions for UI display
# New order: skills → experiences → summary
STEP_DESCRIPTIONS = {
    "validate_inputs": "Validating inputs...",
    "extract_cv": "Extracting CV data...",
    "extract_job": "Analyzing job requirements...",
    "analyze_match": "Analyzing CV-job match...",
    "create_plan": "Creating tailoring strategy...",
    "tailor_skills": "Adding job skills to CV (reasoning → generating)...",
    "tailor_experiences": "Tailoring recent experiences in parallel...",
    "tailor_summary": "Crafting summary from tailored content...",
    "assemble_cv": "Assembling final CV...",
}

STEP_DESCRIPTIONS_FAST = {
    "validate_inputs": "Validating inputs...",
    "extract_cv": "Extracting CV data...",
    "extract_job": "Analyzing job requirements...",
    "analyze_match": "Analyzing CV-job match...",
    "create_plan": "Creating tailoring strategy...",
    "tailor_skills": "Adding job skills to CV...",
    "tailor_experiences": "Tailoring recent work experiences...",
    "tailor_summary": "Crafting professional summary...",
    "assemble_cv": "Assembling final CV...",
}


def _start_step(state: CVWarlockState, step_name: str, use_cot: bool = True) -> dict:
    """Record step start time and description."""
    descriptions = STEP_DESCRIPTIONS if use_cot else STEP_DESCRIPTIONS_FAST
    return {
        "current_step": step_name,
        "current_step_description": descriptions.get(step_name, f"Running {step_name}..."),
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
    updates["current_step_start"] = None

    return updates


def create_nodes(llm_provider: LLMProvider, use_cot: bool = True) -> dict:
    """Create all workflow nodes with the given LLM provider.

    Args:
        llm_provider: The LLM provider to use.
        use_cot: Whether to use chain-of-thought reasoning for generation.
                 Default True for higher quality, False for faster generation.

    Returns:
        dict: Dictionary of node functions.
    """
    cv_extractor = CVExtractor(llm_provider)
    job_extractor = JobExtractor(llm_provider)
    match_analyzer = MatchAnalyzer(llm_provider)
    cv_tailor = CVTailor(llm_provider, use_cot=use_cot)

    def validate_inputs(state: CVWarlockState) -> dict:
        """Validate input documents exist and are non-empty."""
        step_name = "validate_inputs"
        result = _start_step(state, step_name, use_cot)
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
        result = _start_step(state, step_name, use_cot)

        try:
            cv_data = cv_extractor.extract(state["raw_cv"])
            result["cv_data"] = cv_data
        except Exception as e:
            result["errors"] = state.get("errors", []) + [f"CV extraction failed: {e!s}"]

        return _end_step(state, step_name, result)

    def extract_job(state: CVWarlockState) -> dict:
        """Extract structured data from job specification."""
        step_name = "extract_job"
        result = _start_step(state, step_name, use_cot)

        try:
            job_requirements = job_extractor.extract(state["raw_job_spec"])
            result["job_requirements"] = job_requirements
        except Exception as e:
            result["errors"] = state.get("errors", []) + [f"Job extraction failed: {e!s}"]

        return _end_step(state, step_name, result)

    def analyze_match(state: CVWarlockState) -> dict:
        """Analyze match between CV and job requirements.

        If assume_all_tech_skills is True, augments CV skills with all required
        technical skills from the job spec before analysis. The augmented CV
        is persisted in state so downstream nodes use the enhanced skill list.
        """
        step_name = "analyze_match"
        result = _start_step(state, step_name, use_cot)

        if state.get("errors"):
            return _end_step(state, step_name, result)

        try:
            cv_data = state["cv_data"]
            job_requirements = state["job_requirements"]

            # If assume_all_tech_skills is enabled, augment CV skills with job requirements
            if state.get("assume_all_tech_skills", True):
                cv_data = copy.deepcopy(cv_data)
                # Add required and preferred skills to CV skills list
                existing_skills = set(s.lower() for s in cv_data.skills)
                for skill in job_requirements.required_skills + job_requirements.preferred_skills:
                    if skill.lower() not in existing_skills:
                        cv_data.skills.append(skill)
                        existing_skills.add(skill.lower())
                # Persist augmented CV data for downstream nodes
                result["cv_data"] = cv_data

            match_analysis = match_analyzer.analyze_match(
                cv_data,
                job_requirements,
            )
            result["match_analysis"] = match_analysis
        except Exception as e:
            result["errors"] = state.get("errors", []) + [f"Match analysis failed: {e!s}"]

        return _end_step(state, step_name, result)

    def create_plan(state: CVWarlockState) -> dict:
        """Create tailoring plan based on match analysis."""
        step_name = "create_plan"
        result = _start_step(state, step_name, use_cot)

        if state.get("errors"):
            return _end_step(state, step_name, result)

        try:
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
        result = _start_step(state, step_name, use_cot)

        if state.get("errors"):
            return _end_step(state, step_name, result)

        # Summary is LAST: has access to tailored skills and experiences
        tailored_skills = state.get("tailored_skills", [""])[0] if state.get("tailored_skills") else ""

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
        result = _start_step(state, step_name, use_cot)

        if state.get("errors"):
            return _end_step(state, step_name, result)

        # Get lookback_years from state or settings default
        lookback_years = state.get("lookback_years") or get_settings().lookback_years

        try:
            if use_cot:
                # Full CoT with reasoning outputs and lookback filtering
                tailored_texts, exp_results, updated_context = cv_tailor.tailor_experiences_with_cot(
                    state["cv_data"],
                    state["job_requirements"],
                    state["tailoring_plan"],
                    context=state.get("generation_context"),
                    lookback_years=lookback_years,
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
        """Tailor the skills section (FIRST in new pipeline order)."""
        step_name = "tailor_skills"
        result = _start_step(state, step_name, use_cot)

        if state.get("errors"):
            return _end_step(state, step_name, result)

        try:
            if use_cot:
                # Full CoT with reasoning output
                # Skills is FIRST now, so no prior context
                cot_result = cv_tailor.tailor_skills_with_cot(
                    state["cv_data"],
                    state["job_requirements"],
                    context=None,  # Skills is first in the pipeline
                )

                # Initialize context for downstream sections (experiences, summary)
                context = GenerationContext(
                    skills_demonstrated=cot_result.reasoning.required_skills_matched
                    + cot_result.reasoning.preferred_skills_matched,
                    primary_keywords_used=cot_result.reasoning.required_skills_matched[:5],
                )

                result["tailored_skills"] = [cot_result.final_skills]
                result["skills_reasoning_result"] = cot_result
                result["generation_context"] = context  # Pass to downstream nodes
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
                # Initialize basic context for downstream
                result["generation_context"] = GenerationContext()
        except Exception as e:
            result["errors"] = state.get("errors", []) + [f"Skills tailoring failed: {e!s}"]

        return _end_step(state, step_name, result)

    def assemble_cv(state: CVWarlockState) -> dict:
        """Assemble the final tailored CV and compute total generation time."""
        step_name = "assemble_cv"
        result = _start_step(state, step_name, use_cot)

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
            total_time = sum(
                t.get("duration_seconds", 0) or 0
                for t in step_timings
            )
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
        "analyze_match": analyze_match,
        "create_plan": create_plan,
        "tailor_summary": tailor_summary,
        "tailor_experiences": tailor_experiences,
        "tailor_skills": tailor_skills,
        "assemble_cv": assemble_cv,
    }
