"""RLM-enhanced workflow nodes.

These nodes use RLM orchestration for handling large documents
with interpretable reasoning and chunk-by-chunk analysis.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cv_warlock.models.cv import CVData

# Import step descriptions from nodes module
from cv_warlock.graph.nodes import STEP_DESCRIPTIONS
from cv_warlock.llm.base import LLMProvider
from cv_warlock.models.state import CVWarlockState, RLMMetadata, RLMTrajectoryStep
from cv_warlock.rlm import (
    RLMConfig,
    RLMOrchestrator,
    RLMResult,
)
from cv_warlock.rlm.prompts import (
    RLM_CV_EXTRACTION_TASK,
    RLM_JOB_EXTRACTION_TASK,
    RLM_MATCH_ANALYSIS_TASK,
)

logger = logging.getLogger(__name__)


def _is_valid_cv_data(cv_data: CVData) -> bool:
    """Check if CVData has meaningful content (not empty/placeholder values).

    Returns:
        True if the CVData has at least a name and some experiences/skills.
    """
    from cv_warlock.models.cv import CVData

    if not isinstance(cv_data, CVData):
        return False

    # Must have a real name (not placeholder)
    name = cv_data.contact.name if cv_data.contact else ""
    if not name or name.upper() in ["<UNKNOWN>", "UNKNOWN", "NAME NOT PROVIDED", "N/A", ""]:
        return False

    # Must have at least some content - either experiences or skills
    has_experiences = len(cv_data.experiences) > 0
    has_skills = len(cv_data.skills) > 0

    return has_experiences or has_skills


def _convert_trajectory(rlm_result: RLMResult) -> list[RLMTrajectoryStep]:
    """Convert RLM trajectory to state format."""
    steps: list[RLMTrajectoryStep] = []
    for step in rlm_result.trajectory:
        steps.append(
            RLMTrajectoryStep(
                step_number=step.step_number,
                action_type=step.action_type.value,
                execution_result=step.execution_result,
                sub_call_made=step.sub_call_made,
                duration_ms=step.duration_ms,
            )
        )
    return steps


def _check_rlm_timeout(rlm_result: RLMResult) -> bool:
    """Check if RLM result indicates a timeout occurred."""
    if not rlm_result.trajectory:
        return False
    last_step = rlm_result.trajectory[-1]
    return last_step.model_output == "TIMEOUT"


def _get_timeout_message(rlm_result: RLMResult) -> str | None:
    """Get timeout message from RLM result if timeout occurred."""
    if not rlm_result.trajectory:
        return None
    last_step = rlm_result.trajectory[-1]
    if last_step.model_output == "TIMEOUT":
        return last_step.execution_result
    return None


def _create_rlm_metadata(
    enabled: bool,
    used: bool,
    rlm_result: RLMResult | None = None,
) -> RLMMetadata:
    """Create RLM metadata from result."""
    if rlm_result is None:
        return RLMMetadata(
            enabled=enabled,
            used=used,
            total_iterations=0,
            sub_call_count=0,
            execution_time_seconds=0.0,
            trajectory=[],
            intermediate_findings={},
        )

    return RLMMetadata(
        enabled=enabled,
        used=used,
        total_iterations=rlm_result.total_iterations,
        sub_call_count=rlm_result.sub_call_count,
        execution_time_seconds=rlm_result.execution_time_seconds,
        trajectory=_convert_trajectory(rlm_result),
        intermediate_findings={
            k: str(v)[:500] for k, v in rlm_result.intermediate_findings.items()
        },
    )


def create_rlm_nodes(
    root_provider: LLMProvider,
    sub_provider: LLMProvider | None = None,
    config: RLMConfig | None = None,
    use_cot: bool = True,
    on_step_start: Callable[[str, str], None] | None = None,
) -> dict[str, Callable[[CVWarlockState], dict]]:
    """Create RLM-enhanced workflow nodes.

    These nodes use RLM orchestration for:
    - Large document handling
    - Interpretable reasoning
    - Chunk-by-chunk analysis

    Args:
        root_provider: LLM provider for root model.
        sub_provider: LLM provider for sub-calls (optional).
        config: RLM configuration.
        use_cot: Whether to use CoT for tailoring steps.

    Returns:
        Dict of node name to node function.
    """
    # Import standard nodes for non-RLM steps
    from cv_warlock.graph.nodes import create_nodes

    # Create standard nodes as fallback
    # Use sub_provider (fast model) for tailoring steps for better performance
    standard_nodes = create_nodes(
        root_provider,
        use_cot=use_cot,
        on_step_start=on_step_start,
        tailor_provider=sub_provider,
    )

    # Create RLM orchestrator
    orchestrator = RLMOrchestrator(
        root_provider=root_provider,
        sub_provider=sub_provider,
        config=config,
    )

    def should_use_rlm(state: CVWarlockState) -> bool:
        """Determine if RLM should be used based on state."""
        if not state.get("use_rlm", False):
            return False

        # Check size threshold
        threshold = config.size_threshold if config else 8000
        total_size = len(state["raw_cv"]) + len(state["raw_job_spec"])
        return total_size > threshold

    def extract_cv_rlm(state: CVWarlockState) -> dict:
        """Extract CV data using RLM for long documents."""
        if not should_use_rlm(state):
            # Use standard extraction
            result = standard_nodes["extract_cv"](state)
            result["rlm_metadata"] = _create_rlm_metadata(
                enabled=state.get("use_rlm", False),
                used=False,
            )
            return result

        if on_step_start:
            description = "Extracting CV structure via RLM..."
            # Try to use standard description + RLM annotation if available
            base_desc = STEP_DESCRIPTIONS.get("extract_cv", "Extracting CV data...")
            if base_desc:
                description = f"{base_desc.rstrip('...')} (RLM)..."

            try:
                on_step_start("extract_cv", description)
            except Exception:
                pass

        logger.info("Using RLM for CV extraction")

        try:
            from cv_warlock.models.cv import CVData

            rlm_result = orchestrator.complete(
                task=RLM_CV_EXTRACTION_TASK,
                cv_text=state["raw_cv"],
                job_text=state["raw_job_spec"],
                output_schema=CVData,
            )

            # Check for timeout first
            if _check_rlm_timeout(rlm_result):
                timeout_msg = _get_timeout_message(rlm_result)
                logger.warning(f"RLM CV extraction timed out: {timeout_msg}")
                # Try fallback - if successful, log warning but don't add to errors
                # (errors would abort the workflow)
                result = standard_nodes["extract_cv"](state)
                result["rlm_metadata"] = _create_rlm_metadata(True, False, rlm_result)
                # Only add error if fallback also failed
                if result.get("cv_data") is None:
                    result["errors"] = state.get("errors", []) + [
                        f"RLM timeout during CV extraction: {timeout_msg}. "
                        "Fallback extraction also failed."
                    ]
                else:
                    # Fallback succeeded - add info to metadata but not errors
                    logger.info(f"RLM timeout, but standard extraction succeeded: {timeout_msg}")
                return result

            if rlm_result.success and rlm_result.answer:
                cv_data = rlm_result.answer if isinstance(rlm_result.answer, CVData) else None

                # Validate that CVData has meaningful content (not empty/placeholder)
                if cv_data is not None and _is_valid_cv_data(cv_data):
                    return {
                        "cv_data": cv_data,
                        "current_step": "extract_cv",
                        "current_step_description": "Extracted CV structure via RLM",
                        "rlm_metadata": _create_rlm_metadata(True, True, rlm_result),
                    }
                else:
                    # RLM returned empty or invalid CVData - fallback
                    logger.warning("RLM returned empty/invalid CVData, falling back to standard")
                    result = standard_nodes["extract_cv"](state)
                    result["rlm_metadata"] = _create_rlm_metadata(True, False)
                    return result
            else:
                # Fallback to standard
                logger.warning(f"RLM extraction failed: {rlm_result.error}")
                result = standard_nodes["extract_cv"](state)
                result["rlm_metadata"] = _create_rlm_metadata(True, False)
                return result

        except Exception as e:
            logger.exception("RLM CV extraction error, falling back")
            result = standard_nodes["extract_cv"](state)
            result["rlm_metadata"] = _create_rlm_metadata(True, False)
            result["errors"] = state.get("errors", []) + [f"RLM extraction fallback: {e}"]
            return result

    def extract_job_rlm(state: CVWarlockState) -> dict:
        """Extract job requirements using RLM for long documents."""
        if not should_use_rlm(state):
            return standard_nodes["extract_job"](state)

        if on_step_start:
            description = "Analyzing job requirements via RLM..."
            base_desc = STEP_DESCRIPTIONS.get("extract_job", "Analyzing job requirements...")
            if base_desc:
                description = f"{base_desc.rstrip('...')} (RLM)..."

            try:
                on_step_start("extract_job", description)
            except Exception:
                pass

        logger.info("Using RLM for job extraction")

        try:
            from cv_warlock.models.job_spec import JobRequirements

            rlm_result = orchestrator.complete(
                task=RLM_JOB_EXTRACTION_TASK,
                cv_text=state["raw_cv"],
                job_text=state["raw_job_spec"],
                output_schema=JobRequirements,
            )

            # Check for timeout first
            if _check_rlm_timeout(rlm_result):
                timeout_msg = _get_timeout_message(rlm_result)
                logger.warning(f"RLM job extraction timed out: {timeout_msg}")
                result = standard_nodes["extract_job"](state)
                # Only add error if fallback also failed
                if result.get("job_requirements") is None:
                    result["errors"] = state.get("errors", []) + [
                        f"RLM timeout during job extraction: {timeout_msg}. "
                        "Fallback extraction also failed."
                    ]
                else:
                    logger.info(
                        f"RLM timeout, but standard job extraction succeeded: {timeout_msg}"
                    )
                return result

            if rlm_result.success and rlm_result.answer:
                job_requirements = (
                    rlm_result.answer if isinstance(rlm_result.answer, JobRequirements) else None
                )

                if job_requirements is not None:
                    # Update RLM metadata with combined stats
                    existing_metadata = state.get("rlm_metadata")
                    if existing_metadata:
                        new_metadata = _create_rlm_metadata(True, True, rlm_result)
                        new_metadata["total_iterations"] += existing_metadata.get(
                            "total_iterations", 0
                        )
                        new_metadata["sub_call_count"] += existing_metadata.get("sub_call_count", 0)
                    else:
                        new_metadata = _create_rlm_metadata(True, True, rlm_result)

                    return {
                        "job_requirements": job_requirements,
                        "current_step": "extract_job",
                        "current_step_description": "Extracted job requirements via RLM",
                        "rlm_metadata": new_metadata,
                    }
                else:
                    # RLM returned answer but not in correct format - fallback
                    logger.warning(
                        "RLM returned non-JobRequirements answer, falling back to standard"
                    )
                    return standard_nodes["extract_job"](state)
            else:
                logger.warning(f"RLM job extraction failed: {rlm_result.error}")
                return standard_nodes["extract_job"](state)

        except Exception as e:
            logger.exception("RLM job extraction error, falling back")
            result = standard_nodes["extract_job"](state)
            result["errors"] = state.get("errors", []) + [f"RLM job extraction fallback: {e}"]
            return result

    def analyze_match_rlm(state: CVWarlockState) -> dict:
        """Analyze match using RLM for comprehensive analysis."""
        if not should_use_rlm(state):
            return standard_nodes["analyze_match"](state)

        if on_step_start:
            description = "Analyzing CV-job match via RLM..."
            base_desc = STEP_DESCRIPTIONS.get("analyze_match", "Analyzing CV-job match...")
            if base_desc:
                description = f"{base_desc.rstrip('...')} (RLM)..."

            try:
                on_step_start("analyze_match", description)
            except Exception:
                pass

        logger.info("Using RLM for match analysis")

        try:
            from cv_warlock.models.state import MatchAnalysis

            rlm_result = orchestrator.complete(
                task=RLM_MATCH_ANALYSIS_TASK,
                cv_text=state["raw_cv"],
                job_text=state["raw_job_spec"],
            )

            if rlm_result.success and rlm_result.answer:
                # Try to parse as MatchAnalysis
                answer = rlm_result.answer
                if isinstance(answer, dict):
                    match_analysis = MatchAnalysis(
                        strong_matches=answer.get("strong_matches", []),
                        partial_matches=answer.get("partial_matches", []),
                        gaps=answer.get("gaps", []),
                        transferable_skills=answer.get("transferable_skills", []),
                        relevance_score=float(answer.get("relevance_score", 0.5)),
                    )
                else:
                    # Fallback to standard if parsing fails
                    return standard_nodes["analyze_match"](state)

                return {
                    "match_analysis": match_analysis,
                    "current_step": "analyze_match",
                    "current_step_description": "Analyzed match via RLM",
                }
            else:
                logger.warning(f"RLM match analysis failed: {rlm_result.error}")
                return standard_nodes["analyze_match"](state)

        except Exception as e:
            logger.exception("RLM match analysis error, falling back")
            result = standard_nodes["analyze_match"](state)
            result["errors"] = state.get("errors", []) + [f"RLM match analysis fallback: {e}"]
            return result

    # Return nodes dict - RLM for extraction/analysis, standard for tailoring
    return {
        "validate_inputs": standard_nodes["validate_inputs"],
        "extract_cv": extract_cv_rlm,
        "extract_job": extract_job_rlm,
        "analyze_match": analyze_match_rlm,
        "create_plan": standard_nodes["create_plan"],
        "tailor_skills": standard_nodes["tailor_skills"],
        "tailor_experiences": standard_nodes["tailor_experiences"],
        "tailor_summary": standard_nodes["tailor_summary"],
        "assemble_cv": standard_nodes["assemble_cv"],
    }
