"""LangGraph node definitions for CV tailoring workflow."""

import copy

from cv_warlock.extractors.cv_extractor import CVExtractor
from cv_warlock.extractors.job_extractor import JobExtractor
from cv_warlock.llm.base import LLMProvider
from cv_warlock.models.state import CVWarlockState
from cv_warlock.processors.matcher import MatchAnalyzer
from cv_warlock.processors.tailor import CVTailor


def create_nodes(llm_provider: LLMProvider) -> dict:
    """Create all workflow nodes with the given LLM provider.

    Args:
        llm_provider: The LLM provider to use.

    Returns:
        dict: Dictionary of node functions.
    """
    cv_extractor = CVExtractor(llm_provider)
    job_extractor = JobExtractor(llm_provider)
    match_analyzer = MatchAnalyzer(llm_provider)
    cv_tailor = CVTailor(llm_provider)

    def validate_inputs(state: CVWarlockState) -> dict:
        """Validate input documents exist and are non-empty."""
        errors = []

        if not state.get("raw_cv") or not state["raw_cv"].strip():
            errors.append("CV document is empty or missing")

        if not state.get("raw_job_spec") or not state["raw_job_spec"].strip():
            errors.append("Job specification is empty or missing")

        return {
            "current_step": "validate_inputs",
            "errors": state.get("errors", []) + errors,
        }

    def extract_cv(state: CVWarlockState) -> dict:
        """Extract structured data from CV."""
        try:
            cv_data = cv_extractor.extract(state["raw_cv"])
            return {
                "cv_data": cv_data,
                "current_step": "extract_cv",
            }
        except Exception as e:
            return {
                "current_step": "extract_cv",
                "errors": state.get("errors", []) + [f"CV extraction failed: {e!s}"],
            }

    def extract_job(state: CVWarlockState) -> dict:
        """Extract structured data from job specification."""
        try:
            job_requirements = job_extractor.extract(state["raw_job_spec"])
            return {
                "job_requirements": job_requirements,
                "current_step": "extract_job",
            }
        except Exception as e:
            return {
                "current_step": "extract_job",
                "errors": state.get("errors", []) + [f"Job extraction failed: {e!s}"],
            }

    def analyze_match(state: CVWarlockState) -> dict:
        """Analyze match between CV and job requirements.

        If assume_all_tech_skills is True, augments CV skills with all required
        technical skills from the job spec before analysis. The augmented CV
        is persisted in state so downstream nodes use the enhanced skill list.
        """
        if state.get("errors"):
            return {"current_step": "analyze_match"}

        try:
            cv_data = state["cv_data"]
            job_requirements = state["job_requirements"]
            result = {"current_step": "analyze_match"}

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
            return result
        except Exception as e:
            return {
                "current_step": "analyze_match",
                "errors": state.get("errors", []) + [f"Match analysis failed: {e!s}"],
            }

    def create_plan(state: CVWarlockState) -> dict:
        """Create tailoring plan based on match analysis."""
        if state.get("errors"):
            return {"current_step": "create_plan"}

        try:
            tailoring_plan = match_analyzer.create_tailoring_plan(
                state["cv_data"],
                state["job_requirements"],
                state["match_analysis"],
            )
            return {
                "tailoring_plan": tailoring_plan,
                "current_step": "create_plan",
            }
        except Exception as e:
            return {
                "current_step": "create_plan",
                "errors": state.get("errors", []) + [f"Tailoring plan failed: {e!s}"],
            }

    def tailor_summary(state: CVWarlockState) -> dict:
        """Tailor the professional summary."""
        if state.get("errors"):
            return {"current_step": "tailor_summary"}

        try:
            tailored = cv_tailor.tailor_summary(
                state["cv_data"],
                state["job_requirements"],
                state["tailoring_plan"],
            )
            return {
                "tailored_summary": tailored,
                "current_step": "tailor_summary",
            }
        except Exception as e:
            return {
                "current_step": "tailor_summary",
                "errors": state.get("errors", []) + [f"Summary tailoring failed: {e!s}"],
            }

    def tailor_experiences(state: CVWarlockState) -> dict:
        """Tailor experience entries."""
        if state.get("errors"):
            return {"current_step": "tailor_experiences"}

        try:
            tailored = cv_tailor.tailor_experiences(
                state["cv_data"],
                state["job_requirements"],
                state["tailoring_plan"],
            )
            return {
                "tailored_experiences": tailored,
                "current_step": "tailor_experiences",
            }
        except Exception as e:
            return {
                "current_step": "tailor_experiences",
                "errors": state.get("errors", []) + [f"Experience tailoring failed: {e!s}"],
            }

    def tailor_skills(state: CVWarlockState) -> dict:
        """Tailor the skills section."""
        if state.get("errors"):
            return {"current_step": "tailor_skills"}

        try:
            tailored = cv_tailor.tailor_skills(
                state["cv_data"],
                state["job_requirements"],
            )
            return {
                "tailored_skills": [tailored],
                "current_step": "tailor_skills",
            }
        except Exception as e:
            return {
                "current_step": "tailor_skills",
                "errors": state.get("errors", []) + [f"Skills tailoring failed: {e!s}"],
            }

    def assemble_cv(state: CVWarlockState) -> dict:
        """Assemble the final tailored CV."""
        if state.get("errors"):
            return {"current_step": "assemble_cv"}

        try:
            tailored_cv = cv_tailor.assemble_cv(
                state["cv_data"],
                state["tailored_summary"],
                state["tailored_experiences"],
                state["tailored_skills"][0] if state["tailored_skills"] else "",
            )
            return {
                "tailored_cv": tailored_cv,
                "current_step": "assemble_cv",
            }
        except Exception as e:
            return {
                "current_step": "assemble_cv",
                "errors": state.get("errors", []) + [f"CV assembly failed: {e!s}"],
            }

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
