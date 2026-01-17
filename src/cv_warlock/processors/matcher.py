"""Match analyzer for CV-job fit analysis."""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from cv_warlock.llm.base import LLMProvider
from cv_warlock.models.cv import CVData
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.models.state import MatchAnalysis, TailoringPlan
from cv_warlock.prompts.matching import MATCH_ANALYSIS_PROMPT, TAILORING_PLAN_PROMPT


class MatchAnalysisOutput(BaseModel):
    """Pydantic model for structured match analysis output."""

    strong_matches: list[str] = Field(description="Skills/experience that directly match")
    partial_matches: list[str] = Field(description="Skills that partially match")
    gaps: list[str] = Field(description="Missing requirements")
    transferable_skills: list[str] = Field(description="Transferable skills")
    relevance_score: float = Field(description="Match score from 0 to 1", ge=0, le=1)


class TailoringPlanOutput(BaseModel):
    """Pydantic model for structured tailoring plan output."""

    summary_focus: list[str] = Field(description="Key points for summary")
    experiences_to_emphasize: list[str] = Field(description="Which experiences to highlight")
    skills_to_highlight: list[str] = Field(description="Priority skills")
    achievements_to_feature: list[str] = Field(description="Key achievements")
    keywords_to_incorporate: list[str] = Field(description="ATS keywords")
    sections_to_reorder: list[str] = Field(description="Section ordering")


class MatchAnalyzer:
    """Analyze match between CV and job requirements."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.analysis_prompt = ChatPromptTemplate.from_template(MATCH_ANALYSIS_PROMPT)
        self.plan_prompt = ChatPromptTemplate.from_template(TAILORING_PLAN_PROMPT)

    def analyze_match(self, cv_data: CVData, job_requirements: JobRequirements) -> MatchAnalysis:
        """Analyze how well the CV matches job requirements.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.

        Returns:
            MatchAnalysis: Analysis results.
        """
        model = self.llm_provider.get_extraction_model()
        structured_model = model.with_structured_output(MatchAnalysisOutput, method="function_calling")

        chain = self.analysis_prompt | structured_model
        result = chain.invoke({
            "cv_data": cv_data.model_dump_json(indent=2),
            "job_requirements": job_requirements.model_dump_json(indent=2),
        })

        return MatchAnalysis(
            strong_matches=result.strong_matches,
            partial_matches=result.partial_matches,
            gaps=result.gaps,
            transferable_skills=result.transferable_skills,
            relevance_score=result.relevance_score,
        )

    def create_tailoring_plan(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        match_analysis: MatchAnalysis,
    ) -> TailoringPlan:
        """Create a plan for tailoring the CV.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.
            match_analysis: Match analysis results.

        Returns:
            TailoringPlan: Plan for tailoring.
        """
        model = self.llm_provider.get_extraction_model()
        structured_model = model.with_structured_output(TailoringPlanOutput, method="function_calling")

        chain = self.plan_prompt | structured_model
        result = chain.invoke({
            "match_analysis": str(match_analysis),
            "cv_data": cv_data.model_dump_json(indent=2),
            "job_requirements": job_requirements.model_dump_json(indent=2),
        })

        return TailoringPlan(
            summary_focus=result.summary_focus,
            experiences_to_emphasize=result.experiences_to_emphasize,
            skills_to_highlight=result.skills_to_highlight,
            achievements_to_feature=result.achievements_to_feature,
            keywords_to_incorporate=result.keywords_to_incorporate,
            sections_to_reorder=result.sections_to_reorder,
        )
