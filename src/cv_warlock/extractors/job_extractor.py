"""Job specification extraction using LLM with structured output."""

from langchain_core.prompts import ChatPromptTemplate

from cv_warlock.llm.base import LLMProvider
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.prompts.extraction import JOB_EXTRACTION_PROMPT


class JobExtractor:
    """Extract structured job requirements from raw job posting text."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.prompt = ChatPromptTemplate.from_template(JOB_EXTRACTION_PROMPT)

    def extract(self, raw_job_spec: str) -> JobRequirements:
        """Extract structured job requirements from raw text.

        Args:
            raw_job_spec: Raw job posting text content.

        Returns:
            JobRequirements: Structured job requirements.
        """
        model = self.llm_provider.get_extraction_model()
        structured_model = model.with_structured_output(JobRequirements, method="function_calling")

        chain = self.prompt | structured_model
        result = chain.invoke({"job_spec_text": raw_job_spec})

        return result
