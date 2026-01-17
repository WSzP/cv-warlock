"""CV extraction using LLM with structured output."""

from langchain_core.prompts import ChatPromptTemplate

from cv_warlock.llm.base import LLMProvider
from cv_warlock.models.cv import CVData
from cv_warlock.prompts.extraction import CV_EXTRACTION_PROMPT


class CVExtractor:
    """Extract structured CV data from raw text."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.prompt = ChatPromptTemplate.from_template(CV_EXTRACTION_PROMPT)

    def extract(self, raw_cv: str) -> CVData:
        """Extract structured CV data from raw text.

        Args:
            raw_cv: Raw CV text content.

        Returns:
            CVData: Structured CV data.
        """
        model = self.llm_provider.get_extraction_model()
        structured_model = model.with_structured_output(CVData)

        chain = self.prompt | structured_model
        result = chain.invoke({"cv_text": raw_cv})

        return result
