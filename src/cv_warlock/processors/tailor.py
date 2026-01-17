"""CV tailoring processor."""

from langchain_core.prompts import ChatPromptTemplate

from cv_warlock.llm.base import LLMProvider
from cv_warlock.models.cv import CVData, Experience
from cv_warlock.models.job_spec import JobRequirements
from cv_warlock.models.state import TailoringPlan
from cv_warlock.prompts.generation import (
    SUMMARY_TAILORING_PROMPT,
    EXPERIENCE_TAILORING_PROMPT,
    SKILLS_TAILORING_PROMPT,
    CV_ASSEMBLY_PROMPT,
)


class CVTailor:
    """Tailor CV sections based on tailoring plan."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.summary_prompt = ChatPromptTemplate.from_template(SUMMARY_TAILORING_PROMPT)
        self.experience_prompt = ChatPromptTemplate.from_template(EXPERIENCE_TAILORING_PROMPT)
        self.skills_prompt = ChatPromptTemplate.from_template(SKILLS_TAILORING_PROMPT)
        self.assembly_prompt = ChatPromptTemplate.from_template(CV_ASSEMBLY_PROMPT)

    def tailor_summary(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
    ) -> str:
        """Tailor the professional summary.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.
            tailoring_plan: Tailoring plan.

        Returns:
            str: Tailored summary text.
        """
        model = self.llm_provider.get_chat_model(temperature=0.4)

        chain = self.summary_prompt | model
        result = chain.invoke({
            "original_summary": cv_data.summary or "No summary provided",
            "job_title": job_requirements.job_title,
            "company": job_requirements.company or "the company",
            "key_requirements": ", ".join(job_requirements.required_skills[:5]),
            "relevant_strengths": ", ".join(tailoring_plan["skills_to_highlight"][:5]),
        })

        return result.content

    def tailor_experience(
        self,
        experience: Experience,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
    ) -> str:
        """Tailor a single experience entry.

        Args:
            experience: Experience entry to tailor.
            job_requirements: Structured job requirements.
            tailoring_plan: Tailoring plan.

        Returns:
            str: Tailored experience text.
        """
        model = self.llm_provider.get_chat_model(temperature=0.3)

        chain = self.experience_prompt | model
        result = chain.invoke({
            "title": experience.title,
            "company": experience.company,
            "period": f"{experience.start_date} - {experience.end_date or 'Present'}",
            "description": experience.description,
            "achievements": "\n".join(f"- {a}" for a in experience.achievements),
            "target_requirements": ", ".join(job_requirements.required_skills[:5]),
            "skills_to_emphasize": ", ".join(tailoring_plan["skills_to_highlight"][:5]),
        })

        return result.content

    def tailor_experiences(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
        tailoring_plan: TailoringPlan,
    ) -> list[str]:
        """Tailor all experience entries.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.
            tailoring_plan: Tailoring plan.

        Returns:
            list[str]: List of tailored experience texts.
        """
        tailored = []
        for exp in cv_data.experiences:
            tailored_exp = self.tailor_experience(exp, job_requirements, tailoring_plan)
            tailored.append(tailored_exp)
        return tailored

    def tailor_skills(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
    ) -> str:
        """Tailor the skills section.

        Args:
            cv_data: Structured CV data.
            job_requirements: Structured job requirements.

        Returns:
            str: Tailored skills section text.
        """
        model = self.llm_provider.get_chat_model(temperature=0.2)

        chain = self.skills_prompt | model
        result = chain.invoke({
            "all_skills": ", ".join(cv_data.skills),
            "required_skills": ", ".join(job_requirements.required_skills),
            "preferred_skills": ", ".join(job_requirements.preferred_skills),
        })

        return result.content

    def assemble_cv(
        self,
        cv_data: CVData,
        tailored_summary: str,
        tailored_experiences: list[str],
        tailored_skills: str,
    ) -> str:
        """Assemble the final tailored CV.

        Args:
            cv_data: Original CV data.
            tailored_summary: Tailored summary.
            tailored_experiences: List of tailored experience texts.
            tailored_skills: Tailored skills section.

        Returns:
            str: Complete tailored CV in markdown format.
        """
        model = self.llm_provider.get_chat_model(temperature=0.2)

        # Format contact info
        contact = cv_data.contact
        contact_str = f"**{contact.name}**\n"
        if contact.email:
            contact_str += f"Email: {contact.email}\n"
        if contact.phone:
            contact_str += f"Phone: {contact.phone}\n"
        if contact.location:
            contact_str += f"Location: {contact.location}\n"
        if contact.linkedin:
            contact_str += f"LinkedIn: {contact.linkedin}\n"
        if contact.github:
            contact_str += f"GitHub: {contact.github}\n"

        # Format education
        education_str = ""
        for edu in cv_data.education:
            education_str += f"**{edu.degree}** - {edu.institution} ({edu.graduation_date})\n"
            if edu.gpa:
                education_str += f"GPA: {edu.gpa}\n"

        # Format projects
        projects_str = ""
        for proj in cv_data.projects:
            projects_str += f"**{proj.name}**: {proj.description}\n"
            if proj.technologies:
                projects_str += f"Technologies: {', '.join(proj.technologies)}\n"

        # Format certifications
        certs_str = ""
        for cert in cv_data.certifications:
            certs_str += f"- {cert.name} ({cert.issuer})\n"

        chain = self.assembly_prompt | model
        result = chain.invoke({
            "contact": contact_str,
            "tailored_summary": tailored_summary,
            "tailored_experiences": "\n\n---\n\n".join(tailored_experiences),
            "tailored_skills": tailored_skills,
            "education": education_str or "Not provided",
            "projects": projects_str or "Not provided",
            "certifications": certs_str or "Not provided",
        })

        return result.content
