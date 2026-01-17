"""CV generation prompt templates."""

SUMMARY_TAILORING_PROMPT = """You are an expert CV writer. Rewrite the professional summary to be highly targeted for this specific role.

ORIGINAL SUMMARY:
{original_summary}

TARGET JOB:
{job_title} at {company}

KEY REQUIREMENTS TO ADDRESS:
{key_requirements}

CANDIDATE'S RELEVANT STRENGTHS:
{relevant_strengths}

Write a compelling 2-4 sentence professional summary that:
1. Opens with a strong positioning statement that aligns with the role
2. Highlights the most relevant experience and skills
3. Naturally incorporates key industry terms and keywords
4. Shows alignment with the role's core responsibilities
5. Quantifies experience where possible (years, scale, impact)

IMPORTANT: Do NOT fabricate experience or qualifications. Only reframe and emphasize existing information from the original CV."""


EXPERIENCE_TAILORING_PROMPT = """You are an expert CV writer. Rewrite this work experience entry to emphasize relevance to the target role.

ORIGINAL EXPERIENCE:
Title: {title}
Company: {company}
Period: {period}
Description: {description}
Achievements: {achievements}

TARGET ROLE REQUIREMENTS:
{target_requirements}

SKILLS TO EMPHASIZE:
{skills_to_emphasize}

Rewrite the experience entry with:
1. 3-5 bullet points using strong action verbs
2. Quantified achievements where data exists (numbers, percentages, scale)
3. Emphasis on aspects most relevant to the target role
4. Natural incorporation of relevant keywords
5. Concise bullets (1-2 lines each)

Format as bullet points starting with action verbs (Led, Developed, Implemented, etc.)

IMPORTANT: Do NOT fabricate achievements or inflate responsibilities. Only reframe existing information to highlight relevance."""


SKILLS_TAILORING_PROMPT = """Organize and prioritize the candidate's skills for maximum ATS compatibility and relevance.

CANDIDATE'S SKILLS:
{all_skills}

JOB REQUIRED SKILLS:
{required_skills}

JOB PREFERRED SKILLS:
{preferred_skills}

Create an optimized skills section that:
1. Lists matching required skills first
2. Follows with matching preferred skills
3. Groups related skills logically (e.g., "Programming Languages:", "Frameworks:", "Tools:")
4. Uses exact terminology from job posting where the candidate has equivalent skills
5. Removes irrelevant skills that don't add value for this specific role

Output format - organize into categories with comma-separated skills:
Programming Languages: Python, JavaScript, TypeScript
Frameworks: React, FastAPI, LangChain
Tools: Git, Docker, AWS

IMPORTANT: Only include skills the candidate actually possesses. Do not add skills not present in the original CV."""


CV_ASSEMBLY_PROMPT = """Assemble a complete, polished CV in Markdown format.

CONTACT INFORMATION:
{contact}

TAILORED SUMMARY:
{tailored_summary}

TAILORED EXPERIENCES:
{tailored_experiences}

TAILORED SKILLS:
{tailored_skills}

EDUCATION:
{education}

PROJECTS (if relevant):
{projects}

CERTIFICATIONS (if any):
{certifications}

Create a clean, professional CV in Markdown format that:
1. Uses clear section headers (##)
2. Presents information in reverse chronological order
3. Is scannable with consistent formatting
4. Optimizes for both ATS parsing and human readability
5. Keeps total length appropriate (1-2 pages worth of content)

Format the CV professionally with proper Markdown syntax."""
