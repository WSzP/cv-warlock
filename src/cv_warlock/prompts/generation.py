"""CV generation prompt templates (optimized for speed and token efficiency).

These prompts use minimal tokens while maintaining quality output.
For detailed reasoning-based generation, see reasoning.py (CoT mode).
"""

SUMMARY_TAILORING_PROMPT = """Tailor this professional summary for the target role.

ORIGINAL: {original_summary}

TARGET: {job_title} at {company}
REQUIREMENTS: {key_requirements}
STRENGTHS: {relevant_strengths}

FORMAT (2-4 sentences):
1. Hook: [Title] + [years] + [domain] + [differentiator]
2. Value: One quantified achievement (%, $, scale)
3. Fit: Connect to THIS role's needs

RULES:
- Include 1+ hard metric from their background
- Mirror 2-3 exact terms from job posting
- No fluff words (passionate, driven, motivated, dedicated)
- No "I am/have" openings
- Only reframe existing facts, never fabricate

Example: "Senior Backend Engineer with 7+ years building distributed systems, including payment infrastructure processing $2B annually. Specialized in Python/Go microservices and event-driven architectures."

Output ONLY the summary."""


EXPERIENCE_TAILORING_PROMPT = """Tailor this experience for the target role.

EXPERIENCE:
Title: {title}
Company: {company}
Period: {period}
Description: {description}
Achievements: {achievements}

TARGET: {target_requirements}
EMPHASIZE: {skills_to_emphasize}

BULLET FORMULA: [Power Verb] + [Action] + [Result with metric] + [Scale/Context]

GOOD: "Reduced API latency 40% via Redis caching, improving UX for 2M+ daily users"
BAD: "Responsible for API performance" (no metric, passive)

RULES:
- 3-5 bullets, most relevant first
- Start each with strong past-tense verb
- Include metrics where data exists (never fabricate)
- Weave in 2-3 job posting keywords naturally
- Under 20 words per bullet

Output ONLY bullet points:
- [Bullet 1]
- [Bullet 2]
..."""


SKILLS_TAILORING_PROMPT = """Create an ATS-optimized skills section.

CANDIDATE SKILLS: {all_skills}
REQUIRED: {required_skills}
PREFERRED: {preferred_skills}

RULES:
- Use EXACT job posting terminology (React.js not React, CI/CD not continuous integration)
- Include dual formats for acronyms: "Amazon Web Services (AWS)"
- Order: Required matches > Preferred matches > Transferable
- Group logically (Languages, Frameworks, Cloud/DevOps, Databases, Tools)
- Omit irrelevant skills (they dilute ATS score)
- Never add skills candidate doesn't have

FORMAT:
**Technical Skills**
Languages: Python, TypeScript, SQL
Frameworks: React.js, FastAPI, Node.js
Cloud & DevOps: AWS (EC2, S3, Lambda), Docker, Kubernetes
Databases: PostgreSQL, MongoDB, Redis

Output ONLY the skills section."""


CV_ASSEMBLY_PROMPT = """Assemble these sections into a clean, ATS-compatible CV in Markdown.

CONTACT: {contact}
SUMMARY: {tailored_summary}
EXPERIENCE: {tailored_experiences}
SKILLS: {tailored_skills}
EDUCATION: {education}
PROJECTS: {projects}
CERTIFICATIONS: {certifications}

SECTION ORDER: Contact > Summary > Skills > Experience > Education > Projects > Certifications
(Omit Projects/Certifications if "Not provided")

FORMAT:
# [Name]
[Email] | [Phone] | [Location] | [LinkedIn]

## Professional Summary
[Summary paragraph]

## Technical Skills
[Skills section]

## Professional Experience

### [Title] | [Company]
*[Start] - [End]*
- [Bullet]
- [Bullet]

## Education
### [Degree] in [Field]
**[University]** | [Year]

## Projects
[If relevant]

## Certifications
[If relevant]

Output the complete CV in Markdown."""
