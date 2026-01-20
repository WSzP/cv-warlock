"""CV generation prompt templates (optimized for speed and token efficiency).

These prompts implement a section-by-section tailoring strategy:
1. Skills first - Add job requirements to existing skills
2. Experiences second - Rewrite bullets to emphasize requested skills (preserve headers)
3. Summary last - Reference tailored content from skills and experiences
"""

SKILLS_TAILORING_PROMPT = """Create an ATS-optimized skills section for this CV.

CANDIDATE'S EXISTING SKILLS: {all_skills}

JOB REQUIREMENTS:
Required: {required_skills}
Preferred: {preferred_skills}

STRATEGY: Add relevant skills from job requirements to the candidate's existing skills.

RULES:
1. Start with all required skills from job posting (candidate claims these)
2. Add preferred skills that match candidate's background
3. Include candidate's existing skills that are relevant to the role
4. Use EXACT job posting terminology (React.js not React, CI/CD not continuous integration)
5. Include dual formats for acronyms: "Amazon Web Services (AWS)"
6. Group logically: Languages, Frameworks, Cloud/DevOps, Databases, Tools
7. Order: Required matches > Preferred matches > Relevant existing
8. Omit irrelevant skills that dilute ATS score
9. TENSORFLOW RULE: Only include TensorFlow if EXPLICITLY mentioned in job requirements.
   If TensorFlow is not in the job spec but a deep learning framework seems needed, use PyTorch instead.

FORMAT (CRITICAL RULES - MUST follow EXACTLY):
1. EVERY category name MUST be wrapped in ** for bold: **Category:**
2. Category and skills MUST be on the SAME LINE
3. Use colon INSIDE the bold: **Category:** not **Category**:

## Technical Skills

**Languages:** Python, TypeScript, SQL
**Frameworks:** React.js, FastAPI, Node.js
**Cloud & DevOps:** AWS (EC2, S3, Lambda), Docker, Kubernetes
**Databases:** PostgreSQL, MongoDB, Redis

WRONG - missing bold on category:
Languages: Python, TypeScript, SQL

WRONG - colon outside bold:
**Languages**: Python, TypeScript, SQL

WRONG - skills on separate line:
**Languages**
Python, TypeScript, SQL

CORRECT - bold category with colon inside, skills on same line:
**Languages:** Python, TypeScript, SQL

Output ONLY the skills section."""


EXPERIENCE_TAILORING_PROMPT = """Rewrite the job description bullets to emphasize relevant skills.

EXPERIENCE (IMMUTABLE - DO NOT CHANGE):
Title: {title}
Company: {company}
Period: {period}

ORIGINAL DESCRIPTION/BULLETS:
{description}
{achievements}

TARGET SKILLS TO EMPHASIZE: {skills_to_emphasize}
KEYWORDS TO INCORPORATE: {target_requirements}

CRITICAL RULES:
1. DO NOT change title, company, or dates - they are IMMUTABLE
2. ONLY rewrite the bullet points/description text
3. Emphasize skills from the target list naturally
4. Use power verbs: Led, Built, Reduced, Implemented, Designed, Optimized
5. Keep metrics if they exist - NEVER fabricate numbers
6. 3-5 bullets, most impactful first
7. Under 20 words per bullet
8. Only reframe existing facts, never fabricate experience

BULLET FORMULA: [Power Verb] + [Action] + [Result with metric] + [Scale/Context]

GOOD: "Reduced API latency 40% via Redis caching, improving UX for 2M+ daily users"
BAD: "Responsible for API performance" (no metric, passive)

Output ONLY bullet points:
- [Bullet 1]
- [Bullet 2]
..."""


SUMMARY_TAILORING_PROMPT = """Create a professional summary that introduces the tailored CV.

ORIGINAL SUMMARY: {original_summary}

TARGET: {job_title} at {company}

TAILORED SKILLS AVAILABLE:
{tailored_skills_preview}

KEY REQUIREMENTS: {key_requirements}
RELEVANT STRENGTHS: {relevant_strengths}

STRATEGY: The summary should preview the value demonstrated in the CV sections below it.

FORMAT (2-4 sentences):
1. Hook: [Title] + [years] + [domain expertise]
2. Value: Key achievement with metric from experiences
3. Fit: Connect skills to THIS role's needs

RULES:
- Reference skills that appear in the tailored skills section
- Include 1+ hard metric from their background
- Mirror 2-3 exact terms from job posting
- No fluff words (passionate, driven, motivated, dedicated)
- No "I am/have" openings
- Only reframe existing facts, never fabricate

Example: "Senior Backend Engineer with 7+ years building distributed systems, including payment infrastructure processing $2B annually. Specialized in Python/Go microservices and event-driven architectures."

Output ONLY the summary."""


CV_ASSEMBLY_PROMPT = """Assemble these sections into a clean, ATS-compatible CV in Markdown.

CONTACT (IMMUTABLE - COPY EXACTLY): {contact}
SUMMARY: {tailored_summary}
SKILLS: {tailored_skills}
EXPERIENCE: {tailored_experiences}
EDUCATION (IMMUTABLE - COPY EXACTLY): {education}
PROJECTS: {projects}
CERTIFICATIONS: {certifications}

SECTION ORDER: Contact > Summary > Skills > Experience > Education > Projects > Certifications
(Omit Projects/Certifications if "Not provided")

=== IMMUTABLE SECTIONS - COPY EXACTLY AS PROVIDED ===
1. CONTACT: Copy EXACTLY as provided, including markdown links. Do NOT add/remove/modify any fields.
2. EDUCATION: This section is IMMUTABLE and MUST NOT be modified in ANY way.
   - Copy the ENTIRE education section EXACTLY as provided
   - Do NOT reformat, reorganize, or "improve" it
   - Do NOT remove ANY details (including "All But Dissertation", partial completion, honors, thesis titles, GPA, etc.)
   - Do NOT change degree names, institution names, or dates
   - Education is SET IN STONE - not part of tailoring

OUTPUT FORMAT:

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
[COPY EDUCATION SECTION EXACTLY AS PROVIDED - NO MODIFICATIONS]

## Projects
### [Project Name] | [Role]
[Description of project and achievements]

## Certifications
- [Certification Name] ([Year])

Output the complete CV in Markdown."""
