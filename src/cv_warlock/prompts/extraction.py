"""Extraction prompt templates.

These prompts extract structured data that feeds into the tailoring pipeline.
Optimized to capture everything needed for high-impact CV rewriting.
"""

CV_EXTRACTION_PROMPT = """You are an expert CV analyst. Extract ALL structured information from this CV with precision.

Your extraction quality directly determines the quality of the tailored CV. Miss nothing.

=== EXTRACTION PRIORITIES ===

**1. METRICS & NUMBERS (Critical for impact bullets)**
Extract EVERY quantifiable item:
- Revenue/cost figures ($, €, etc.)
- Percentages (growth, improvement, reduction)
- Scale numbers (users, transactions, team size, data volume)
- Time metrics (years of experience, project duration, time saved)
- Counts (projects delivered, features shipped, clients served)

**2. ACHIEVEMENTS vs RESPONSIBILITIES**
Distinguish between:
- ACHIEVEMENTS: Results, outcomes, impact (most valuable)
- RESPONSIBILITIES: What they were supposed to do (less valuable)
Flag achievements explicitly - these become the best bullets.

**3. TECHNICAL DEPTH**
For each technology/skill mentioned:
- Context of use (built, maintained, optimized, led migration)
- Scale of use (production, enterprise, startup)
- Recency (current role vs. old role)

**4. CAREER PROGRESSION**
Note:
- Promotions within companies
- Increasing scope/responsibility
- Leadership evolution (IC → Lead → Manager)

=== CV TEXT ===

{cv_text}

=== EXTRACTION REQUIREMENTS ===

For WORK EXPERIENCE, extract:
- Job title (exact)
- Company name
- Location (if present)
- Start date and end date (or "Present")
- Full description text
- List of specific achievements WITH metrics
- Technologies/tools used in this role specifically
- Team size managed (if applicable)
- Scope indicators (budget, users, revenue responsibility)

For EDUCATION, extract:
- Degree type and field
- Institution name
- Graduation year
- GPA (if impressive, i.e., 3.5+/4.0 or equivalent)
- Honors, awards, relevant coursework

For SKILLS, extract:
- Programming languages (with proficiency if indicated)
- Frameworks and libraries
- Tools and platforms
- Cloud services and infrastructure
- Databases
- Methodologies (Agile, Scrum, etc.)
- Soft skills only if explicitly listed

For PROJECTS (if present):
- Project name
- Description
- Technologies used
- Your specific role/contribution
- Impact or results

For CERTIFICATIONS:
- Certification name (exact)
- Issuing organization
- Date obtained (if present)
- Expiration (if applicable)

For CONTACT INFO:
- Full name
- Email
- Phone
- LinkedIn URL
- GitHub/Portfolio (if present)
- Location (city, country)

=== OUTPUT QUALITY ===

- Extract information EXACTLY as written, preserving original phrasing for achievements
- Never fabricate or infer information not present
- Flag ambiguous items for clarification
- Preserve ALL numbers and metrics exactly
- Note the overall years of experience
- Identify the candidate's apparent seniority level"""


JOB_EXTRACTION_PROMPT = """You are an expert job posting analyst specializing in ATS optimization and hiring patterns.

Extract and categorize requirements for strategic CV tailoring. Your analysis determines which parts of the candidate's background to emphasize.

=== JOB SPECIFICATION ===

{job_spec_text}

=== CRITICAL EXTRACTIONS ===

**1. REQUIRED vs PREFERRED - Be Precise**
This distinction is crucial for prioritization:
- REQUIRED: "Must have", "Required", "X years of experience in", "You have"
- PREFERRED: "Nice to have", "Bonus", "Preferred", "Ideally", "Experience with X is a plus"
- INFERRED REQUIRED: If listed first or emphasized heavily without qualifier

**2. TECHNICAL SKILLS (exact terminology)**
Extract the EXACT phrasing used:
- Programming languages (note versions if specified)
- Frameworks and libraries (exact names: "React.js" vs "React")
- Cloud platforms (AWS, GCP, Azure - specific services if mentioned)
- Databases and data tools
- DevOps and infrastructure tools
- Methodologies and practices

**3. EXPERIENCE REQUIREMENTS**
- Years of experience (minimum and preferred)
- Seniority level (Junior, Mid, Senior, Staff, Principal, Lead)
- Industry experience (fintech, healthcare, SaaS, etc.)
- Company size experience (startup, scale-up, enterprise)
- Specific domain knowledge

**4. ATS KEYWORDS (Critical)**
Identify terms that will be used for automated filtering:
- Technical terms and acronyms
- Industry-specific jargon
- Certifications mentioned
- Tool names and versions
- Methodology names

**5. SOFT SKILLS & CULTURE FIT**
Extract expectations around:
- Communication requirements
- Collaboration style
- Leadership expectations
- Work style (remote, agile, fast-paced)
- Values alignment indicators

**6. RESPONSIBILITIES (for relevance matching)**
What will this person DO? Extract:
- Primary functions
- Team interactions
- Stakeholder management
- Scope of impact

**7. COMPANY CONTEXT**
- Company name
- Industry/sector
- Company stage/size (if mentioned)
- Team they'll join
- Products/services they'll work on

=== OUTPUT STRUCTURE ===

Organize into:
1. Job Title (exact)
2. Company Name
3. Required Skills (with source phrase)
4. Preferred Skills (with source phrase)
5. Years of Experience Required
6. Seniority Level
7. Key Responsibilities
8. ATS Keywords (comprehensive list)
9. Culture/Soft Skill Requirements
10. Notable Company Context

=== ACCURACY NOTES ===

- Distinguish between STATED requirements and INFERRED requirements
- When in doubt, categorize as preferred not required
- Extract terminology EXACTLY as written (ATS matching is literal)
- Note any unusual requirements that might be dealbreakers
- Flag if the posting seems to have inflated requirements (common in job postings)"""
