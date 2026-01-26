"""Extraction prompt templates.

OPTIMIZED: Reduced token usage by ~40% while maintaining extraction quality.
"""

CV_EXTRACTION_PROMPT = """Extract ALL structured data from this CV. Preserve exact phrasing, numbers, and metrics.

=== CV TEXT ===
{cv_text}

=== EXTRACT ===

**EXPERIENCE** (for each role):
- Look strictly for headers like "### Role at Company" or "### Role | Company"
- Split "Role at Company" into Title="Role" and Company="Company"
- Extract dates from the line immediately following the header
- Title, company, location, dates (start-end or "Present")
- Description, achievements WITH metrics (revenue, %, users, team size, time saved)
- Technologies used, team size managed, scope (budget, impact)

**EDUCATION**: Degree, field, institution, year, GPA (if 3.5+), honors
- **raw_education_text**: Copy the ENTIRE education section EXACTLY as it appears in the CV, preserving ALL formatting, line breaks, and details (including "All But Dissertation", honors, thesis titles, etc.). This field is IMMUTABLE during tailoring.

**SKILLS**: Languages, frameworks, tools, cloud, databases, methodologies

**PROJECTS**: Name, description, tech, your role, results

**CERTIFICATIONS** (professional credentials ONLY):
- Examples: AWS Solutions Architect, PMP, CISSP, Google Cloud Professional, Scrum Master
- Name, issuer, date, expiration
- NOT books, papers, or publications - those go in PUBLICATIONS

**PUBLICATIONS** (books, papers, articles authored by candidate):
- Examples: Books, research papers, journal articles, technical blog posts
- Title, publisher/journal, year, URL/DOI
- NOT certifications or credentials - those go in CERTIFICATIONS

**CONTACT**: Name, email, phone, LinkedIn, GitHub, location
- **raw_contact_line**: Copy the EXACT contact line from the CV header (preserve markdown links like `[email](mailto:email)` and `[phone](tel:phone)` EXACTLY as written)

=== PRIORITIES ===
1. METRICS: Extract ALL numbers (revenue, %, scale, time, counts)
2. ACHIEVEMENTS vs RESPONSIBILITIES: Flag results/outcomes separately
3. TECHNICAL DEPTH: Context (built/maintained), scale (prod/enterprise), recency
4. PROGRESSION: Promotions, scope increases, leadership evolution

Never fabricate. Preserve original phrasing. Note years of experience and seniority level."""


JOB_EXTRACTION_PROMPT = """Extract and categorize job requirements for CV tailoring. Use EXACT terminology from posting.

=== JOB SPECIFICATION ===
{job_spec_text}

=== EXTRACT ===

**REQUIRED vs PREFERRED** (critical distinction):
- REQUIRED: "Must have", "Required", "X years experience", explicit requirements
- PREFERRED: "Nice to have", "Bonus", "Ideally", "Plus"
- When unclear, categorize as preferred

**SKILLS** (exact phrasing for ATS):
- Languages (with versions), frameworks, cloud services, databases, DevOps tools, methodologies

**EXPERIENCE**:
- Years required/preferred, seniority level, industry experience, company size preference

**ATS KEYWORDS**: Technical terms, acronyms, certifications, tool names, methodology names

**RESPONSIBILITIES**: Primary functions, team interactions, scope of impact

**SOFT SKILLS**: Communication, collaboration, leadership, work style expectations

**CONTEXT**: Job title, company, industry, team, products

=== OUTPUT ===
1. Job Title, Company
2. Required Skills (with source)
3. Preferred Skills (with source)
4. Years Experience, Seniority Level
5. Key Responsibilities
6. ATS Keywords (comprehensive)
7. Soft Skills/Culture Fit
8. Company Context

Extract EXACTLY as written. Note dealbreakers and inflated requirements."""
