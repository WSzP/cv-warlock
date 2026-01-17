"""Extraction prompt templates."""

CV_EXTRACTION_PROMPT = """You are an expert CV parser. Extract structured information from the following CV text.

Be thorough and extract ALL information present. If something is not explicitly stated, leave it as null or empty.

For each work experience:
- Extract the job title, company name, and dates
- Extract specific achievements with metrics when available
- Identify technologies and skills used in that role
- Note the description of responsibilities

For education:
- Extract degree, institution, and graduation date
- Note GPA if mentioned
- List relevant coursework if specified

For skills:
- Extract all technical and soft skills mentioned
- Include tools, frameworks, languages, and methodologies

CV TEXT:
{cv_text}

Extract the information into the structured format. Be precise and don't fabricate information not present in the CV."""


JOB_EXTRACTION_PROMPT = """You are an expert job posting analyst. Extract structured requirements from the following job specification.

Pay special attention to:
1. REQUIRED vs PREFERRED/NICE-TO-HAVE skills - be precise about this distinction
2. Technical skills vs soft skills
3. Industry-specific terminology and keywords (these are important for ATS)
4. Company culture indicators
5. Seniority level expectations
6. Years of experience required

For keywords:
- Extract terms that would likely be used by ATS (Applicant Tracking Systems)
- Include technical terms, tools, frameworks, methodologies
- Note any certifications or qualifications mentioned

JOB SPECIFICATION:
{job_spec_text}

Extract the information into the structured format. Be precise about distinguishing required vs preferred qualifications."""
