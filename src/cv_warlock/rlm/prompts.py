"""RLM-specific prompts for orchestration."""

# System prompt for the root model that orchestrates the RLM workflow
RLM_SYSTEM_PROMPT = """You are an expert CV analyst with access to a Python environment.

## Your Task
{task}

## Available Variables
The following variables are pre-loaded in your environment:
{context_summary}

## Available Functions
- `rlm_query(text, question)`: Ask a sub-model to analyze a piece of text. Use this for deep analysis of specific chunks.
- `find_keyword(keyword, text)`: Find all occurrences of a keyword in text. Returns list of (start, end) positions.
- `find_sections(text)`: Parse text into sections based on headers. Returns dict of section_name -> content.

## Important Rules
1. Do NOT try to read or print entire cv_text or job_text in one go - they may be very large
2. Use Python code to explore content strategically (slicing, searching, iterating)
3. Call `rlm_query()` when you need deep semantic analysis of a specific section
4. Store intermediate findings in variables (e.g., `skills_found = [...]`)
5. When you have completed your analysis, output your final answer using: `FINAL(your_answer)`

## Output Format
Your responses should contain either:
1. **Python code** in ```python blocks - will be executed and output shown
2. **Text analysis** - your reasoning about what you've found
3. **FINAL(answer)** - your final answer when analysis is complete

## Strategy
1. First, get an overview of the document structure
2. Identify relevant sections for the task
3. Use keyword search to find specific evidence
4. Use rlm_query() for sections that need interpretation
5. Compile findings and provide final answer

Begin your analysis:"""

# Prompt for sub-model queries
RLM_SUB_QUERY_PROMPT = """Analyze the following text and answer the question.

## Text to Analyze
{context}

## Question
{question}

## Instructions
- Provide a concise, factual answer based ONLY on the text provided
- If the text doesn't contain the answer, say "Not found in provided text"
- Be specific and quote relevant parts when helpful
- Keep your response under 500 words

## Answer"""

# Task prompt for CV-job matching analysis
RLM_MATCH_ANALYSIS_TASK = """Analyze how well this CV matches the job requirements.

For each job requirement:
1. Search the CV for relevant evidence
2. Assess the strength of the match (strong/partial/no match)
3. Note any transferable skills that could apply

At the end, provide:
- List of strong matches with evidence
- List of partial matches with explanation
- List of gaps (requirements not met)
- Overall match score (0-100)
- Key recommendations for tailoring

Output your final analysis as a structured summary using FINAL(your_analysis)."""

# Task prompt for CV extraction with RLM
RLM_CV_EXTRACTION_TASK = """Extract structured information from this CV.

Extract the following:
1. Contact information (name, email, phone, location, links)
2. Professional summary (if present)
3. Work experiences (for each: title, company, dates, description, achievements)
4. Education (for each: degree, institution, date, GPA if mentioned)
5. Skills (list all technical and soft skills)
6. Certifications and projects (if any)

For long CVs, process section by section. Store findings in variables as you go.

Output your final extraction as a structured JSON using FINAL(extracted_data)."""

# Task prompt for job spec extraction with RLM
RLM_JOB_EXTRACTION_TASK = """Extract structured requirements from this job specification.

Extract the following:
1. Job title and company
2. Required qualifications (must-have skills, experience, education)
3. Preferred qualifications (nice-to-have)
4. Key responsibilities
5. Experience requirements (years, specific domains)
6. Keywords that would be important for ATS matching
7. Company values or culture indicators

Process section by section. Store findings in variables as you go.

Output your final extraction as structured JSON using FINAL(extracted_requirements)."""

# Task prompt for experience tailoring
RLM_TAILOR_EXPERIENCE_TASK = """Tailor this work experience to better match the job requirements.

Job Requirements Summary:
{job_summary}

For the experience provided:
1. Identify which job requirements this experience addresses
2. Rewrite bullet points to emphasize relevant achievements
3. Incorporate relevant keywords from the job description
4. Quantify achievements where possible
5. Keep the core facts accurate - don't fabricate

The experience to tailor:
{experience_text}

Output your tailored version using FINAL(tailored_experience)."""

# Task prompt for skills tailoring
RLM_TAILOR_SKILLS_TASK = """Organize and tailor skills to match the job requirements.

Job Requirements:
- Required: {required_skills}
- Preferred: {preferred_skills}

Current Skills from CV:
{cv_skills}

Task:
1. Identify skills from CV that match requirements (even if worded differently)
2. Group skills into logical categories
3. Order by relevance to the job
4. Add any skills from job requirements the candidate likely has based on their experience
5. Remove irrelevant skills that might dilute the CV

Output your tailored skills section using FINAL(tailored_skills)."""

# Continuation prompt when model needs to continue analysis
RLM_CONTINUE_PROMPT = """Continue your analysis.

Current state:
{state_summary}

Remember:
- You can execute more Python code
- You can call rlm_query() for deeper analysis
- When done, use FINAL(your_answer)

Continue:"""
