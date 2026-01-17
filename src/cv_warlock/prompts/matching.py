"""Matching and analysis prompt templates."""

MATCH_ANALYSIS_PROMPT = """You are a career coach expert at analyzing CV-job fit.

Analyze how well the candidate's CV matches the job requirements.

CANDIDATE'S CV DATA:
{cv_data}

JOB REQUIREMENTS:
{job_requirements}

Provide a detailed analysis:

1. STRONG MATCHES: Skills, experiences, or qualifications that directly match requirements.
   List specific items from the CV that align with specific job requirements.

2. PARTIAL MATCHES: Related skills or experiences that could be positioned as relevant.
   These are transferable skills or adjacent technologies.

3. GAPS: Important requirements the candidate lacks.
   Focus on required qualifications, not nice-to-haves.

4. TRANSFERABLE SKILLS: Skills from other contexts that apply here.
   Consider how experience in one area could translate to the role.

5. RELEVANCE SCORE: Overall match score from 0 to 1.
   - 0.8-1.0: Excellent match, strong candidate
   - 0.6-0.8: Good match, some gaps
   - 0.4-0.6: Moderate match, significant gaps
   - Below 0.4: Weak match

Be specific and reference actual items from both the CV and job spec."""


TAILORING_PLAN_PROMPT = """You are a professional CV writer creating a tailoring strategy.

Based on the match analysis, create a plan to tailor this CV for the target role.

MATCH ANALYSIS:
{match_analysis}

CV DATA:
{cv_data}

JOB REQUIREMENTS:
{job_requirements}

Create a tailoring plan that includes:

1. SUMMARY FOCUS: Key points to emphasize in the professional summary.
   What should the opening statement highlight to grab attention?

2. EXPERIENCES TO EMPHASIZE: Which roles/experiences are most relevant?
   Rank them by relevance to the target position.

3. SKILLS TO HIGHLIGHT: Which skills should be featured prominently?
   Prioritize required skills, then preferred skills.

4. ACHIEVEMENTS TO FEATURE: Which achievements best demonstrate fit?
   Look for quantifiable results that align with job responsibilities.

5. KEYWORDS TO INCORPORATE: ATS-friendly terms from the job posting.
   These should be naturally woven into the CV content.

6. SECTIONS TO REORDER: Optimal ordering of CV sections.
   Put most relevant content first.

The goal is to create a tailored CV that passes ATS screening and captures human attention, while remaining 100% truthful about the candidate's background."""
