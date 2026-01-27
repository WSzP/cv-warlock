"""Chain-of-thought reasoning prompts for CV generation.

These prompts implement a REASON -> GENERATE -> CRITIQUE -> REFINE pattern
for each CV section. The prompts elicit explicit step-by-step thinking
before generation, enabling higher quality output through structured reasoning.

Note: CoT generation is slower than direct generation (3-4x more LLM calls)
but produces significantly higher quality tailored CVs.
"""

# =============================================================================
# SUMMARY PROMPTS
# =============================================================================

SUMMARY_REASONING_PROMPT = """You are an expert CV strategist. Before writing the professional summary, you must REASON through your approach step-by-step.

=== INPUTS ===

ORIGINAL SUMMARY:
{original_summary}

TARGET ROLE: {job_title} at {company}

KEY REQUIREMENTS: {key_requirements}

CANDIDATE STRENGTHS: {relevant_strengths}

TAILORING PLAN HIGHLIGHTS:
{tailoring_plan_summary}

=== YOUR TASK: REASON BEFORE GENERATING ===

Think through each step carefully:

1. **TITLE POSITIONING**: How should we position the candidate's title?
   - What's the closest match to "{job_title}"?
   - Should we use their exact title or adapt it slightly?

2. **KEYWORD SELECTION**: Which exact phrases from the job posting MUST appear?
   - Identify the top 5 terms that ATS systems will scan for
   - Confirm the candidate genuinely has these skills

3. **METRIC SELECTION**: What's the single most impressive relevant number?
   - Review their achievements for: revenue, scale, %, time metrics
   - Which metric best proves they can do THIS specific job?

4. **DIFFERENTIATOR**: What makes this candidate unique for THIS role?
   - Not generic strengths - specific to this job match
   - What would make a recruiter think "I need to talk to this person"?

5. **HOOK STRATEGY**: Plan the opening sentence using:
   [Title] + [years of experience] + [domain expertise] + [differentiator]
   - The first 10 words must grab attention immediately

6. **VALUE PROPOSITION**: What's the core value statement?
   - Must include the strongest metric
   - Shows concrete impact, not vague claims

7. **FIT STATEMENT**: How to connect to specific job requirements?
   - Bridge their experience to what THIS role needs

8. **RISK ASSESSMENT**: What should be avoided?
   - Employment gaps that shouldn't be mentioned
   - Irrelevant experience that dilutes the message
   - Fluffy adjectives (passionate, driven, motivated, dedicated)

Provide your reasoning as structured output following the SummaryReasoning schema."""


SUMMARY_GENERATION_PROMPT = """You are an elite CV writer. Generate a professional summary based on the strategic reasoning provided.

*** CRITICAL: NEVER USE EM DASHES (—) OR EN DASHES (–) ANYWHERE IN YOUR OUTPUT ***
Use commas, or semicolons instead. This is a HARD requirement.

=== YOUR REASONING ===
{reasoning_json}

=== CONSTRAINTS (STRICT) ===

1. Maximum 4 sentences - brevity signals confidence
2. First sentence MUST be a hook: [Title] + [years] + [domain] + [differentiator]
3. MUST include this metric: {strongest_metric}
4. MUST include these keywords naturally: {keywords}
5. NO fluffy adjectives: passionate, driven, motivated, dedicated, hardworking
6. NO "I am" or "I have" openings - waste of prime real estate
7. NO objectives about what YOU want - focus on what you DELIVER

=== OUTPUT ===

Write ONLY the summary paragraph. No explanations, no preamble."""


SUMMARY_CRITIQUE_PROMPT = """You are a senior recruiter who has reviewed 10,000+ CVs. Critique this summary honestly and specifically.

=== THE SUMMARY ===
{generated_summary}

=== TARGET ROLE ===
{job_title} at {company}

=== KEYWORDS THAT SHOULD APPEAR ===
{required_keywords}

=== CRITIQUE CHECKLIST ===

Evaluate each criterion (true/false) with brutal honesty:

1. **Opening Hook**: Does the FIRST sentence immediately establish relevant identity and create interest? (Not generic, not weak)

2. **Quantified Achievement**: Is there at least ONE hard number (%, $, users, transactions, team size, years)?

3. **Keyword Matching**: Are at least 2-3 EXACT terms from the job posting naturally present? (Not synonyms - exact matches)

4. **Length**: Is it 2-4 sentences total? (Not longer - recruiters won't read more)

5. **No Fluff**: Is it completely free of weak words: "passionate", "driven", "motivated", "dedicated", "hardworking", "team player"?

=== QUALITY ASSESSMENT ===

Rate overall quality:
- EXCELLENT: All 5 checks pass, would definitely get interview callback
- GOOD: 4/5 checks pass, minor polish needed
- NEEDS_IMPROVEMENT: 2-3 checks pass, significant revision required
- POOR: Less than 2 checks pass, complete rewrite needed

If not EXCELLENT:
- List each specific issue found
- Provide actionable fix for each issue

Set should_refine=true if quality is below GOOD."""


SUMMARY_REFINE_PROMPT = """Refine this professional summary based on the critique. Fix the identified issues while preserving what works.

*** CRITICAL: NEVER USE EM DASHES (—) OR EN DASHES (–) ANYWHERE IN YOUR OUTPUT ***
Use commas or semicolons instead. This is a HARD requirement.

=== CURRENT SUMMARY ===
{current_summary}

=== ISSUES TO FIX ===
{issues}

=== SPECIFIC IMPROVEMENTS NEEDED ===
{suggestions}

=== ORIGINAL REASONING (follow this strategy) ===
{reasoning_json}

=== CONSTRAINTS (unchanged) ===
- Maximum 4 sentences
- Must include metric: {strongest_metric}
- Must include keywords: {keywords}
- No fluffy adjectives
- No "I am" / "I have" openings

Write ONLY the improved summary. No explanations."""


# =============================================================================
# EXPERIENCE PROMPTS
# =============================================================================

EXPERIENCE_REASONING_PROMPT = """You are a CV strategist. Before tailoring this experience entry, REASON through your approach step-by-step.

=== EXPERIENCE TO TAILOR ===
Title: {title}
Company: {company}
Period: {period}
Description: {description}
Achievements:
{achievements}

=== TARGET ROLE ===
{job_title}

Requirements: {target_requirements}

=== SKILLS TO EMPHASIZE ===
{skills_to_emphasize}

=== CONTEXT FROM PREVIOUS SECTIONS ===
Identity established in summary: {established_identity}
Keywords already heavily used: {keywords_already_used}
Metrics already featured: {metrics_already_used}

=== REASONING TASKS ===

Think through each step:

1. **RELEVANCE SCORE** (0.0 to 1.0): How relevant is this experience to the target role?
   - Is it the same function/domain?
   - Are the responsibilities similar?
   - Would a recruiter see clear connection?

2. **EMPHASIS STRATEGY**: Based on relevance score (keep CV scannable!):
   - HIGH (0.7+): 4-5 bullets max - cornerstone experience
   - MEDIUM (0.4-0.7): 3 focused bullets - highlight transferable aspects only
   - LOW (<0.4): 2 brief bullets - minimal coverage

3. **ACHIEVEMENT PRIORITIZATION**: Which achievements prove fit for target role?
   - Rank by direct relevance to job requirements
   - Identify which have quantifiable metrics
   - Select top 3-5 based on emphasis strategy

4. **KEYWORD INJECTION PLAN**: Which job posting terms fit naturally here?
   - Don't repeat keywords already overused in summary
   - Aim for 2-3 NEW relevant terms per experience
   - Must feel natural, not forced

5. **BULLET-BY-BULLET REASONING**: For each bullet you'll create:
   - What's the original content?
   - How does it relate to the job?
   - What metric can be highlighted?
   - Which power verb to use?
   - Which keywords to inject?
   - What's the reframed version?

6. **ASPECTS TO DOWNPLAY**: What from this role doesn't help the application?
   - Irrelevant technologies
   - Unrelated responsibilities
   - Anything that might raise concerns

Provide structured reasoning output following the ExperienceReasoning schema."""


BATCH_EXPERIENCE_REASONING_PROMPT = """You are a CV strategist. Analyze ALL the following experiences and provide reasoning for each one in a SINGLE response.

=== TARGET ROLE ===
{job_title}

Key Requirements: {target_requirements}

Skills to Emphasize: {skills_to_emphasize}

=== EXPERIENCES TO ANALYZE ===
{experiences_text}

=== CONTEXT FROM PREVIOUS SECTIONS ===
Identity established in summary: {established_identity}
Keywords already heavily used: {keywords_already_used}
Metrics already featured: {metrics_already_used}

=== YOUR TASK ===

For EACH experience listed above, provide reasoning following these steps:

1. **RELEVANCE SCORE** (0.0 to 1.0): How relevant is this experience to the target role?

2. **EMPHASIS STRATEGY**: Based on relevance (keep CV scannable!):
   - HIGH (0.7+): 4-5 bullets max - cornerstone experience
   - MEDIUM (0.4-0.7): 3 focused bullets
   - LOW (<0.4): 2 brief bullets

3. **ACHIEVEMENT PRIORITIZATION**: Which achievements prove fit for target role?

4. **KEYWORD INJECTION PLAN**: Which job posting terms fit naturally?

5. **BULLET-BY-BULLET REASONING**: For each bullet you'll create:
   - Original content, relevance, metric, power verb, keywords, reframed version

6. **ASPECTS TO DOWNPLAY**: What doesn't help the application?

Return a BatchExperienceReasoning with reasoning for ALL experiences. Match each reasoning to the correct experience_index (0-based)."""


EXPERIENCE_GENERATION_PROMPT = """Generate experience bullets based on your strategic reasoning.

*** CRITICAL: NEVER USE EM DASHES (—) OR EN DASHES (–) ANYWHERE IN YOUR OUTPUT ***
Use commas or semicolons instead. This is a HARD requirement.

=== YOUR REASONING ===
{reasoning_json}

=== TARGET NUMBER OF BULLETS ===
{bullet_count} (based on your emphasis strategy)

=== KEYWORDS TO INCORPORATE ===
{keywords_to_use}

=== BULLET FORMULA (every bullet must follow this) ===
[Power Verb] + [What you did] + [Quantified Result/Impact] + [Scale/Context]

Examples:
GOOD: "Reduced API latency by 40% by implementing Redis caching, improving experience for 2M+ daily users"
BAD: "Responsible for API performance improvements" (passive, no metrics)

GOOD: "Led migration of 15 microservices to Kubernetes, achieving 99.99% uptime and cutting costs $180K/year"
BAD: "Worked on Kubernetes migration" (vague, no impact)

=== CONSTRAINTS (STRICT) ===
1. EXACTLY {bullet_count} bullets - no more, no less
2. Every bullet starts with power verb (past tense for past roles)
3. Every bullet shows measurable impact where data exists
4. Max 15-20 words per bullet - must fit on 1-2 lines
5. Order by relevance to target role (most relevant FIRST)
6. Keywords NATURALLY incorporated - don't force them

=== OUTPUT FORMAT ===
- [Bullet 1]
- [Bullet 2]
- [Bullet 3]
...

Write ONLY the bullet points. No explanations, no headers."""


EXPERIENCE_CRITIQUE_PROMPT = """Critique these experience bullets as a senior recruiter with high standards.

=== THE BULLETS ===
{generated_bullets}

=== FOR TARGET ROLE ===
{job_title}

=== JOB REQUIREMENTS ===
{job_requirements}

=== CRITIQUE CHECKLIST ===

For EACH bullet, verify:
1. Starts with power verb (past tense)?
2. Shows clear impact/result (not just duties)?
3. Includes metric if the achievement originally had one?
4. Contains at least one relevant keyword?
5. Under 20 words?

Overall checks:
6. Bullets ordered by relevance (most relevant first)?
7. No redundant or overlapping bullets?
8. All bullets pass the "SO WHAT?" test - clear why it matters?

=== ASSESSMENT ===

Rate overall quality and specifically identify:
- Which bullets are strong (keep as-is)
- Which bullets are weak (with specific reason)
- What improvements would elevate quality

Set should_refine=true if any bullet fails multiple checks or overall quality is below GOOD."""


EXPERIENCE_REFINE_PROMPT = """Refine these experience bullets based on the critique.

*** CRITICAL: NEVER USE EM DASHES (—) OR EN DASHES (–) ANYWHERE IN YOUR OUTPUT ***
Use commas or semicolons instead. This is a HARD requirement.

=== CURRENT BULLETS ===
{current_bullets}

=== WEAK BULLETS IDENTIFIED ===
{weak_bullets}

=== IMPROVEMENTS NEEDED ===
{suggestions}

=== ORIGINAL REASONING (follow this strategy) ===
{reasoning_json}

=== CONSTRAINTS ===
- Keep strong bullets unchanged
- Fix weak bullets per critique
- Maintain {bullet_count} total bullets
- Every bullet: Power verb + Action + Impact + Context
- Incorporate keywords: {keywords_to_use}

=== OUTPUT FORMAT ===
- [Bullet 1]
- [Bullet 2]
...

Write ONLY the refined bullet points."""


# =============================================================================
# SKILLS PROMPTS
# =============================================================================

SKILLS_REASONING_PROMPT = """You are an ATS optimization expert. REASON through the skills section strategy to maximize ATS score and recruiter impact.

=== CANDIDATE'S VERIFIED SKILLS ===
{all_skills}

=== JOB REQUIREMENTS ===
Required (must-have): {required_skills}
Preferred (nice-to-have): {preferred_skills}

=== CONTEXT FROM OTHER SECTIONS ===
Skills already demonstrated in experience bullets: {skills_from_experience}
Keywords already used frequently: {keywords_used}

=== REASONING TASKS ===

Think through each step:

1. **MATCH ANALYSIS**:
   - Which required skills does candidate DEFINITELY have? (exact matches only)
   - Which required skills are missing? (gaps to be aware of)
   - Which preferred skills does candidate have?

2. **TERMINOLOGY DECISIONS**:
   - Map candidate's terms to job posting's EXACT terms
   - If candidate says "AWS" but job says "Amazon Web Services", use job's version
   - If candidate says "ML" but job says "Machine Learning", use job's version
   - Create mapping: candidate_term -> job_posting_term

3. **DUAL FORMAT TERMS**:
   - Which acronyms benefit from including full form?
   - Examples: "Machine Learning (ML)", "Amazon Web Services (AWS)"
   - This catches both ATS search variants

4. **CATEGORIZATION STRATEGY**:
   - How to group skills logically?
   - Match job posting's structure if possible
   - Common categories: Languages, Frameworks, Cloud/DevOps, Databases, Tools
   - Order within categories: most relevant first

5. **ORDERING PRIORITY**:
   - Required skill matches first
   - Preferred skill matches second
   - Relevant transferable skills third
   - Rationale for this specific ordering?

6. **OMISSIONS**:
   - Which candidate skills are IRRELEVANT to this specific role?
   - Including them dilutes the relevance signal
   - Be aggressive about cutting irrelevant skills

7. **TENSORFLOW RULE**:
   - ONLY include TensorFlow if it is EXPLICITLY mentioned in the job requirements
   - If TensorFlow is NOT in the job spec but a deep learning framework seems beneficial, use PyTorch instead
   - This applies even if the candidate lists TensorFlow in their skills

Provide structured reasoning output following the SkillsReasoning schema."""


SKILLS_GENERATION_PROMPT = """Generate the skills section based on your strategic reasoning.

*** CRITICAL: NEVER USE EM DASHES (—) OR EN DASHES (–) ANYWHERE IN YOUR OUTPUT ***
Use commas or semicolons instead. This is a HARD requirement.

=== YOUR REASONING ===
{reasoning_json}

=== CONSTRAINTS (STRICT) ===

1. Use EXACT terminology from job posting (not synonyms)
2. Include ALL matched required skills
3. Include matched preferred skills
4. Group by logical categories
5. Order: Required matches -> Preferred matches -> Relevant transferable
6. Include dual formats for key acronyms: "AWS (Amazon Web Services)"
7. Do NOT include skills candidate doesn't have (no fabrication)
8. Do NOT include irrelevant skills (they dilute the signal)
9. TENSORFLOW RULE: Only include TensorFlow if EXPLICITLY required in job posting.
   If deep learning framework is needed but TensorFlow not specified, use PyTorch instead.

=== OUTPUT FORMAT (CRITICAL RULES - MUST follow EXACTLY) ===

1. EVERY category name MUST be wrapped in ** for bold: **Category:**
2. Category and skills MUST be on the SAME LINE
3. Use colon INSIDE the bold: **Category:** not **Category**:

## Technical Skills

**[Category 1]:** skill1, skill2, skill3
**[Category 2]:** skill1, skill2, skill3
**[Category 3]:** skill1, skill2, skill3

Example:
## Technical Skills

**Languages:** Python, TypeScript, SQL, Go
**Frameworks:** React.js, FastAPI, Django, Node.js
**Cloud & DevOps:** Amazon Web Services (AWS), Docker, Kubernetes, CI/CD
**Databases:** PostgreSQL, MongoDB, Redis, DynamoDB
**Tools:** Git, JIRA, Terraform, Datadog

WRONG - missing bold on category:
Languages: Python, TypeScript, SQL

WRONG - colon outside bold:
**Languages**: Python, TypeScript, SQL

WRONG - skills on separate line:
**Languages**
Python, TypeScript, SQL

CORRECT - bold category with colon inside, skills on same line:
**Languages:** Python, TypeScript, SQL

Write ONLY the formatted skills section. No explanations."""


SKILLS_CRITIQUE_PROMPT = """Critique this skills section for ATS optimization as a technical recruiter.

=== THE SKILLS SECTION ===
{generated_skills}

=== JOB REQUIREMENTS ===
Required: {required_skills}
Preferred: {preferred_skills}

=== CANDIDATE'S ACTUAL VERIFIED SKILLS ===
{candidate_skills}

=== CRITIQUE CHECKLIST ===

1. **Completeness**: All required skills (that candidate has) present?
   - List any missing required skills

2. **Terminology**: Uses EXACT job posting wording?
   - Flag any terms using synonyms instead of exact matches

3. **Categorization**: Skills grouped logically?
   - Does structure help or hinder scanning?

4. **Relevance**: No irrelevant skills diluting the section?
   - Flag any skills that don't serve this application

5. **Integrity**: No fabricated skills?
   - Cross-check against candidate's verified skills

=== ASSESSMENT ===

Rate quality and identify:
- Missing critical terms that must be added
- Terms using wrong terminology
- Skills that should be removed

Set should_refine=true if any required skill is missing or wrong terminology is used."""


SKILLS_REFINE_PROMPT = """Refine this skills section based on the critique.

*** CRITICAL: NEVER USE EM DASHES (—) OR EN DASHES (–) ANYWHERE IN YOUR OUTPUT ***
Use commas or semicolons instead. This is a HARD requirement.

=== CURRENT SKILLS SECTION ===
{current_skills}

=== MISSING CRITICAL TERMS ===
{missing_terms}

=== IMPROVEMENTS NEEDED ===
{suggestions}

=== ORIGINAL REASONING ===
{reasoning_json}

=== CANDIDATE'S ACTUAL SKILLS ===
{candidate_skills}

=== CONSTRAINTS ===
- Add missing required skills (only if candidate has them)
- Fix terminology to match job posting exactly
- Remove irrelevant skills flagged in critique
- Maintain logical categorization

Write ONLY the refined skills section."""
