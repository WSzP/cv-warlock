"""Chain-of-thought prompts for cover letter generation.

These prompts implement a REASON -> GENERATE pattern for producing
plain text cover letters suitable for job application text areas.
The output is intentionally plain text (no markdown) to paste directly
into web forms.
"""

# =============================================================================
# COVER LETTER PROMPTS
# =============================================================================

COVER_LETTER_REASONING_PROMPT = """You are an expert cover letter strategist. Before writing the cover letter, REASON through your approach step-by-step.

=== INPUTS ===

TAILORED CV (already optimized for this role):
{tailored_cv}

TARGET ROLE: {job_title} at {company}

JOB REQUIREMENTS:
Required Skills: {required_skills}
Preferred Skills: {preferred_skills}
Key Responsibilities: {responsibilities}

CANDIDATE MATCH ANALYSIS:
Strong Matches: {strong_matches}
Transferable Skills: {transferable_skills}

=== YOUR TASK: REASON BEFORE GENERATING ===

Think through each step carefully:

1. **OPENING HOOK**: How to start compellingly?
   - What specific connection to THIS company/role?
   - Avoid generic "I am writing to apply for..."
   - What would make them want to keep reading?

2. **KEY SELLING POINTS**: What are the top 3 achievements/skills to feature?
   - Must directly address job requirements
   - Should have quantifiable impact where possible
   - What proof points from CV demonstrate fit?

3. **STRONGEST ALIGNMENT**: What's the single best match?
   - The one thing that makes this candidate perfect for THIS role
   - Should be specific and compelling

4. **COMPANY CONNECTION**: Why THIS company specifically?
   - What specific aspect of the company/role is genuinely appealing?
   - Avoid generic flattery ("great company culture")
   - Show you've done research

5. **PARAGRAPH STRUCTURE**: Plan 3-4 paragraphs:
   - Para 1: Hook + role interest + strongest qualification
   - Para 2: Relevant achievement with metrics
   - Para 3: Skills alignment + additional value
   - Para 4: Call to action + closing

6. **KEYWORD INCORPORATION**: Which job terms to weave in?
   - Max 5 key terms from job posting
   - Must feel natural, not forced
   - Different from what's heavily used in CV

7. **METRIC TO FEATURE**: What's the most impressive relevant number?
   - Must be from the CV (no fabrication)
   - Should directly relate to job responsibilities

8. **CALL TO ACTION**: How to close confidently?
   - Request specific next step (interview)
   - Express enthusiasm without desperation
   - Keep it professional

9. **TONE GUIDANCE**: What tone to strike?
   - Confident but not arrogant
   - Specific but not verbose
   - Professional but personable

10. **ASPECTS TO AVOID**:
    - Salary discussion
    - Generic company praise
    - Desperation signals
    - Repeating entire CV
    - Markdown or formatting

Provide structured reasoning following the CoverLetterReasoning schema."""


COVER_LETTER_GENERATION_PROMPT = """Generate a professional cover letter based on your strategic reasoning.

=== YOUR REASONING ===
{reasoning_json}

=== TARGET CHARACTER LIMIT ===
{character_limit} characters (STRICT - count carefully)

=== CONSTRAINTS (STRICT) ===

1. **PLAIN TEXT ONLY**: No markdown, no bullet points, no headers, no bold/italic
2. **CHARACTER LIMIT**: Must be under {character_limit} characters including spaces
3. **STRUCTURE**: 3-4 paragraphs with blank line between each
4. **OPENING**: Start with compelling hook, NOT "Dear Hiring Manager" or "I am writing to apply"
5. **METRIC**: Include this achievement: {metric_to_feature}
6. **KEYWORDS**: Incorporate naturally: {keywords}
7. **CLOSING**: End with clear call to action requesting interview
8. **NO FABRICATION**: Only use facts from the provided CV

=== OPENING ALTERNATIVES (vary based on context) ===

Good openings:
- "When I saw [Company] was looking for a [Role]..."
- "My experience [specific achievement] directly aligns with [Company]'s need for..."
- "Having [relevant experience], I was excited to see [Company]'s [Role] opening..."
- "[Specific company achievement/news] caught my attention, and I believe my background in..."

Avoid:
- "I am writing to apply for..."
- "Dear Hiring Manager..."
- "I am interested in..."
- "Please find attached..."

=== PARAGRAPH FORMULA ===

Para 1: Hook + specific interest in role + strongest single qualification
Para 2: Key achievement with metric + how it relates to their needs
Para 3: Additional skills alignment + unique value you bring
Para 4: Enthusiasm + clear interview request + professional close

=== OUTPUT ===

Write ONLY the cover letter text. No salutation header, no signature block.
Plain text paragraphs separated by blank lines.
Do not exceed {character_limit} characters."""


COVER_LETTER_CRITIQUE_PROMPT = """Critique this cover letter as a hiring manager who reviews 100+ applications daily.

=== THE COVER LETTER ===
{generated_cover_letter}

=== TARGET ROLE ===
{job_title} at {company}

=== CHARACTER LIMIT ===
{character_limit} characters

=== CURRENT LENGTH ===
{current_length} characters

=== CRITIQUE CHECKLIST ===

Evaluate each criterion with brutal honesty:

1. **Opening Hook**: Does the FIRST sentence grab attention? (Not "Dear..." or "I am writing...")

2. **Company Research**: Does it show specific knowledge of THIS company? (Not generic praise)

3. **Quantified Achievement**: Is there at least ONE hard metric from their background?

4. **Keyword Matching**: Are 2-3 job posting terms naturally incorporated?

5. **Character Limit**: Is it within {character_limit} characters?

6. **Professional Tone**: Confident but not arrogant? Specific but not verbose?

7. **Call to Action**: Clear closing requesting interview?

8. **Plain Text**: NO markdown, bullets, headers, or special formatting?

=== QUALITY ASSESSMENT ===

Rate overall quality:
- EXCELLENT: All 8 checks pass, compelling and differentiated
- GOOD: 6-7 checks pass, minor refinement needed
- NEEDS_IMPROVEMENT: 4-5 checks pass, significant revision required
- POOR: Less than 4 checks pass, rewrite needed

If not EXCELLENT:
- List each specific issue found
- Provide actionable fix for each issue

Set should_refine=true if quality is below GOOD or over character limit."""


COVER_LETTER_REFINE_PROMPT = """Refine this cover letter based on the critique. Fix identified issues while preserving what works.

=== CURRENT COVER LETTER ===
{current_cover_letter}

=== ISSUES TO FIX ===
{issues}

=== SPECIFIC IMPROVEMENTS NEEDED ===
{suggestions}

=== ORIGINAL REASONING ===
{reasoning_json}

=== CONSTRAINTS ===
- Maximum {character_limit} characters (STRICT)
- Plain text only - no markdown or formatting
- Must include metric: {metric_to_feature}
- Must include keywords: {keywords}
- Keep strong elements, fix weak ones

Write ONLY the improved cover letter. No explanations."""
