"""Chain-of-thought prompts for cover letter generation.

These prompts implement a REASON -> GENERATE pattern for producing
plain text cover letters suitable for job application text areas.
The output is intentionally plain text (no markdown) to paste directly
into web forms.
"""

# =============================================================================
# COVER LETTER PROMPTS
# =============================================================================

COVER_LETTER_REASONING_PROMPT = """You are an expert cover letter strategist for technology leadership roles (CTO, VP Engineering, IT Director, Head of AI).

=== 2026 HIRING CONTEXT ===
- 83% of hiring managers read cover letters (nearly half BEFORE the resume)
- 72% prioritize letters tailored to THEIR company
- A personalized letter distinguishes you from generic AI-generated text

=== INPUTS ===

TAILORED CV:
{tailored_cv}

TARGET: {job_title} at {company}

REQUIREMENTS:
- Required: {required_skills}
- Preferred: {preferred_skills}
- Responsibilities: {responsibilities}

MATCH ANALYSIS:
- Strong: {strong_matches}
- Transferable: {transferable_skills}

=== REASON THROUGH EACH ELEMENT ===

1. **OPENING HOOK** (critical - first sentence determines if they read on)
   - Reference specific company initiative, product, or recent news
   - AVOID: "I am writing to apply..." or "I am interested in..."
   - GOOD: "[Company]'s [specific initiative] resonates with my experience in..."

2. **KEY SELLING POINTS** (top 3 achievements matching job requirements)
   - Must have quantifiable impact
   - Must directly address their stated needs

3. **STRONGEST ALIGNMENT** (the ONE thing that makes you perfect)

4. **COMPANY CONNECTION** (why THIS company, not generic praise)

5. **LEADERSHIP QUALITIES** (pick 2-3 most relevant):
   - Visionary Leadership: strategic foresight, roadmaps aligned with business
   - Adaptability: leading through change, agile mindset
   - AI Fluency: championing AI/ML and emerging technologies
   - Data-Driven: analytical approach, metrics-focused decisions
   - Collaboration: bridging tech and business stakeholders
   - People Development: building high-performing teams

6. **PROBLEM-SOLUTION FRAMING** (how you solve THEIR challenge)

7. **4-PARAGRAPH STRUCTURE**:
   - P1: Hook + role interest + strongest qualification
   - P2: Key achievement with metric + solves their problem
   - P3: Leadership qualities + skills alignment + unique value
   - P4: Value proposition + enthusiasm + interview request

8. **KEYWORDS** (max 5 job posting terms to weave in naturally)

9. **METRIC TO FEATURE** (most impressive number from CV)

10. **CALL TO ACTION** ("I welcome the opportunity to discuss...")

11. **TONE**: Confident but not arrogant, strategic not tactical

12. **AVOID**: Salary, generic praise, desperation, markdown, clichés without proof

Output structured reasoning per CoverLetterReasoning schema."""


COVER_LETTER_GENERATION_PROMPT = """Generate a plain text cover letter for a tech leadership role.

=== REASONING (follow this strategy) ===
{reasoning_json}

=== CONSTRAINTS ===
- CHARACTER LIMIT: {character_limit} (STRICT)
- PLAIN TEXT ONLY: No markdown, bullets, headers, or formatting
- STRUCTURE: 4 paragraphs with blank line between each
- METRIC: Feature this achievement: {metric_to_feature}
- KEYWORDS: Incorporate naturally: {keywords}

=== 4-PARAGRAPH FORMULA ===
P1: Company-specific hook + role interest + strongest qualification
P2: Key achievement with metric + solves THEIR problem (problem-solution framing)
P3: Leadership qualities ({leadership_qualities}) + skills alignment + unique value
P4: Value proposition + enthusiasm + interview request ("I welcome the opportunity to discuss...")

=== OPENING (first sentence is critical) ===
GOOD: "[Company]'s [specific initiative] resonates with my experience..."
GOOD: "My track record of [result] directly addresses [Company]'s need for..."
AVOID: "I am writing to apply...", "I am interested in...", "Dear Hiring Manager..."

=== OUTPUT ===
Write ONLY the cover letter body. No salutation, no signature.
Plain text paragraphs separated by blank lines.
Stay under {character_limit} characters."""


COVER_LETTER_CRITIQUE_PROMPT = """Critique this cover letter as a hiring manager for a senior technology leadership role who reviews 100+ applications daily.

=== THE COVER LETTER ===
{generated_cover_letter}

=== TARGET ROLE ===
{job_title} at {company}

=== CHARACTER LIMIT ===
{character_limit} characters

=== CURRENT LENGTH ===
{current_length} characters

=== CRITIQUE CHECKLIST (TECH LEADERSHIP FOCUS) ===

Evaluate each criterion with brutal honesty:

1. **Opening Hook**: Does the FIRST sentence grab attention with company-specific relevance? (Not "Dear..." or "I am writing...")

2. **Company Research**: Does it show specific knowledge of THIS company? (Not generic praise like "great culture")

3. **Quantified Achievement**: Is there at least ONE hard metric demonstrating leadership impact?

4. **Problem-Solution Framing**: Does it frame the candidate's experience as solving a challenge the company faces?

5. **Leadership Qualities**: Does it convey executive-level qualities (strategic vision, team building, cross-functional influence)?

6. **Keyword Matching**: Are 2-3 job posting terms naturally incorporated?

7. **Character Limit**: Is it within {character_limit} characters?

8. **Professional Tone**: Confident but not arrogant? Strategic but not verbose? Executive-level language?

9. **Call to Action**: Clear closing requesting interview with enthusiasm for their mission?

10. **Plain Text**: NO markdown, bullets, headers, or special formatting?

=== QUALITY ASSESSMENT ===

Rate overall quality:
- EXCELLENT: All 10 checks pass, compelling executive-level narrative
- GOOD: 8-9 checks pass, minor refinement needed
- NEEDS_IMPROVEMENT: 5-7 checks pass, significant revision required
- POOR: Less than 5 checks pass, rewrite needed

If not EXCELLENT:
- List each specific issue found
- Provide actionable fix for each issue

Set should_refine=true if quality is below GOOD or over character limit."""


COVER_LETTER_REFINE_PROMPT = """Refine this cover letter for a technology leadership role based on the critique. Fix identified issues while preserving what works.

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
- Must demonstrate leadership qualities: {leadership_qualities}
- Keep strong elements, fix weak ones
- Maintain executive-level tone and strategic framing
- Ensure problem-solution framing connects your experience to their needs

Write ONLY the improved cover letter. No explanations."""
