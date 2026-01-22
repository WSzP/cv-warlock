# Hybrid ATS Scoring System

CV Warlock uses a **hybrid scoring system** that combines deterministic algorithmic analysis with LLM-powered qualitative assessment. This approach provides the best of both worlds: reproducible, fast algorithmic scores plus nuanced human-like evaluation that captures transferable skills and contextual strengths.

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                   HYBRID SCORER                         │
                    └─────────────────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
        ┌───────────────────────┐                      ┌───────────────────────┐
        │  ALGORITHMIC SCORER   │                      │   LLM ASSESSMENT      │
        │  (Fast, Free, Local)  │                      │   (Qualitative)       │
        └───────────────────────┘                      └───────────────────────┘
                    │                                               │
        ┌───────────┴───────────┐                      ┌───────────┴───────────┐
        │                       │                      │                       │
        ▼                       ▼                      ▼                       ▼
┌───────────────┐   ┌───────────────┐       ┌───────────────┐   ┌───────────────┐
│ Exact Match   │   │  Years Fit    │       │ Transferable  │   │  Contextual   │
│ Score (35%)   │   │  Score (25%)  │       │    Skills     │   │  Strengths    │
└───────────────┘   └───────────────┘       └───────────────┘   └───────────────┘
        │                       │                      │                       │
        ▼                       ▼                      │                       │
┌───────────────┐   ┌───────────────┐                  └───────────┬───────────┘
│ Education     │   │  Recency      │                              │
│ Score (15%)   │   │  Score (25%)  │                              ▼
└───────────────┘   └───────────────┘                  ┌───────────────────────┐
        │                       │                      │  LLM Adjustment       │
        └───────────┬───────────┘                      │  (-0.10 to +0.10)     │
                    │                                  └───────────────────────┘
                    ▼                                               │
        ┌───────────────────────┐                                   │
        │  Algorithmic Total    │◄──────────────────────────────────┘
        │     (0.0 - 1.0)       │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  KNOCKOUT CHECK       │
        │  (Required Skills)    │
        └───────────────────────┘
                    │
            ┌───────┴───────┐
            │               │
            ▼               ▼
    [Missing Skills]   [All Present]
            │               │
            ▼               ▼
    Score = 0.0        Final Score
```

## Scoring Flow

1. **Compute Algorithmic Scores** - Fast, deterministic, no API calls
2. **Check Knockout Rules** - Auto-fail if required skills missing
3. **Get LLM Assessment** - Qualitative evaluation (if not knocked out)
4. **Combine Scores** - Apply LLM adjustment to algorithmic total

## Algorithmic Scoring (Deterministic)

The algorithmic scorer computes four sub-scores using exact string matching only. No external APIs or embeddings are required.

### Sub-Scores and Weights

| Sub-Score | Weight | Description |
|-----------|--------|-------------|
| **Exact Skill Match** | 35% | String match for required/preferred skills |
| **Experience Years Fit** | 25% | Candidate's years vs job requirement |
| **Education Match** | 15% | Education level vs requirement |
| **Recency Score** | 25% | Recent experience relevance (with decay) |

### Exact Skill Match (35%)

Calculates the percentage of job skills found in the CV:

```
Required Skills Score = (matched required skills) / (total required skills)
Preferred Skills Score = (matched preferred skills) / (total preferred skills)

Final Score = 0.7 × Required + 0.3 × Preferred
```

**Matching Logic:**
- Checks explicit skills list (case-insensitive)
- Checks experience descriptions for keyword mentions
- No semantic/embedding matching (exact only)

### Experience Years Fit (25%)

Compares candidate's total years of experience to job requirements:

| Candidate Years vs Required | Score |
|-----------------------------|-------|
| >= 100% of required | 1.0 |
| >= 70% of required | 0.8 |
| >= 50% of required | 0.5 |
| < 50% of required | `max(0.2, years/required)` |
| No requirement specified | 1.0 |

### Education Match (15%)

Compares candidate's highest education level to job requirements:

**Education Level Hierarchy:**
```
Level 5: PhD, Doctorate
Level 4: Master's, MBA, MSc, MS, MA
Level 3: Bachelor's, BSc, BS, BA
Level 2: Associate's, Diploma
Level 1: High School, GED
```

| Match Status | Score |
|--------------|-------|
| Meets or exceeds requirement | 1.0 |
| One level below | 0.7 |
| Has some education | 0.4 |
| No matching education | 0.2 |
| No requirement specified | 1.0 |

### Recency Score (25%)

Weights recent experience more heavily using exponential decay:

```python
recency_weight = 0.85 ^ years_ago  # ~15% decay per year
```

**Special Handling for Recent Experiences (within 2 years):**
- Overall CV skills are included in skill matching
- Rationale: If someone lists "Python" as a skill and is currently employed, they're likely using it in their current role

**Calculation:**
```
For each experience:
    1. Calculate recency weight (exponential decay)
    2. Calculate skill overlap with job requirements
    3. Count keyword mentions in description
    4. Combine: relevance = (skill_overlap + 0.5 × keyword_mentions) / target_skills
    5. Weight: weighted_score = relevance × recency_weight

Final = average(weighted_scores) × 1.5  # scaled up, capped at 1.0
```

## Knockout Rules

The knockout rule provides an **automatic fail** mechanism for candidates missing required skills.

### How It Works

1. Extract all required skills from job spec
2. Check each against CV (exact string match, case-insensitive)
3. If ANY required skill is missing: **knockout triggered**
4. Knockout results in **score = 0.0**

### Important Design Decisions

- **Exact matching only** - No semantic/embedding matching
- **Conservative approach** - If skill isn't explicitly listed, it counts as missing
- **ATS accuracy** - Mimics real ATS behavior which uses keyword matching

### Knockout Result

When knockout is triggered:
- `relevance_score = 0.0`
- `knockout_triggered = true`
- `knockout_reason = "Missing required skills: X, Y, Z"`
- `gaps = [list of missing skills]`

## LLM Qualitative Assessment

After algorithmic scoring (if not knocked out), an LLM provides qualitative assessment for factors algorithms cannot capture.

### What the LLM Evaluates

| Factor | Description |
|--------|-------------|
| **Transferable Skills** | Non-obvious skills that transfer (e.g., leadership from non-tech roles, domain knowledge) |
| **Contextual Strengths** | Career narrative that matches the role (progression, company types, project scale) |
| **Concerns** | Red flags algorithms miss (job hopping, skill depth vs breadth, overqualification) |
| **Adjustment** | Score modification recommendation (-0.10 to +0.10) |

### LLM Prompt Structure

The LLM receives:
1. Algorithmic sub-scores (so it knows what's already calculated)
2. Serialized CV data (without PII)
3. Serialized job requirements
4. Instructions to focus on qualitative insights

### Score Adjustment

The LLM recommends an adjustment between **-0.10 and +0.10**:

- **Positive adjustment**: Qualitative factors strengthen the match beyond what numbers show
- **Negative adjustment**: Concerns the algorithm missed
- **Zero adjustment**: Algorithmic score seems accurate

### Final Score Calculation

```python
final_score = clamp(algorithmic_total + llm_adjustment, 0.0, 1.0)
```

## Combined Analysis Optimization

For efficiency, CV Warlock can combine the LLM assessment and tailoring plan into a **single LLM call**:

```python
# Instead of:
assessment = get_llm_assessment()  # 1 LLM call
plan = get_tailoring_plan()        # 1 LLM call

# We do:
combined = get_combined_analysis()  # 1 LLM call (saves ~10-20 seconds)
```

This is implemented in `HybridScorer.score_with_plan()`.

## Result Structure

### HybridMatchResult

```python
class HybridMatchResult(TypedDict):
    # Core match data
    strong_matches: list[str]      # Skills with exact match
    partial_matches: list[str]     # Skills mentioned in experience
    gaps: list[str]                # Missing required skills
    transferable_skills: list[str] # From LLM assessment

    # Scores
    relevance_score: float         # Final combined score (0-1)
    algorithmic_score: float       # Pre-LLM algorithmic score
    llm_adjustment: float          # LLM adjustment applied

    # Detailed breakdown
    score_breakdown: ScoreBreakdown

    # Knockout status
    knockout_triggered: bool
    knockout_reason: str | None

    # Metadata
    scoring_method: str            # "hybrid" or "llm_only"
```

### ScoreBreakdown

```python
class ScoreBreakdown(TypedDict):
    exact_skill_match: float       # 0-1
    semantic_skill_match: float    # Same as exact (no embeddings)
    document_similarity: float     # Same as exact (no embeddings)
    experience_years_fit: float    # 0-1
    education_match: float         # 0-1
    recency_score: float           # 0-1
```

## Design Philosophy

### Why Hybrid?

| Approach | Pros | Cons |
|----------|------|------|
| **Pure Algorithmic** | Fast, reproducible, free | Misses transferable skills, context |
| **Pure LLM** | Nuanced, contextual | Expensive, non-deterministic, slow |
| **Hybrid** | Best of both | Slightly more complex |

### Why Exact Matching Only?

Real ATS systems use keyword matching. Using semantic/embedding matching would:
- Create false positives (skills that "seem similar" but aren't the same)
- Give unrealistic scores
- Not reflect how candidates are actually filtered

### Why Knockout Rules?

Enterprise ATS systems automatically reject candidates missing required skills. The knockout rule:
- Mimics real ATS behavior
- Prevents wasted LLM calls on unqualified candidates
- Provides clear, actionable feedback ("You're missing: X, Y, Z")

## Usage

### Basic Scoring

```python
from cv_warlock.scoring import HybridScorer
from cv_warlock.llm import get_llm_provider

provider = get_llm_provider("anthropic")
scorer = HybridScorer(provider)

result = scorer.score(cv_data, job_requirements)
print(f"Score: {result['relevance_score']:.0%}")
print(f"Gaps: {result['gaps']}")
```

### Scoring with Tailoring Plan

```python
# Get both score and plan in single LLM call
match_result, tailoring_plan = scorer.score_with_plan(cv_data, job_requirements)

print(f"Score: {match_result['relevance_score']:.0%}")
print(f"Summary focus: {tailoring_plan.summary_focus}")
```

### Algorithmic-Only Scoring

```python
from cv_warlock.scoring import AlgorithmicScorer

scorer = AlgorithmicScorer()
scores = scorer.compute(cv_data, job_requirements)

print(f"Algorithmic: {scores.total:.0%}")
print(f"Knockout: {scores.knockout_triggered}")
```

## Configuration

### Score Weights

Weights are defined in `AlgorithmicScorer.WEIGHTS`:

```python
WEIGHTS = {
    "exact_skill_match": 0.35,
    "experience_years_fit": 0.25,
    "education_match": 0.15,
    "recency_score": 0.25,
}
```

These can be customized per job type in future versions.

### Recency Decay

The recency decay rate is set to 0.85 (15% decay per year):

```python
recency_weight = max(0.1, pow(0.85, years_ago))
```

This gives a half-life of approximately 4.6 years.

## Module Structure

```
src/cv_warlock/scoring/
├── __init__.py        # Public exports
├── models.py          # Pydantic models (ScoreBreakdown, AlgorithmicScores, etc.)
├── algorithmic.py     # AlgorithmicScorer class
└── hybrid.py          # HybridScorer class
```
