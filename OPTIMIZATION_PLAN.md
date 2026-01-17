# CV Warlock Performance Optimization Plan

## Status: IMPLEMENTED

The following optimizations have been implemented to achieve 3-4x performance improvement.

## Problem Statement (Before Optimization)

Performance with Haiku 4.5:
- **Time**: 520.85 seconds (~8.7 minutes)
- **Input tokens**: 161.8K ($0.16)
- **Output tokens**: 50.4K ($0.25)
- **Total**: ~$0.41 per CV tailoring

Target: **Under 120 seconds and ~60K input tokens** (3-4x improvement)

---

## Root Cause Analysis

### 1. Sequential LLM Calls (Primary Time Sink)
With CoT enabled, the system makes **29-34 sequential LLM calls**:
- Extract CV + Job: 2 calls
- Analyze + Plan: 2 calls
- Summary (REASON→GENERATE→CRITIQUE→REFINE): 3-5 calls
- **Experiences (per experience)**: 3-5 calls × N experiences = 12-20 calls
- Skills: 3-5 calls
- Assembly: 1 call

Each call incurs ~15-20s network latency. Sequential execution = multiplicative delay.

### 2. Redundant Context Passing (Token Waste)
The `reasoning.model_dump_json(indent=2)` pattern sends full reasoning JSON:
- In GENERATE step (~800 tokens)
- Again in REFINE step (~800 tokens)
- Same context = 2x tokens per section

### 3. Verbose Prompts (Overhead)
Each prompt template is 300-500 tokens. With 30 calls × 400 avg = **12K tokens just in prompt overhead**.

### 4. Over-Aggressive Refinement
Refinement triggers if quality < GOOD. Most outputs are GOOD on first try, yet refinement still runs additional critique calls.

---

## Optimization Strategies

### TIER 1: High Impact (Implement First)

#### 1.1 Parallelize Experience Tailoring (**50-60% time reduction**)

**Problem**: Experiences processed sequentially despite being independent.

**Solution**: Use `asyncio` for concurrent processing.

```python
# Before: Sequential (N × 15s)
for exp in cv_data.experiences:
    result = self.tailor_experience_with_cot(exp, ...)

# After: Parallel (max 15s for all)
async def tailor_all_experiences_async(...):
    tasks = [
        self.tailor_experience_with_cot_async(exp, ...)
        for exp in cv_data.experiences
    ]
    return await asyncio.gather(*tasks)
```

**Estimated savings**: 4 experiences × 15s each = 60s → 15s total = **45 seconds saved**

#### 1.2 Compress Reasoning Context (**30-40% token reduction**)

**Problem**: Full Pydantic JSON passed multiple times.

**Solution**: Extract only needed fields for each step.

```python
# Before: 800 tokens
"reasoning_json": reasoning.model_dump_json(indent=2)

# After: ~150 tokens
"reasoning_summary": f"""Hook: {reasoning.hook_strategy}
Keywords: {', '.join(reasoning.key_keywords_to_include)}
Metric: {reasoning.strongest_metric}
Differentiator: {reasoning.differentiator}"""
```

**Estimated savings**: 30 calls × 600 tokens saved = **18K tokens saved**

#### 1.3 Single-Pass Mode (New Option) (**40% fewer calls**)

**Problem**: CoT always runs 4 steps even when 2 would suffice.

**Solution**: Add `quality_mode` parameter: "fast", "balanced", "thorough"

| Mode | Steps | Calls per Section |
|------|-------|-------------------|
| fast | GENERATE only | 1 |
| balanced | REASON→GENERATE | 2 |
| thorough | REASON→GENERATE→CRITIQUE→REFINE | 3-5 |

```python
class CVTailor:
    def __init__(self, llm_provider, quality_mode="balanced"):
        self.quality_mode = quality_mode  # "fast", "balanced", "thorough"
```

**Estimated savings**:
- Fast mode: 30 calls → 9 calls = **70% reduction**
- Balanced mode: 30 calls → 18 calls = **40% reduction**

#### 1.4 Smarter Refinement Threshold

**Problem**: Refines if quality < GOOD, but most outputs are already GOOD.

**Solution**: Only refine if quality = POOR or NEEDS_IMPROVEMENT.

```python
# Before
while critique.should_refine and refinement_count < MAX:
    ...

# After
while (critique.quality_level == QualityLevel.POOR
       and refinement_count < MAX):
    ...
```

**Estimated savings**: Skip ~50% of refinement cycles = **3-6 fewer calls**

---

### TIER 2: Medium Impact

#### 2.1 Merge Analyze + Plan into Single Call

**Problem**: Two separate LLM calls for related analysis.

**Solution**: Combined prompt that outputs both MatchAnalysis and TailoringPlan.

```python
# Before: 2 calls
match_analysis = analyzer.analyze_match(cv_data, job_req)
tailoring_plan = analyzer.create_tailoring_plan(cv_data, job_req, match_analysis)

# After: 1 call with combined schema
analysis_and_plan = analyzer.analyze_and_plan(cv_data, job_req)
```

**Estimated savings**: 1 call × 15s = **15 seconds saved**

#### 2.2 Condense Prompt Templates (**20% prompt token reduction**)

**Problem**: Prompts include verbose examples and explanations.

**Solution**: Create concise versions for production use.

```python
# Before (500 tokens):
SUMMARY_REASONING_PROMPT = """You are an expert CV strategist...
[8 detailed reasoning steps with examples]
[quality assessment criteria]
..."""

# After (200 tokens):
SUMMARY_REASONING_PROMPT_CONCISE = """Analyze CV-job fit for summary:
Original: {original_summary}
Target: {job_title} at {company}
Required: {key_requirements}

Output: title_positioning, keywords(5), best_metric, differentiator, hook_strategy"""
```

**Estimated savings**: 30 calls × 300 tokens = **9K tokens saved**

#### 2.3 Lazy Context Accumulation

**Problem**: `GenerationContext` grows with each experience, increasing payload.

**Solution**: Only pass essential accumulated context.

```python
# Before: Full context passed
"keywords_already_used": ", ".join(context.primary_keywords_used)
"metrics_already_used": ", ".join(context.metrics_used)

# After: Only top-N items
"top_keywords_used": ", ".join(context.primary_keywords_used[:5])
"top_metrics_used": ", ".join(context.metrics_used[:3])
```

---

### TIER 3: Infrastructure Optimizations

#### 3.1 Enable Prompt Caching (Claude API)

If using Claude, enable `cache_control` for repeated prompt prefixes.

```python
# In llm/anthropic_provider.py
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": SYSTEM_PREFIX, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": user_content}
        ]
    }
]
```

**Estimated savings**: 90% cache hits on repeated prompts = **~50K tokens saved**

#### 3.2 Use Smaller Models for Critique

Critiques are simpler than generation - use Haiku for critique, Sonnet for generation.

```python
def _critique_summary(self, ...):
    model = self.llm_provider.get_extraction_model(
        model_override="claude-3-haiku-20240307"  # Cheaper, faster
    )
```

**Estimated savings**: 30% cost reduction on critique calls

---

## Implementation Priority

| Priority | Change | Time Saved | Tokens Saved | Effort |
|----------|--------|------------|--------------|--------|
| 1 | Parallelize experiences | 45s | - | Medium |
| 2 | Compress reasoning context | - | 18K | Low |
| 3 | Add quality_mode="balanced" | 60s+ | 30K | Medium |
| 4 | Smarter refinement | 30s | 10K | Low |
| 5 | Merge analyze+plan | 15s | 5K | Low |
| 6 | Condense prompts | - | 9K | Low |
| 7 | Prompt caching | - | 50K | Low |

---

## Expected Results After Optimization

### Balanced Mode (Recommended Default)
- **Time**: ~90-120 seconds (from 520s)
- **Input tokens**: ~60K (from 161K)
- **Output tokens**: ~25K (from 50K)
- **Cost**: ~$0.12 (from $0.41)

### Fast Mode (Speed Priority)
- **Time**: ~45-60 seconds
- **Input tokens**: ~30K
- **Output tokens**: ~15K
- **Cost**: ~$0.06

### Thorough Mode (Quality Priority)
- **Time**: ~180-240 seconds (parallelized)
- **Input tokens**: ~80K
- **Output tokens**: ~35K
- **Cost**: ~$0.20

---

## Files to Modify

1. **[src/cv_warlock/processors/tailor.py](src/cv_warlock/processors/tailor.py)** - Add async support, compress context, add quality modes
2. **[src/cv_warlock/graph/nodes.py](src/cv_warlock/graph/nodes.py)** - Use async workflow for parallel processing
3. **[src/cv_warlock/prompts/reasoning.py](src/cv_warlock/prompts/reasoning.py)** - Add concise prompt variants
4. **[src/cv_warlock/processors/matcher.py](src/cv_warlock/processors/matcher.py)** - Merge analyze+plan
5. **[src/cv_warlock/config.py](src/cv_warlock/config.py)** - Add quality_mode setting
6. **[app/app.py](app/app.py)** - Add quality mode selector to UI
