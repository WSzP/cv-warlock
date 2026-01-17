"""Hybrid ATS-style scoring module for CV-job matching.

This module provides a hybrid scoring system that combines:
- Algorithmic sub-scores (deterministic, reproducible, fast, free)
- Exact string matching for knockout rules (no false positives)
- LLM qualitative assessment (transferable skills, context)

No external embedding APIs required - pure string matching for accuracy.
"""

from cv_warlock.scoring.algorithmic import AlgorithmicScorer
from cv_warlock.scoring.hybrid import HybridScorer
from cv_warlock.scoring.models import (
    AlgorithmicScores,
    HybridMatchResult,
    LLMAssessmentOutput,
    ScoreBreakdown,
)

__all__ = [
    "AlgorithmicScorer",
    "AlgorithmicScores",
    "HybridMatchResult",
    "HybridScorer",
    "LLMAssessmentOutput",
    "ScoreBreakdown",
]
