"""Recursive Language Model (RLM) integration for CV Warlock.

This module enables handling of arbitrarily long CVs and job specs by using
recursive language model techniques. The model can explore context via code
execution and spawn sub-calls to analyze specific chunks.

Key components:
- RLMOrchestrator: Core orchestration loop
- REPLEnvironment: Sandboxed execution context
- Chunking utilities: Smart CV/job spec parsing
- RLMConfig: Configuration for RLM behavior
"""

from cv_warlock.rlm.chunking import CVChunker, JobChunker
from cv_warlock.rlm.environment import REPLEnvironment
from cv_warlock.rlm.models import (
    CVChunks,
    ExecutionResult,
    JobChunks,
    RLMConfig,
    RLMResult,
    TrajectoryStep,
)
from cv_warlock.rlm.orchestrator import RLMOrchestrator, create_rlm_orchestrator

__all__ = [
    # Orchestrator
    "RLMOrchestrator",
    "create_rlm_orchestrator",
    # Environment
    "REPLEnvironment",
    # Chunking
    "CVChunker",
    "JobChunker",
    # Models
    "RLMResult",
    "RLMConfig",
    "ExecutionResult",
    "TrajectoryStep",
    "CVChunks",
    "JobChunks",
]
