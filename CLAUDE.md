# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run single test file
uv run pytest tests/test_file.py

# Run linting
uv run ruff check .

# Run type checking
uv run mypy src/cv_warlock

# Start web UI
uv run streamlit run app/app.py

# CLI usage
uv run cv-warlock tailor my_cv.md job.txt -o tailored_cv.md
uv run cv-warlock analyze my_cv.md job.txt

# Test API keys
uv run python scripts/test_api_keys.py
uv run python scripts/test_api_keys.py anthropic  # Test specific provider
```

## Architecture

### LangGraph Workflow
The core processing is a LangGraph `StateGraph` defined in `graph/workflow.py`:

```
validate_inputs → extract_cv → extract_job → analyze_match → create_plan
    → tailor_summary → tailor_experiences → tailor_skills → assemble_cv
```

- **State**: `CVWarlockState` (TypedDict in `models/state.py`) flows through all nodes
- **Nodes**: Created in `graph/nodes.py` via `create_nodes(llm_provider, use_cot)` factory
- **Edges**: Conditional edges handle error paths (skip to END if errors exist)

### LLM Provider Abstraction
- Base class `LLMProvider` in `llm/base.py` with `get_chat_model()` and `get_extraction_model()`
- Implementations: `AnthropicProvider`, `OpenAIProvider`, `GoogleProvider`
- Factory function `get_llm_provider(provider, model, api_key)` for instantiation
- Uses LangChain's `with_structured_output()` for Pydantic model extraction

### Chain-of-Thought (CoT) Mode
When `use_cot=True` (default), each tailoring step follows REASON → GENERATE → CRITIQUE → REFINE:
- `CVTailor` in `processors/tailor.py` has both direct and CoT methods
- CoT outputs stored in `*_reasoning_result` fields for transparency
- `GenerationContext` in `models/reasoning.py` maintains consistency across sections

### Key Models
- `CVData` (`models/cv.py`): Parsed CV structure with experiences, skills, education
- `JobRequirements` (`models/job_spec.py`): Required/preferred skills, responsibilities
- `MatchAnalysis`, `TailoringPlan` (`models/state.py`): Analysis and strategy

### Entry Points
- **CLI**: `main.py` using Typer (`cv-warlock` command)
- **Web UI**: `app/app.py` using Streamlit (components in `app/components/`)

## Configuration

Environment variables loaded from `.env.local`:
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`: Provider API keys
- `CV_WARLOCK_PROVIDER`: Default provider (anthropic/openai/google)
- `CV_WARLOCK_MODEL`: Default model name

Settings loaded via Pydantic Settings in `config.py`.

## Code Style

- Python 3.11+ with type hints
- Line length: 100 characters (ruff config)
- Strict mypy mode enabled
- Uses Pydantic v2 for all data models
