# Contributing to CV Warlock

Thank you for your interest in contributing to CV Warlock! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## Getting Started

### Prerequisites

- **Python 3.11+** (minimum required)
- **Python 3.13+** (recommended for best performance)
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager

### Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cv-warlock.git
   cd cv-warlock
   ```

3. **Install dependencies with dev extras**:
   ```bash
   uv sync --all-extras
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your API keys
   ```

5. **Verify your setup**:
   ```bash
   uv run pytest
   uv run ruff check .
   uv run mypy src/cv_warlock
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names:
- `feature/add-pdf-export`
- `fix/api-key-validation`
- `docs/update-readme`
- `refactor/llm-provider-interface`

### Making Changes

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the code style guidelines below

3. Run the test suite:
   ```bash
   uv run pytest
   ```

4. Run linting and type checking:
   ```bash
   uv run ruff check .
   uv run mypy src/cv_warlock
   ```

5. Commit your changes with a clear message:
   ```bash
   git commit -m "feat: add PDF export functionality"
   ```

### Commit Message Format

Use conventional commit prefixes:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

## Code Style

### Python Guidelines

- **Python 3.11+** with type hints on all functions
- **Line length**: 100 characters maximum
- **Formatting**: Enforced by ruff
- **Type checking**: Strict mypy mode enabled
- **Data models**: Use Pydantic v2 for all data models

### Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check . --fix

# Format code
uv run ruff format .
```

### Type Checking

We use [mypy](https://mypy-lang.org/) with strict mode:

```bash
uv run mypy src/cv_warlock
```

All new code must pass type checking without errors.

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_file.py

# Run with coverage
uv run pytest --cov=src/cv_warlock
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names
- Include both positive and negative test cases

### Test Output Naming Convention

When creating test output files, use this format:
```
YYYY-MM-DD_HH-MM_modelname.md
```

Example: `2026-01-17_14-30_claude-sonnet-4-5.md`

## Pull Request Process

1. **Update documentation** if your changes affect usage or APIs

2. **Ensure all checks pass**:
   - Tests (`uv run pytest`)
   - Linting (`uv run ruff check .`)
   - Type checking (`uv run mypy src/cv_warlock`)

3. **Create a pull request** with:
   - Clear title describing the change
   - Description of what changed and why
   - Link to any related issues

4. **Address review feedback** promptly

### PR Checklist

- [ ] Code follows the project's style guidelines
- [ ] Tests added for new functionality
- [ ] All existing tests pass
- [ ] Documentation updated if needed
- [ ] Type hints added for new functions
- [ ] No new linting or mypy errors

## Reporting Issues

### Bug Reports

When reporting bugs, include:
- Python version (`python --version`)
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces

### Feature Requests

For feature requests:
- Describe the use case
- Explain why existing features don't solve it
- Suggest a possible implementation (optional)

## Project Architecture

Understanding the codebase structure helps when contributing:

```
cv-warlock/
├── src/cv_warlock/          # Main package
│   ├── models/              # Pydantic data models
│   ├── llm/                 # LLM provider abstraction
│   ├── extractors/          # CV and job extraction
│   ├── processors/          # Matching and tailoring logic
│   ├── graph/               # LangGraph workflow
│   ├── prompts/             # Prompt templates
│   └── output/              # Output formatters
├── app/                     # Streamlit web UI
├── examples/                # Sample CV and job files
└── tests/                   # Test suite
```

### Key Components

- **LangGraph Workflow** (`graph/workflow.py`): Core processing pipeline
- **LLM Providers** (`llm/`): Abstraction for OpenAI, Anthropic, Google
- **Models** (`models/`): Pydantic schemas for CV, job specs, state
- **Processors** (`processors/`): Business logic for matching and tailoring

## Questions?

If you have questions about contributing, feel free to:
- Open a GitHub issue with the `question` label
- Contact me personally on LinkedIn: <https://www.linkedin.com/in/wszabopeter/> 
- Check existing issues and discussions

Thank you for contributing to CV Warlock!

*Peter*
