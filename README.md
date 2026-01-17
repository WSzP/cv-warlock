# CV Warlock

AI-powered CV tailoring for job applications. Takes your generic CV and a job posting, then generates a tailored CV that emphasizes your most relevant experience and skills.

## Features

- **Smart CV Tailoring**: Analyzes job requirements and rewrites your CV to highlight relevant experience
- **Chain-of-Thought Reasoning**: Optional CoT mode for higher quality generation (REASON → GENERATE → CRITIQUE → REFINE)
- **Match Analysis**: Shows how well your CV matches the job, including gaps and transferable skills
- **Assume All Tech Skills**: Checkbox option (enabled by default) that assumes you have all technical skills from the job posting
- **Multi-Provider LLM Support**: Works with OpenAI, Anthropic Claude, and Google Gemini
- **CLI & Web UI**: Use via command line or Streamlit web interface
- **LangGraph Workflow**: Robust multi-step processing with state management
- **Real-Time Progress**: Live timing display during CV generation

## Requirements

- **Python 3.11+** (minimum required)
- **Python 3.13+** (recommended for best performance)
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager (recommended)

## Installation

```bash
# Install uv (if not already installed)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone the repository
git clone https://github.com/yourusername/cv-warlock.git
cd cv-warlock

# Install dependencies with uv
uv sync
```

## Configuration

Create a `.env.local` file in the project root (copy from `.env.example`):

```bash
# At least one API key is required
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Optional: Default provider and model
CV_WARLOCK_PROVIDER=anthropic
CV_WARLOCK_MODEL=claude-sonnet-4-5-20250929
```

### Testing API Keys

You can validate your API keys using the included test script:

```bash
# Test all configured keys
uv run python scripts/test_api_keys.py

# Test a specific provider
uv run python scripts/test_api_keys.py openai
uv run python scripts/test_api_keys.py anthropic
uv run python scripts/test_api_keys.py google
```

The script will:

- Check which API keys are configured in `.env.local`
- Make a test request to each configured provider
- Report success or failure with detailed messages

You can also test API keys directly in the Web UI by clicking the **Test API Key** button in the sidebar (appears when an API key is loaded from `.env.local`).

## Usage

### CLI

```bash
# Tailor your CV for a job
uv run cv-warlock tailor my_cv.md job_posting.txt --output tailored_cv.md

# Analyze CV-job fit without generating a tailored CV
uv run cv-warlock analyze my_cv.md job_posting.txt

# Use a specific provider/model
uv run cv-warlock tailor my_cv.md job.txt --provider anthropic --model claude-opus-4-5-20251101

# Show version
uv run cv-warlock version
```

### Web UI (Streamlit)

```bash
uv run streamlit run app/app.py
```

Then open http://localhost:8501 in your browser.

## How It Works

CV Warlock uses a LangGraph workflow to process your CV in stages:

1. **Extract**: Parse CV and job spec into structured data
2. **Analyze**: Find matches, gaps, and transferable skills
3. **Plan**: Create a tailoring strategy
4. **Tailor**: Rewrite summary, experiences, and skills
5. **Assemble**: Generate the final tailored CV

## Assume All Tech Skills

The **"Assume all requested tech skills"** checkbox (enabled by default) is a key feature designed for software developers and technical professionals.

### Why This Feature Exists

1. **The "thousands of libraries" problem**: In software development, there are thousands of libraries, frameworks, tools, and technologies. No CV can realistically list every technology a developer knows or has used. If your CV mentions "Python" and "FastAPI", you probably also know pip, pytest, Pydantic, and dozens of other Python ecosystem tools—but listing them all would make your CV unwieldy.

2. **LinkedIn skill limitations**: Not everyone has LinkedIn connected or has taken the time to add all their skills. Even if you have, LinkedIn's skill list may not include every technology the job requires.

3. **Implicit knowledge**: Senior developers often have implicit knowledge that isn't explicitly listed. If you've worked with React for 5 years, you almost certainly know JavaScript, npm, webpack/vite, JSX, CSS, and related technologies—even if they're not all listed on your CV.

### What It Does

When enabled (the default), CV Warlock assumes you possess all technical skills mentioned in the job posting's requirements. The system:

- Adds all required and preferred skills from the job spec to your skill list before analysis
- This prevents false "gaps" for technologies you likely know but didn't list
- Results in a tailored CV that confidently includes all relevant technical skills
- Produces higher match scores that better reflect your actual capabilities

### When to Disable It

Turn off this checkbox if:

- You want an honest assessment of skill gaps based only on what's explicitly in your CV
- You're applying for a role requiring technologies you genuinely don't know
- You want to identify areas where you need to upskill before applying

## Chain-of-Thought Reasoning

CV Warlock supports an optional Chain-of-Thought (CoT) reasoning mode that produces higher quality tailored CVs through explicit multi-step reasoning.

### How CoT Works

When enabled (the default), each CV section goes through a 4-phase process:

1. **REASON**: Analyze the job requirements and plan the approach
2. **GENERATE**: Create content based on the reasoning
3. **CRITIQUE**: Evaluate quality against specific criteria
4. **REFINE**: Improve if quality is below threshold (max 2 iterations)

### Benefits

- **Higher Quality Output**: Self-critique catches issues like missing metrics, weak verbs, or keyword gaps
- **Consistent Formatting**: Explicit reasoning ensures all sections follow best practices
- **Better ATS Optimization**: Critique phase verifies keyword inclusion and exact terminology matching
- **Transparency**: All reasoning is stored and can be inspected for debugging

### Trade-offs

- **Slower Generation**: CoT mode makes 3-4x more LLM calls per section
- **Higher Cost**: More API calls means higher token usage

### When to Disable CoT

Turn off CoT reasoning if:

- You need quick iterations and are willing to trade quality for speed
- You're testing or debugging the workflow
- Cost is a concern and you want minimal API usage

Toggle CoT in the Web UI sidebar under "Generation Quality" or pass `use_cot=False` in CLI/code.

## Supported Models

### Anthropic (Recommended)

| Model | Description |
|-------|-------------|
| `claude-sonnet-4-5-20250929` | Best balance of quality and speed (default) |
| `claude-haiku-4-5-20251001` | Fastest, most cost-efficient |
| `claude-opus-4-5-20251101` | Most capable, highest quality |

### OpenAI

| Model | Description |
|-------|-------------|
| `gpt-5.2` | Latest flagship model |
| `gpt-5.2-instant` | Fast version of GPT-5.2 |
| `gpt-5-mini` | Cost-efficient option |
| `gpt-4o` | Previous generation flagship |

### Google

| Model | Description |
|-------|-------------|
| `gemini-3-flash-preview` | Fast and capable (recommended) |
| `gemini-3-pro-preview` | Most capable Gemini model |

## Project Structure

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

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Run type checking
uv run mypy src/cv_warlock
```

## License

Apache License 2.0
