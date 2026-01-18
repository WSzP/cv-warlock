"""RLM Orchestrator - Core orchestration loop for recursive language models.

Implements the RLM pattern:
1. Root LLM plans and executes code
2. Environment stores context and executes code
3. Sub-LLM calls analyze chunks when needed
"""

import json
import logging
import re
import time
from typing import Any, TypeVar

from pydantic import BaseModel

from cv_warlock.llm.base import LLMProvider
from cv_warlock.rlm.chunking import CVChunker, JobChunker
from cv_warlock.rlm.environment import REPLEnvironment
from cv_warlock.rlm.models import (
    ActionType,
    ModelAction,
    RLMConfig,
    RLMResult,
    SubCallResult,
    TrajectoryStep,
)
from cv_warlock.rlm.prompts import (
    RLM_CONTINUE_PROMPT,
    RLM_SUB_QUERY_PROMPT,
    RLM_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class RLMOrchestrator:
    """Orchestrates the RLM workflow for CV analysis.

    The orchestrator manages:
    - Root LLM that plans and executes code
    - REPL environment holding context
    - Sub-LLM calls for chunk analysis
    """

    # Patterns for parsing model output
    CODE_BLOCK_PATTERN = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
    FINAL_PATTERN = re.compile(r"FINAL\s*\(\s*(.*)\s*\)", re.DOTALL)
    FINAL_VAR_PATTERN = re.compile(r"FINAL_VAR\s*\(\s*(\w+)\s*\)")

    def __init__(
        self,
        root_provider: LLMProvider,
        sub_provider: LLMProvider | None = None,
        config: RLMConfig | None = None,
    ):
        """Initialize the RLM orchestrator.

        Args:
            root_provider: LLM provider for root model (needs strong coding ability).
            sub_provider: LLM provider for sub-calls (can be smaller/faster).
            config: Configuration for orchestration.
        """
        self.root_provider = root_provider
        self.sub_provider = sub_provider or root_provider
        self.config = config or RLMConfig()

        # Statistics tracking
        self.sub_call_count = 0
        self.total_tokens = 0

    def complete(
        self,
        task: str,
        cv_text: str,
        job_text: str,
        output_schema: type[T] | None = None,
    ) -> RLMResult:
        """Run the RLM loop to completion.

        Args:
            task: The analysis task to perform.
            cv_text: Full CV text.
            job_text: Full job specification text.
            output_schema: Optional Pydantic model for structured output.

        Returns:
            RLMResult containing answer, trajectory, and metadata.
        """
        start_time = time.time()
        self.sub_call_count = 0
        self.total_tokens = 0

        trajectory: list[TrajectoryStep] = []

        # Pre-chunk the documents
        cv_chunker = CVChunker()
        job_chunker = JobChunker()
        cv_chunks = cv_chunker.chunk(cv_text)
        job_chunks = job_chunker.chunk(job_text)

        # Initialize environment with context
        env = REPLEnvironment(
            cv_text=cv_text,
            job_text=job_text,
            cv_chunks=cv_chunks,
            job_chunks=job_chunks,
            sandbox_mode=self.config.sandbox_mode,
        )

        # Register helper functions
        self._register_helpers(env)

        # Build initial prompt
        context_summary = env.get_context_summary()
        system_prompt = RLM_SYSTEM_PROMPT.format(
            task=task,
            context_summary=context_summary,
        )

        # Conversation history for the root model
        # Note: Anthropic requires at least one user message after the system message
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Begin the analysis. Write Python code to explore the data and answer the task.",
            },
        ]

        iteration = 0
        final_answer = None

        try:
            while iteration < self.config.max_iterations:
                iteration += 1
                step_start = time.time()

                logger.debug(f"RLM iteration {iteration}")

                # Get response from root model
                model = self.root_provider.get_chat_model()
                response = model.invoke(messages)
                model_output = response.content

                # Parse the model output
                action = self._parse_model_output(model_output)

                step = TrajectoryStep(
                    step_number=iteration,
                    action_type=action.action_type,
                    model_output=model_output if self.config.verbose_trajectory else "",
                    parsed_action=action,
                    execution_result=None,
                    duration_ms=(time.time() - step_start) * 1000,
                )

                if action.action_type == ActionType.FINAL:
                    # Model has provided final answer
                    final_answer = self._process_final_answer(action.content, output_schema, env)
                    step.execution_result = "Final answer provided"
                    trajectory.append(step)
                    break

                elif action.action_type == ActionType.CODE:
                    # Execute code in environment
                    result = env.execute(action.content)
                    step.execution_result = result.output if result.success else result.error

                    # Add result to conversation
                    if result.success:
                        feedback = f"Execution output:\n{result.output}"
                    else:
                        feedback = f"Execution error:\n{result.error}"

                    messages.append({"role": "assistant", "content": model_output})
                    messages.append({"role": "user", "content": feedback})

                elif action.action_type == ActionType.QUERY:
                    # Sub-LLM call
                    if self.sub_call_count >= self.config.max_sub_calls:
                        feedback = (
                            "Maximum sub-calls reached. "
                            "Please compile your findings and provide FINAL(answer)."
                        )
                    else:
                        sub_result = self._execute_sub_call(
                            action.context_var or "",
                            action.question or action.content,
                            env,
                        )
                        step.sub_call_made = True
                        step.sub_call_question = action.question
                        step.sub_call_answer = sub_result.answer
                        step.execution_result = sub_result.answer
                        feedback = f"Sub-query result:\n{sub_result.answer}"

                    messages.append({"role": "assistant", "content": model_output})
                    messages.append({"role": "user", "content": feedback})

                else:
                    # Plain text - model is thinking, prompt to continue
                    state_summary = env.get_context_summary()
                    continue_prompt = RLM_CONTINUE_PROMPT.format(state_summary=state_summary)
                    messages.append({"role": "assistant", "content": model_output})
                    messages.append({"role": "user", "content": continue_prompt})

                trajectory.append(step)

                # Check timeout
                if time.time() - start_time > self.config.timeout_seconds:
                    logger.warning("RLM timeout reached")
                    break

            # If no final answer, try to extract from stored variables
            if final_answer is None:
                final_answer = self._extract_fallback_answer(env, output_schema)

            return RLMResult(
                answer=final_answer,
                trajectory=trajectory,
                sub_call_count=self.sub_call_count,
                total_iterations=iteration,
                total_tokens=self.total_tokens,
                execution_time_seconds=time.time() - start_time,
                success=final_answer is not None,
                intermediate_findings=dict(env.variables),
            )

        except Exception as e:
            logger.exception("RLM orchestration error")
            return RLMResult(
                answer=None,
                trajectory=trajectory,
                sub_call_count=self.sub_call_count,
                total_iterations=iteration,
                total_tokens=self.total_tokens,
                execution_time_seconds=time.time() - start_time,
                success=False,
                error=str(e),
                intermediate_findings=dict(env.variables),
            )

    def _register_helpers(self, env: REPLEnvironment) -> None:
        """Register helper functions in the environment."""

        def rlm_query(text: str, question: str) -> str:
            """Sub-model query function available in REPL."""
            result = self._execute_sub_call(text, question, env)
            return result.answer

        def find_keyword(keyword: str, text: str) -> list[tuple[int, int]]:
            """Find all occurrences of keyword in text."""
            positions = []
            start = 0
            while True:
                pos = text.lower().find(keyword.lower(), start)
                if pos == -1:
                    break
                positions.append((pos, pos + len(keyword)))
                start = pos + 1
            return positions

        def find_sections(text: str) -> dict[str, str]:
            """Parse text into sections based on headers."""
            header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
            headers = list(header_pattern.finditer(text))

            if not headers:
                return {"content": text}

            sections = {}
            for i, match in enumerate(headers):
                name = match.group(2).strip().lower()
                start = match.end()
                end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
                sections[name] = text[start:end].strip()

            return sections

        env.register_function("rlm_query", rlm_query)
        env.register_function("find_keyword", find_keyword)
        env.register_function("find_sections", find_sections)

    def _parse_model_output(self, output: str) -> ModelAction:
        """Parse model output into an action.

        Args:
            output: Raw model output text.

        Returns:
            Parsed ModelAction.
        """
        # Check for FINAL answer first
        final_match = self.FINAL_PATTERN.search(output)
        if final_match:
            return ModelAction(
                action_type=ActionType.FINAL,
                content=final_match.group(1).strip(),
            )

        # Check for FINAL_VAR
        final_var_match = self.FINAL_VAR_PATTERN.search(output)
        if final_var_match:
            return ModelAction(
                action_type=ActionType.FINAL,
                content=final_var_match.group(1),  # Variable name
            )

        # Check for code blocks
        code_match = self.CODE_BLOCK_PATTERN.search(output)
        if code_match:
            code = code_match.group(1).strip()

            # Check if code contains rlm_query call
            if "rlm_query(" in code:
                # Extract the query parameters
                query_match = re.search(
                    r"rlm_query\s*\(\s*(.+?)\s*,\s*[\"'](.+?)[\"']\s*\)",
                    code,
                    re.DOTALL,
                )
                if query_match:
                    return ModelAction(
                        action_type=ActionType.QUERY,
                        content=code,
                        context_var=query_match.group(1).strip(),
                        question=query_match.group(2).strip(),
                    )

            return ModelAction(
                action_type=ActionType.CODE,
                content=code,
            )

        # No specific action detected - treat as thinking/text
        return ModelAction(
            action_type=ActionType.CODE,  # Will prompt for code
            content="# Please provide Python code to continue analysis",
        )

    def _execute_sub_call(
        self,
        context: str,
        question: str,
        env: REPLEnvironment,
    ) -> SubCallResult:
        """Execute a sub-LLM call on a context chunk.

        Args:
            context: Context text or variable name.
            question: Question to ask about the context.
            env: REPL environment for variable resolution.

        Returns:
            SubCallResult with the answer.
        """
        start_time = time.time()
        self.sub_call_count += 1

        # Resolve context if it's a variable name
        if context.isidentifier():
            resolved = env.get_variable(context)
            if resolved is not None:
                context = str(resolved)

        # Truncate context if too long
        max_context = 4000
        if len(context) > max_context:
            context = context[:max_context] + "\n... (truncated)"

        # Build prompt for sub-model
        prompt = RLM_SUB_QUERY_PROMPT.format(
            context=context,
            question=question,
        )

        # Call sub-model
        try:
            model = self.sub_provider.get_chat_model()
            response = model.invoke(prompt)
            answer = response.content
        except Exception as e:
            logger.error(f"Sub-call error: {e}")
            answer = f"Error querying sub-model: {e}"

        return SubCallResult(
            question=question,
            context_preview=context[:200] + "..." if len(context) > 200 else context,
            answer=answer,
            duration_ms=(time.time() - start_time) * 1000,
        )

    def _process_final_answer(
        self,
        content: str,
        output_schema: type[T] | None,
        env: REPLEnvironment,
    ) -> Any:
        """Process the final answer from the model.

        Args:
            content: Final answer content or variable name.
            output_schema: Optional Pydantic model for structured output.
            env: REPL environment for variable resolution.

        Returns:
            Processed answer.
        """
        # Check if content is a variable reference
        if content.isidentifier():
            value = env.get_variable(content)
            if value is not None:
                content = str(value) if not isinstance(value, str) else value

        # Try to parse as JSON if schema provided
        if output_schema:
            try:
                # Try direct JSON parse
                data = json.loads(content)
                return output_schema.model_validate(data)
            except (json.JSONDecodeError, Exception):
                # Fall back to string content
                pass

        return content

    def _extract_fallback_answer(
        self,
        env: REPLEnvironment,
        output_schema: type[T] | None,
    ) -> Any:
        """Extract fallback answer from stored variables.

        Args:
            env: REPL environment with stored variables.
            output_schema: Optional Pydantic model for structured output.

        Returns:
            Best available answer or None.
        """
        # Look for common result variable names
        result_keys = ["result", "answer", "output", "analysis", "findings", "summary"]

        for key in result_keys:
            if key in env.variables:
                value = env.variables[key]
                if output_schema:
                    try:
                        if isinstance(value, dict):
                            return output_schema.model_validate(value)
                        elif isinstance(value, str):
                            data = json.loads(value)
                            return output_schema.model_validate(data)
                    except Exception:
                        pass
                return value

        # Return all variables as fallback
        if env.variables:
            return dict(env.variables)

        return None


def create_rlm_orchestrator(
    root_provider: str = "anthropic",
    root_model: str = "claude-opus-4-5-20251101",
    sub_provider: str | None = None,
    sub_model: str | None = None,
    config: RLMConfig | None = None,
) -> RLMOrchestrator:
    """Factory function to create an RLM orchestrator.

    Args:
        root_provider: Provider for root model.
        root_model: Model ID for root model.
        sub_provider: Provider for sub-calls (defaults to root provider).
        sub_model: Model ID for sub-calls (defaults to a faster model).
        config: Optional RLM configuration.

    Returns:
        Configured RLMOrchestrator.
    """
    from cv_warlock.llm.base import get_llm_provider

    # Default to faster model for sub-calls if not specified
    if sub_model is None:
        if root_provider == "anthropic" or sub_provider == "anthropic":
            sub_model = "claude-haiku-4-5-20251001"
        elif root_provider == "openai" or sub_provider == "openai":
            sub_model = "gpt-5-mini"
        elif root_provider == "google" or sub_provider == "google":
            sub_model = "gemini-3-flash-preview"
        else:
            sub_model = root_model

    root = get_llm_provider(root_provider, root_model)
    sub = get_llm_provider(sub_provider or root_provider, sub_model)

    return RLMOrchestrator(
        root_provider=root,
        sub_provider=sub,
        config=config,
    )
