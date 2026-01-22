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
from typing import Any, Literal, TypeVar, cast

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
    RLM_EXTRACTION_DIRECT_PROMPT,
    RLM_EXTRACTION_WITH_CONTEXT_PROMPT,
    RLM_SUB_QUERY_PROMPT,
    RLM_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Module-level helper functions (registered in REPL environment)
# =============================================================================


def find_keyword(keyword: str, text: str) -> list[tuple[int, int]]:
    """Find all occurrences of keyword in text (case-insensitive)."""
    positions = []
    start = 0
    keyword_lower = keyword.lower()
    text_lower = text.lower()
    while True:
        pos = text_lower.find(keyword_lower, start)
        if pos == -1:
            break
        positions.append((pos, pos + len(keyword)))
        start = pos + 1
    return positions


def find_sections(text: str) -> dict[str, str]:
    """Parse text into sections based on markdown headers."""
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


def _try_parse_json(content: str, schema: type[T] | None) -> T | None:
    """Try to parse content as JSON and validate against schema."""
    if not schema:
        return None
    try:
        data = json.loads(content)
        return schema.model_validate(data)
    except (json.JSONDecodeError, Exception):
        pass
    # Try to extract JSON from within the content
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return schema.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass
    return None


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
                # Ensure model_output is a string (LangChain can return list for multimodal)
                raw_content = response.content
                model_output = raw_content if isinstance(raw_content, str) else str(raw_content)

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

                    # Check if FINAL was also present in the same response
                    # (model may create variable and call FINAL in same response)
                    text_without_code = model_output
                    for match in reversed(list(self.CODE_BLOCK_PATTERN.finditer(model_output))):
                        text_without_code = (
                            text_without_code[: match.start()] + text_without_code[match.end() :]
                        )
                    final_in_output = self.FINAL_PATTERN.search(text_without_code)
                    final_var_in_output = (
                        self.FINAL_VAR_PATTERN.search(text_without_code)
                        if not final_in_output
                        else None
                    )

                    if final_in_output or final_var_in_output:
                        # FINAL was present - process it now that code has been executed
                        if final_in_output:
                            final_content = final_in_output.group(1).strip()
                        else:
                            assert final_var_in_output is not None  # for type narrowing
                            final_content = final_var_in_output.group(1)
                        final_answer = self._process_final_answer(final_content, output_schema, env)
                        step.execution_result = (
                            f"Code executed, then FINAL: {step.execution_result}"
                        )
                        trajectory.append(step)
                        break

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
                    # Plain text - model is thinking, prompt to continue with urgency
                    state_summary = env.get_context_summary()
                    remaining = self.config.max_iterations - iteration
                    urgency = (
                        f"\n\n⚠️ ITERATION {iteration}/{self.config.max_iterations} - "
                        f"Only {remaining} iterations left! Provide FINAL(answer) now."
                        if remaining <= 3
                        else ""
                    )
                    continue_prompt = (
                        RLM_CONTINUE_PROMPT.format(state_summary=state_summary) + urgency
                    )
                    messages.append({"role": "assistant", "content": model_output})
                    messages.append({"role": "user", "content": continue_prompt})

                trajectory.append(step)

                # Check timeout
                if time.time() - start_time > self.config.timeout_seconds:
                    elapsed = time.time() - start_time
                    timeout_msg = (
                        f"RLM timeout after {elapsed:.1f}s "
                        f"(limit: {self.config.timeout_seconds}s, "
                        f"iterations: {iteration}/{self.config.max_iterations})"
                    )
                    logger.warning(timeout_msg)
                    # Store timeout info for caller to detect
                    trajectory.append(
                        TrajectoryStep(
                            step_number=iteration + 1,
                            action_type=ActionType.FINAL,
                            model_output="TIMEOUT",
                            parsed_action=None,
                            execution_result=timeout_msg,
                            duration_ms=0,
                        )
                    )
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
            return self._execute_sub_call(text, question, env).answer

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
        # Step 1: Extract all code blocks and create text without code blocks
        # This prevents FINAL() inside code blocks from being matched
        code_blocks = list(self.CODE_BLOCK_PATTERN.finditer(output))
        text_without_code = output
        for match in reversed(code_blocks):  # Reverse to preserve indices
            text_without_code = (
                text_without_code[: match.start()] + text_without_code[match.end() :]
            )

        # Step 2: Check for FINAL answer ONLY in text outside code blocks
        final_match = self.FINAL_PATTERN.search(text_without_code)
        final_var_match = (
            self.FINAL_VAR_PATTERN.search(text_without_code) if not final_match else None
        )

        # Step 3: PRIORITY - If we have code blocks, execute them FIRST
        # The model often builds variables in code and then calls FINAL(var)
        # We need to execute the code before we can resolve the FINAL
        if code_blocks:
            # Combine ALL code blocks - model may create vars in one and use in another
            all_code = "\n\n".join(match.group(1).strip() for match in code_blocks)

            # Check if any code contains rlm_query call
            if "rlm_query(" in all_code:
                # Execute first code block (may need sub-call)
                code = code_blocks[0].group(1).strip()
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

            # Return code for execution - FINAL will be processed in next iteration
            # after the variables have been created
            return ModelAction(
                action_type=ActionType.CODE,
                content=all_code,
            )

        # Step 4: No code blocks - now check for FINAL
        if final_match:
            return ModelAction(
                action_type=ActionType.FINAL,
                content=final_match.group(1).strip(),
            )

        if final_var_match:
            return ModelAction(
                action_type=ActionType.FINAL,
                content=final_var_match.group(1),  # Variable name
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
            raw_answer = response.content
            answer = raw_answer if isinstance(raw_answer, str) else str(raw_answer)
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
        """Process the final answer from the model."""
        # Resolve variable reference if applicable
        if content.isidentifier():
            value = env.get_variable(content)
            if value is not None:
                content = str(value) if not isinstance(value, str) else value

        # Try JSON parsing first
        parsed = _try_parse_json(content, output_schema)
        if parsed is not None:
            return parsed

        # Fall back to LLM extraction if schema provided
        if output_schema:
            return self._extract_with_llm(content, output_schema, env)

        return content

    def _extract_with_llm(
        self,
        content: str,
        output_schema: type[T],
        env: REPLEnvironment,
    ) -> T | str:
        """Use LLM to extract structured data from text content."""
        schema_name = output_schema.__name__
        schema_fields = ", ".join(output_schema.model_fields.keys())

        # Build extraction prompt based on content availability
        if content and len(content.strip()) > 100:
            context = self._build_context_string(env)
            extraction_prompt = RLM_EXTRACTION_WITH_CONTEXT_PROMPT.format(
                content=content,
                context=context,
                schema_name=schema_name,
                schema_fields=schema_fields,
            )
        else:
            logger.info(f"No analysis content, using direct extraction for {schema_name}")
            extraction_prompt = RLM_EXTRACTION_DIRECT_PROMPT.format(
                cv_text=getattr(env, "cv_text", ""),
                job_text=getattr(env, "job_text", ""),
                schema_name=schema_name,
                schema_fields=schema_fields,
            )

        try:
            model = self.sub_provider.get_extraction_model()
            result = model.with_structured_output(output_schema).invoke(extraction_prompt)
            logger.info(f"LLM extraction successful for {schema_name}")
            return cast(T, result)
        except Exception as e:
            logger.warning(f"LLM extraction failed for {schema_name}: {e}")
            return content

    def _build_context_string(self, env: REPLEnvironment) -> str:
        """Build context string from environment variables."""
        if not env.variables:
            return "No additional context"
        parts = []
        for key, value in env.variables.items():
            if isinstance(value, (dict, list)):
                parts.append(f"{key}: {json.dumps(value, indent=2)}")
            else:
                parts.append(f"{key}: {value}")
        return "\n".join(parts)

    def _extract_fallback_answer(
        self,
        env: REPLEnvironment,
        output_schema: type[T] | None,
    ) -> Any:
        """Extract fallback answer from stored variables."""
        result_keys = ["result", "answer", "output", "analysis", "findings", "summary"]

        for key in result_keys:
            if key not in env.variables:
                continue
            value = env.variables[key]

            # Try direct validation for dicts
            if output_schema and isinstance(value, dict):
                try:
                    return output_schema.model_validate(value)
                except Exception:
                    pass

            # Try JSON parsing for strings
            content = str(value) if not isinstance(value, str) else value
            parsed = _try_parse_json(content, output_schema)
            if parsed is not None:
                return parsed

            # Try LLM extraction
            if output_schema and value:
                result = self._extract_with_llm(content, output_schema, env)
                if isinstance(result, output_schema):
                    return result
            return value

        # Try LLM extraction on combined context
        if env.variables and output_schema:
            combined = self._build_context_string(env)
            result = self._extract_with_llm(combined, output_schema, env)
            if isinstance(result, output_schema):
                return result
            return dict(env.variables)

        return dict(env.variables) if env.variables else None


ProviderType = Literal["openai", "anthropic", "google"]


def create_rlm_orchestrator(
    root_provider: ProviderType = "anthropic",
    root_model: str = "claude-opus-4-5-20251101",
    sub_provider: ProviderType | None = None,
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
    effective_sub_provider: ProviderType = sub_provider if sub_provider else root_provider
    sub = get_llm_provider(effective_sub_provider, sub_model)

    return RLMOrchestrator(
        root_provider=root,
        sub_provider=sub,
        config=config,
    )
