from __future__ import annotations

import runpy
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark_core.interfaces import ModelBackend
from benchmark_core.schema import Generation, Problem
from executors import ProgramExecutionError, RestrictedPythonExecutor

from .base import MethodAdapter


_PROGRAM_RETURN_CONTRACTS = {
    "math-perturb": (
        "Return only the final mathematical answer from solution(). The return "
        "value may be a Python number, Fraction, SymPy expression, or a plain "
        "string for symbolic answers. Preserve exact symbolic values instead of "
        "converting them to floating-point approximations. A returned string "
        "must contain valid LaTeX answer content, not Python expression syntax. "
        "For multipart answers, return one formatted string rather than a Python "
        "list or tuple. Do not include Answer:, \\boxed, or math mode delimiters "
        "in the returned string."
    ),
    "harp": (
        "Return only the single final mathematical answer from solution(). The "
        "return value may be a Python number, Fraction, SymPy expression, or a "
        "plain string for symbolic answers. Preserve exact symbolic values "
        "instead of converting them to floating-point approximations. A returned "
        "string must contain valid LaTeX answer content, not Python expression "
        "syntax. Preserve units, variable assignments, and textual qualifiers "
        "required by the question. For multipart answers, return one formatted "
        "string rather than a Python list or tuple. Do not include Answer:, "
        "\\boxed, or math mode delimiters in the returned string."
    ),
    "harp-small": (
        "Return only the single final mathematical answer from solution(). The "
        "return value may be a Python number, Fraction, SymPy expression, or a "
        "plain string for symbolic answers. Preserve exact symbolic values "
        "instead of converting them to floating-point approximations. A returned "
        "string must contain valid LaTeX answer content, not Python expression "
        "syntax. Preserve units, variable assignments, and textual qualifiers "
        "required by the question. For multipart answers, return one formatted "
        "string rather than a Python list or tuple. Do not include Answer:, "
        "\\boxed, or math mode delimiters in the returned string."
    ),
    "u-math-text-only": (
        "Return the complete final answer from solution(). Use a Python number "
        "or mathematical object when possible; for symbolic, multipart, or "
        "textual answers, return a plain Python string containing only the final "
        "answer, using valid LaTeX rather than Python expression syntax for its "
        "mathematical parts. Return one formatted string rather than a Python "
        "list or tuple. Do not include Answer:, \\boxed, or math mode delimiters "
        "in the returned string."
    ),
}

_GENERIC_PROGRAM_SYSTEM_MESSAGE = (
    "You solve mathematical problems by writing Python programs. Output only one "
    "Python code block and no surrounding explanation."
)


def _generic_program_prompt(question: str, return_contract: str) -> str:
    return f"""Write a self-contained Python program that solves the problem.
The program must define a zero-argument function named solution(). Do not read
input, access the network, or print the answer. You may use the mathematical
libraries available in the execution environment.

Return-value contract:
{return_contract}

Problem:
{question.strip()}
""".strip()


def extract_python_code(text: str) -> str:
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    marker = "# solution in Python:"
    if marker in text:
        text = text.rsplit(marker, 1)[-1]
    return text.strip()


def format_final_answer(value: Any, *, latex_value: str | None = None) -> str:
    """Wrap a method-produced answer in the shared textual envelope."""

    text = (latex_value if latex_value is not None else str(value)).strip()
    if text.lower().startswith("answer:"):
        text = text.split(":", 1)[1].strip()
    if _is_complete_box(text):
        return f"Answer: {text}"
    return rf"Answer: \boxed{{{text}}}"


def _is_complete_box(text: str) -> bool:
    if not text.startswith(r"\boxed{"):
        return False
    depth = 0
    escaped = False
    for index, character in enumerate(text[len(r"\boxed") :], len(r"\boxed")):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index == len(text) - 1
    return False


class PALAdapter(MethodAdapter):
    name = "pal"
    required_paths = ("pal/core/interface.py", "pal/prompt/math_prompts.py")

    def __init__(
        self,
        source_path: Path | str,
        *,
        executor: RestrictedPythonExecutor | None = None,
    ) -> None:
        super().__init__(source_path)
        self.executor = executor or RestrictedPythonExecutor()
        self.executor.validate_environment()
        self.validate_source()
        prompt_module = runpy.run_path(
            str(self.source_path / "pal" / "prompt" / "math_prompts.py")
        )
        self.prompt_template = prompt_module["MATH_CHAT_BETA_PROMPT"]
        self.system_message = prompt_module["MATH_CHAT_BETA_SYSTEM_MESSAGE"]

    def _messages_for(self, problem: Problem) -> tuple[list[dict[str, str]], str]:
        return_contract = _PROGRAM_RETURN_CONTRACTS.get(problem.dataset)
        if return_contract is None:
            return (
                [
                    {"role": "system", "content": self.system_message},
                    {
                        "role": "user",
                        "content": self.prompt_template.format(
                            question=problem.prompt
                        ),
                    },
                ],
                "upstream-gsm-python",
            )
        return (
            [
                {
                    "role": "system",
                    "content": _GENERIC_PROGRAM_SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": _generic_program_prompt(
                        problem.prompt, return_contract
                    ),
                },
            ],
            f"{problem.dataset}-program-return",
        )

    def run(
        self,
        problem: Problem,
        backend: ModelBackend,
        *,
        config: dict[str, Any],
    ) -> Generation:
        if config:
            raise ValueError(
                f"PAL has no method-specific parameters: {sorted(config)}"
            )
        messages, prompt_profile = self._messages_for(problem)
        response = backend.generate(messages, config={})
        code = extract_python_code(response.text)
        try:
            executed = self.executor.execute(code)
        except ProgramExecutionError as exc:
            # An invalid generated program is a method outcome, not an
            # infrastructure failure. Submit no answer so the dataset scorer
            # records this sample as incorrect while retaining diagnostics.
            return replace(
                response,
                text="",
                metadata={
                    **response.metadata,
                    "method": "pal",
                    "prompt_profile": prompt_profile,
                    "generated_program": code,
                    "execution": {
                        "backend": "restricted_python",
                        "protocol": self.executor.protocol,
                        "status": "error",
                        "error": str(exc),
                    },
                },
            )
        return replace(
            response,
            text=format_final_answer(
                executed.value,
                latex_value=executed.latex_value,
            ),
            finish_reason="stop",
            metadata={
                **response.metadata,
                "method": "pal",
                "prompt_profile": prompt_profile,
                "generated_program": code,
                "execution": {
                    "backend": "restricted_python",
                    "protocol": self.executor.protocol,
                    "status": "success",
                    "value_type": executed.value_type,
                    "answer_rendering": (
                        "latex" if executed.latex_value is not None else "plain"
                    ),
                },
            },
        )
