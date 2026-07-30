from __future__ import annotations

import runpy
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark_core.interfaces import ModelBackend
from benchmark_core.schema import Generation, Problem
from executors import RestrictedPythonExecutor

from .base import MethodAdapter


def extract_python_code(text: str) -> str:
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    marker = "# solution in Python:"
    if marker in text:
        text = text.rsplit(marker, 1)[-1]
    return text.strip()


def format_final_answer(value: Any) -> str:
    """Wrap a method-produced answer in the shared textual envelope."""

    text = str(value).strip()
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
        self.validate_source()
        prompt_module = runpy.run_path(
            str(self.source_path / "pal" / "prompt" / "math_prompts.py")
        )
        self.prompt_template = prompt_module["MATH_CHAT_BETA_PROMPT"]
        self.system_message = prompt_module["MATH_CHAT_BETA_SYSTEM_MESSAGE"]

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
        response = backend.generate(
            [
                {"role": "system", "content": self.system_message},
                {
                    "role": "user",
                    "content": self.prompt_template.format(question=problem.prompt),
                },
            ],
            config={},
        )
        code = extract_python_code(response.text)
        executed = self.executor.execute(code)
        return replace(
            response,
            text=format_final_answer(executed.value),
            finish_reason="stop",
            metadata={
                **response.metadata,
                "method": "pal",
                "generated_program": code,
                "execution": {
                    "backend": "restricted_python",
                    "value_type": executed.value_type,
                },
            },
        )
