from __future__ import annotations

from typing import Any

from benchmark_core.interfaces import ModelBackend
from benchmark_core.schema import Generation, Problem

from .registry import register_method


_SYSTEM_PROMPT = (
    "You are a mathematical problem solver. Follow the output format requested "
    "by the problem exactly. Do not use external tools."
)


class PromptBaseline:
    name: str
    instruction: str
    model_calls_per_problem = 1
    uses_examples = False
    uses_tools = False

    def run(
        self,
        problem: Problem,
        backend: ModelBackend,
        *,
        config: dict[str, Any],
    ) -> Generation:
        if config:
            raise ValueError(f"{self.name} has no method-specific parameters")
        answer_instruction = (
            f" {problem.answer_instruction}" if problem.answer_instruction else ""
        )
        return backend.generate(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"{problem.prompt}\n\n{self.instruction}"
                        f"{answer_instruction}"
                    ),
                },
            ],
            config={},
        )


class DirectAnswerMethod(PromptBaseline):
    """Zero-shot answer baseline with one model call and no demonstrations."""

    name = "direct"
    instruction = "Solve the problem and answer concisely."


class ZeroShotChainOfThoughtMethod(PromptBaseline):
    """Zero-shot chain-of-thought baseline with the same one-call budget."""

    name = "zero_shot_cot"
    instruction = "Solve the problem by reasoning step by step."


register_method(DirectAnswerMethod())
register_method(ZeroShotChainOfThoughtMethod())
