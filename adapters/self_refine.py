from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark_core.interfaces import ModelBackend
from benchmark_core.schema import Generation, Problem
from executors import RestrictedPythonExecutor

from .base import MethodAdapter
from .pal import extract_python_code


class SelfRefineAdapter(MethodAdapter):
    name = "self_refine"
    required_paths = (
        "src/gsm/run.py",
        "src/gsm/feedback.py",
        "src/gsm/task_init.py",
        "data/prompt/gsm/init.txt",
        "data/prompt/gsm/feedback.txt",
    )

    def __init__(
        self,
        source_path: Path | str,
        *,
        executor: RestrictedPythonExecutor | None = None,
    ) -> None:
        super().__init__(source_path)
        self.executor = executor or RestrictedPythonExecutor()
        self.validate_source()
        prompt_root = self.source_path / "data" / "prompt" / "gsm"
        self.initial_examples = (prompt_root / "init.txt").read_text(encoding="utf-8")
        self.feedback_examples = (prompt_root / "feedback.txt").read_text(
            encoding="utf-8"
        )

    @staticmethod
    def _feedback_says_correct(feedback: str) -> bool:
        lowered = feedback.lower()
        return "it is correct" in lowered or "there is no error" in lowered

    @staticmethod
    def _split_feedback(response: str) -> tuple[str, str]:
        content = response.split("### END", 1)[0].strip()
        marker = "def solution():"
        if marker not in content:
            raise ValueError("Self-Refine feedback did not include rewritten solution()")
        feedback, body = content.rsplit(marker, 1)
        return feedback.strip(), f"{marker}{body.rstrip()}"

    def run(
        self,
        problem: Problem,
        backend: ModelBackend,
        *,
        config: dict[str, Any],
    ) -> Generation:
        max_refinements = int(config.get("max_refinements", 4))
        feedback_overrides = config.get("feedback_inference_overrides", {})
        unknown = set(config) - {
            "max_refinements",
            "feedback_inference_overrides",
        }
        if unknown:
            raise ValueError(f"Unknown Self-Refine config: {sorted(unknown)}")
        if max_refinements < 1:
            raise ValueError("max_refinements must be at least 1")

        initial_prompt = (
            f"{self.initial_examples}# Q: {problem.prompt.strip()}\n"
            "# solution using Python:\n"
        )
        current_generation = backend.generate(
            [{"role": "user", "content": initial_prompt}], config={}
        )
        solution = extract_python_code(current_generation.text)
        trace: list[dict[str, Any]] = [{"stage": "initial", "solution": solution}]

        feedback_prompt = self.feedback_examples
        for index in range(max_refinements):
            instruction = (
                "# There is an error in the code above because of lack of "
                "understanding of the question. What is the error? To find the "
                "error, go through semantically complete blocks of the code, and "
                "check if everything looks good."
            )
            query = f"{feedback_prompt}{solution}\n\n{instruction}\n"
            current_generation = backend.generate(
                [{"role": "user", "content": query}],
                config=feedback_overrides,
            )
            feedback, rewritten = self._split_feedback(current_generation.text)
            trace.append(
                {
                    "stage": "refinement",
                    "iteration": index + 1,
                    "feedback": feedback,
                    "solution": rewritten,
                }
            )
            solution = rewritten
            if self._feedback_says_correct(feedback):
                break
            feedback_prompt = (
                f"{feedback_prompt}{query[len(feedback_prompt):]}\n\n"
                f"{feedback}\n\n{solution}\n\n### END ###\n\n"
            )

        executed = self.executor.execute(solution)
        return replace(
            current_generation,
            text=rf"\boxed{{{executed.value}}}",
            metadata={
                **current_generation.metadata,
                "method": "self_refine",
                "generated_program": solution,
                "refinement_trace": trace,
                "execution": {
                    "backend": "restricted_python",
                    "value_type": executed.value_type,
                },
            },
        )
