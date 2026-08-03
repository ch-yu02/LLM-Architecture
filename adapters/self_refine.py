from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark_core.interfaces import ModelBackend
from benchmark_core.schema import Generation, Problem
from executors import ProgramExecutionError, RestrictedPythonExecutor

from .base import MethodAdapter
from .pal import extract_python_code, format_final_answer


_TEXT_DATASETS = {
    "math-perturb",
    "harp",
    "harp-small",
    "u-math-text-only",
}


def _answer_requirement(problem: Problem) -> str:
    instruction = problem.answer_instruction.strip()
    return instruction or "Give a clear, unambiguous final answer."


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

    @staticmethod
    def _split_text_refinement(response: str) -> tuple[str, str, str]:
        def tagged(name: str) -> str:
            opening = f"<{name}>"
            closing = f"</{name}>"
            if opening not in response or closing not in response:
                raise ValueError(
                    f"Self-Refine response did not include {opening}...{closing}"
                )
            return response.split(opening, 1)[1].split(closing, 1)[0].strip()

        status = tagged("status").lower()
        if status not in {"correct", "incorrect"}:
            raise ValueError(
                "Self-Refine <status> must be exactly correct or incorrect"
            )
        feedback = tagged("feedback")
        revised = tagged("revised_solution")
        if not feedback:
            raise ValueError("Self-Refine feedback was empty")
        if not revised:
            raise ValueError("Self-Refine revised solution was empty")
        return status, feedback, revised

    @staticmethod
    def _failed_generation(
        generation: Generation,
        *,
        prompt_profile: str,
        trace: list[dict[str, Any]],
        stage: str,
        error: Exception,
        generated_program: str | None = None,
    ) -> Generation:
        metadata = {
            **generation.metadata,
            "method": "self_refine",
            "prompt_profile": prompt_profile,
            "refinement_trace": trace,
            "method_failure": {
                "stage": stage,
                "exception_type": type(error).__name__,
                "error": str(error),
                "raw_output": generation.text,
            },
        }
        if generated_program is not None:
            metadata["generated_program"] = generated_program
            metadata["execution"] = {
                "backend": "restricted_python",
                "status": "error",
                "error": str(error),
            }
        return replace(generation, text="", metadata=metadata)

    def _run_python(
        self,
        problem: Problem,
        backend: ModelBackend,
        *,
        max_refinements: int,
        feedback_overrides: dict[str, Any],
    ) -> Generation:
        prompt_profile = "upstream-gsm-python"
        initial_prompt = (
            f"{self.initial_examples}# Q: {problem.prompt.strip()}\n"
            "# solution using Python:\n"
        )
        current_generation = backend.generate(
            [{"role": "user", "content": initial_prompt}], config={}
        )
        solution = extract_python_code(current_generation.text)
        trace: list[dict[str, Any]] = [
            {"stage": "initial", "solution": solution}
        ]

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
            try:
                feedback, rewritten = self._split_feedback(
                    current_generation.text
                )
            except ValueError as exc:
                return self._failed_generation(
                    current_generation,
                    prompt_profile=prompt_profile,
                    trace=trace,
                    stage="feedback_parse",
                    error=exc,
                )
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

        try:
            executed = self.executor.execute(solution)
        except ProgramExecutionError as exc:
            return self._failed_generation(
                current_generation,
                prompt_profile=prompt_profile,
                trace=trace,
                stage="program_execution",
                error=exc,
                generated_program=solution,
            )
        return replace(
            current_generation,
            text=format_final_answer(executed.value),
            finish_reason="stop",
            metadata={
                **current_generation.metadata,
                "method": "self_refine",
                "prompt_profile": prompt_profile,
                "generated_program": solution,
                "refinement_trace": trace,
                "execution": {
                    "backend": "restricted_python",
                    "protocol": self.executor.protocol,
                    "status": "success",
                    "value_type": executed.value_type,
                },
            },
        )

    def _run_text(
        self,
        problem: Problem,
        backend: ModelBackend,
        *,
        max_refinements: int,
        feedback_overrides: dict[str, Any],
    ) -> Generation:
        prompt_profile = f"{problem.dataset}-text-refinement"
        requirement = _answer_requirement(problem)
        initial_prompt = f"""Solve the following mathematical problem. Give a complete, self-contained
solution, followed by the required final answer.

Problem:
{problem.prompt.strip()}

Final-answer requirement:
{requirement}
""".strip()
        current_generation = backend.generate(
            [{"role": "user", "content": initial_prompt}], config={}
        )
        solution = current_generation.text.strip()
        trace: list[dict[str, Any]] = [
            {"stage": "initial", "solution": solution}
        ]

        for index in range(max_refinements):
            refinement_prompt = f"""Review and improve the candidate solution to the mathematical problem below.
Check the reasoning, calculations, interpretation of the question, and final
answer. Even when it is already correct, return a complete standalone revised
solution rather than only commenting on it.

Problem:
{problem.prompt.strip()}

Candidate solution:
{solution}

Final-answer requirement for the revised solution:
{requirement}

Return exactly these three tagged fields and no text outside them:
<status>STATUS</status>
<feedback>your concise assessment</feedback>
<revised_solution>the complete standalone revised solution</revised_solution>
Replace STATUS with exactly correct or incorrect.
""".strip()
            current_generation = backend.generate(
                [{"role": "user", "content": refinement_prompt}],
                config=feedback_overrides,
            )
            try:
                status, feedback, rewritten = self._split_text_refinement(
                    current_generation.text
                )
            except ValueError as exc:
                return self._failed_generation(
                    current_generation,
                    prompt_profile=prompt_profile,
                    trace=trace,
                    stage="feedback_parse",
                    error=exc,
                )
            trace.append(
                {
                    "stage": "refinement",
                    "iteration": index + 1,
                    "status": status,
                    "feedback": feedback,
                    "solution": rewritten,
                }
            )
            solution = rewritten
            if status == "correct":
                break

        return replace(
            current_generation,
            text=solution,
            metadata={
                **current_generation.metadata,
                "method": "self_refine",
                "prompt_profile": prompt_profile,
                "refinement_trace": trace,
            },
        )

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
        if problem.dataset in _TEXT_DATASETS:
            return self._run_text(
                problem,
                backend,
                max_refinements=max_refinements,
                feedback_overrides=feedback_overrides,
            )
        return self._run_python(
            problem,
            backend,
            max_refinements=max_refinements,
            feedback_overrides=feedback_overrides,
        )
