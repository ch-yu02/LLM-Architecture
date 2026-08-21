from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

from benchmark_core.interfaces import MethodOutcomeError, ModelBackend
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

_GSM_FEEDBACK_END = "### END ###"
_GSM_REWRITE_CUE = "Okay! Here is the rewrite:"
_GSM_CORRECT_CUE = (
    "There is no error in the code! It is correct! Here is the rewrite "
    "(for the sake of completeness):"
)


def _answer_requirement(problem: Problem) -> str:
    instruction = problem.answer_instruction.strip()
    return instruction or "Give a clear, unambiguous final answer."


def _gsm_feedback_examples_with_verdicts(source: str) -> str:
    chunks = source.split(_GSM_FEEDBACK_END)
    if chunks[-1].strip():
        raise ValueError("Self-Refine GSM feedback examples have no final marker")

    examples: list[str] = []
    correct_count = 0
    incorrect_count = 0
    for raw_chunk in chunks[:-1]:
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if _GSM_CORRECT_CUE in chunk:
            review = chunk.split(_GSM_CORRECT_CUE, 1)[0].rstrip()
            chunk = (
                f"{review}\n\nThere is no error in the code! It is correct!"
                "\n\n# VERDICT: CORRECT"
            )
            correct_count += 1
        else:
            if chunk.count(_GSM_REWRITE_CUE) != 1:
                raise ValueError(
                    "Self-Refine incorrect GSM feedback example has no "
                    "unique rewrite cue"
                )
            chunk = chunk.replace(
                _GSM_REWRITE_CUE,
                f"# VERDICT: INCORRECT\n\n{_GSM_REWRITE_CUE}",
                1,
            )
            incorrect_count += 1
        examples.append(f"{chunk}\n\n{_GSM_FEEDBACK_END}")

    if (correct_count, incorrect_count) != (1, 3):
        raise ValueError(
            "Self-Refine GSM feedback examples must contain one correct "
            "and three incorrect cases"
        )
    return "\n\n".join(examples) + "\n\n"


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
        self.feedback_examples = _gsm_feedback_examples_with_verdicts(
            (prompt_root / "feedback.txt").read_text(encoding="utf-8")
        )

    @staticmethod
    def _feedback_verdict(response: str) -> str | None:
        verdicts = re.findall(
            r"(?im)^[ \t]*#[ \t]*VERDICT:[ \t]*(CORRECT|INCORRECT)[ \t]*$",
            response,
        )
        return verdicts[-1].lower() if verdicts else None

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
        status_match = re.search(
            r"<status>\s*(correct|incorrect)\s*</status>",
            response,
            flags=re.IGNORECASE,
        )
        if status_match is None:
            raise ValueError(
                "Self-Refine response did not include a complete valid <status>"
            )
        feedback_open = response.find("<feedback>", status_match.end())
        revised_open = response.find("<revised_solution>", status_match.end())
        if feedback_open < 0:
            raise ValueError("Self-Refine response did not include <feedback>")
        if revised_open < 0:
            raise ValueError(
                "Self-Refine response did not include <revised_solution>"
            )
        if feedback_open > revised_open:
            raise ValueError(
                "Self-Refine response placed <feedback> after <revised_solution>"
            )

        status = status_match.group(1).lower()
        if status not in {"correct", "incorrect"}:
            raise ValueError(
                "Self-Refine <status> must be exactly correct or incorrect"
            )
        feedback = response[
            feedback_open + len("<feedback>") : revised_open
        ].strip()
        feedback = re.sub(r"</[^>]+>\s*$", "", feedback).strip()
        revised_start = revised_open + len("<revised_solution>")
        revised_close = response.find("</revised_solution>", revised_start)
        revised = response[
            revised_start : revised_close if revised_close >= 0 else len(response)
        ].strip()
        if not feedback:
            raise ValueError("Self-Refine feedback was empty")
        if not revised:
            raise ValueError("Self-Refine revised solution was empty")
        return status, feedback, revised

    @staticmethod
    def _raise_method_failure(
        generation: Generation,
        *,
        prompt_profile: str,
        trace: list[dict[str, Any]],
        stage: str,
        error: Exception,
        generated_program: str | None = None,
    ) -> NoReturn:
        metadata = {
            **generation.metadata,
            "method": "self_refine",
            "prompt_profile": prompt_profile,
            "refinement_trace": trace,
            "method_failure": {
                "stage": stage,
                "exception_type": type(error).__name__,
                "error": str(error),
            },
        }
        if generated_program is not None:
            metadata["generated_program"] = generated_program
            metadata["execution"] = {
                "backend": "restricted_python",
                "status": "error",
                "error": str(error),
            }
        failed_generation = replace(generation, metadata=metadata)
        raise MethodOutcomeError(
            failed_generation,
            stage=stage,
            reason=f"{type(error).__name__}: {error}",
            details={"exception_type": type(error).__name__},
        )

    def _run_python(
        self,
        problem: Problem,
        backend: ModelBackend,
        *,
        max_refinements: int,
        initial_overrides: dict[str, Any],
        feedback_overrides: dict[str, Any],
    ) -> Generation:
        prompt_profile = "upstream-gsm-python"
        initial_prompt = (
            f"{self.initial_examples}# Q: {problem.prompt.strip()}\n"
            "# solution using Python:\n"
        )
        current_generation = backend.generate(
            [{"role": "user", "content": initial_prompt}],
            config=initial_overrides,
        )
        solution = extract_python_code(current_generation.text)
        trace: list[dict[str, Any]] = [
            {"stage": "initial", "solution": solution}
        ]
        refinement_termination: str | None = None

        feedback_prompt = self.feedback_examples
        for index in range(max_refinements):
            instruction = (
                "# There is an error in the code above because of lack of "
                "understanding of the question. What is the error? To find the "
                "error, go through semantically complete blocks of the code, and "
                "check if everything looks good. If no error is found, output "
                "# VERDICT: CORRECT and do not rewrite the program. If an error "
                "is found, output # VERDICT: INCORRECT and then one complete "
                "rewritten def solution() function. End the entire response "
                "with ### END ###."
            )
            query = f"{feedback_prompt}{solution}\n\n{instruction}\n"
            current_generation = backend.generate(
                [{"role": "user", "content": query}],
                config=feedback_overrides,
            )
            finish_reason = current_generation.finish_reason
            if finish_reason == "length":
                verdict = self._feedback_verdict(current_generation.text)
                if verdict == "incorrect":
                    self._raise_method_failure(
                        current_generation,
                        prompt_profile=prompt_profile,
                        trace=trace,
                        stage="feedback_generation",
                        error=ValueError(
                            "Self-Refine feedback declared the program "
                            "incorrect but reached the token limit before "
                            "the correction completed"
                        ),
                    )
                refinement_termination = (
                    "feedback_declared_correct"
                    if verdict == "correct"
                    else "assumed_correct_after_feedback_token_limit"
                )
                trace.append(
                    {
                        "stage": "refinement",
                        "iteration": index + 1,
                        "feedback": current_generation.text.strip(),
                        "solution": solution,
                        "decision": refinement_termination,
                    }
                )
                break
            if finish_reason != "stop":
                self._raise_method_failure(
                    current_generation,
                    prompt_profile=prompt_profile,
                    trace=trace,
                    stage="feedback_generation",
                    error=ValueError(
                        "Self-Refine feedback generation ended with "
                        f"finish_reason={finish_reason!r}"
                    ),
                )

            verdict = self._feedback_verdict(current_generation.text)
            if verdict == "correct":
                refinement_termination = "feedback_declared_correct"
                trace.append(
                    {
                        "stage": "refinement",
                        "iteration": index + 1,
                        "feedback": current_generation.text.strip(),
                        "solution": solution,
                        "decision": refinement_termination,
                    }
                )
                break
            if verdict is None:
                self._raise_method_failure(
                    current_generation,
                    prompt_profile=prompt_profile,
                    trace=trace,
                    stage="feedback_parse",
                    error=ValueError(
                        "Self-Refine feedback did not include a complete "
                        "# VERDICT line"
                    ),
                )
            try:
                feedback, rewritten = self._split_feedback(
                    current_generation.text
                )
            except ValueError as exc:
                self._raise_method_failure(
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
            feedback_prompt = (
                f"{feedback_prompt}{query[len(feedback_prompt):]}\n\n"
                f"{feedback}\n\n{solution}\n\n### END ###\n\n"
            )

        try:
            executed = self.executor.execute(solution)
        except ProgramExecutionError as exc:
            self._raise_method_failure(
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
                **(
                    {"refinement_termination": refinement_termination}
                    if refinement_termination is not None
                    else {}
                ),
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
        initial_finish_reason = current_generation.finish_reason
        if initial_finish_reason not in {"stop", "length"}:
            self._raise_method_failure(
                current_generation,
                prompt_profile=prompt_profile,
                trace=[],
                stage="initial_generation",
                error=ValueError(
                    "Self-Refine initial generation ended with "
                    f"finish_reason={initial_finish_reason!r}"
                ),
            )
        solution = current_generation.text.strip()
        trace: list[dict[str, Any]] = [
            {
                "stage": "initial",
                "solution": solution,
                "finish_reason": initial_finish_reason,
            }
        ]
        candidate_state = (
            "The candidate generation reached its token limit and may end "
            "mid-sentence. Treat it as an incomplete draft and reconstruct "
            "a complete standalone revised solution."
            if initial_finish_reason == "length"
            else ""
        )

        for index in range(max_refinements):
            refinement_prompt = f"""Review and improve the candidate solution to the mathematical problem below.
Check the reasoning, calculations, interpretation of the question, and final
answer. Even when it is already correct, return a complete standalone revised
solution rather than only commenting on it.

{candidate_state}

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
            if current_generation.finish_reason != "stop":
                self._raise_method_failure(
                    current_generation,
                    prompt_profile=prompt_profile,
                    trace=trace,
                    stage="feedback_generation",
                    error=ValueError(
                        "Self-Refine feedback generation did not finish with stop"
                    ),
                )
            try:
                status, feedback, rewritten = self._split_text_refinement(
                    current_generation.text
                )
            except ValueError as exc:
                self._raise_method_failure(
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
        gsm_initial_overrides = config.get(
            "gsm_initial_inference_overrides", {}
        )
        gsm_feedback_overrides = config.get(
            "gsm_feedback_inference_overrides", {}
        )
        text_feedback_overrides = config.get(
            "text_feedback_inference_overrides", {}
        )
        unknown = set(config) - {
            "max_refinements",
            "gsm_initial_inference_overrides",
            "gsm_feedback_inference_overrides",
            "text_feedback_inference_overrides",
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
                feedback_overrides=text_feedback_overrides,
            )
        return self._run_python(
            problem,
            backend,
            max_refinements=max_refinements,
            initial_overrides=gsm_initial_overrides,
            feedback_overrides=gsm_feedback_overrides,
        )
