from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from benchmark_core.interfaces import ModelBackend
from benchmark_core.schema import Problem

from .judge import JudgeDecision, JudgeVerdict


PROTOCOL_NAME = "u-math-manual-cot-self-verdict"
PROTOCOL_VERSION = 1
_VERDICT_LINE = re.compile(
    r"^\s*(?:final\s+verdict\s*[:\-]\s*)?"
    r"(Yes|No|Inconclusive)\s*[.!]?\s*$",
    flags=re.IGNORECASE,
)


def judge_protocol_metadata() -> dict[str, Any]:
    return {
        "name": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "prompt": "u-math-official-manual-cot-adapted-tristate",
        "verdicts": [verdict.value for verdict in JudgeVerdict],
        "inconclusive_treatment": "incorrect",
        "verdict_extractor": None,
    }


def _parse_verdict(text: str) -> tuple[JudgeVerdict, str]:
    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        return JudgeVerdict.INCONCLUSIVE, "malformed-output-fallback"
    match = _VERDICT_LINE.fullmatch(nonempty_lines[-1])
    if match is None:
        return JudgeVerdict.INCONCLUSIVE, "malformed-final-line-fallback"
    normalized = match.group(1).lower()
    verdict = {
        "yes": JudgeVerdict.YES,
        "no": JudgeVerdict.NO,
        "inconclusive": JudgeVerdict.INCONCLUSIVE,
    }[normalized]
    return verdict, "standalone-final-line"


@dataclass(frozen=True)
class ModelJudgeBackend:
    backend: ModelBackend
    model_name: str
    inference_config: dict[str, Any]

    @property
    def identity(self) -> dict[str, Any]:
        return {
            **judge_protocol_metadata(),
            "model": self.model_name,
            "inference_config": self.inference_config,
        }

    def judge(self, problem: Problem, candidate: str) -> JudgeDecision:
        prompt = f"""You’ll be provided with a math problem, a correct answer for it and a solution for evaluation.
You have to answer whether the solution is correct or not.
---
PROBLEM STATEMENT:
{problem.prompt}
CORRECT ANSWER:
{problem.reference_answer}
SOLUTION TO EVALUATE:
{candidate}
---
Now please compare the answer obtained in the solution with the provided correct answer to evaluate whether the solution is correct or not.
Think step-by-step, following these steps, don’t skip any:
1. Extract the answer from the provided solution
2. Make any derivations or transformations that may be necessary to compare the provided correct answer with the extracted answer
3. Perform the comparison
4. Conclude with your final verdict — put exactly one of "Yes", "No", or "Inconclusive" on a separate final line

Use "Inconclusive" only when you cannot reliably determine whether the solution is correct.
"""
        generation = self.backend.generate(
            [{"role": "user", "content": prompt}],
            config=self.inference_config,
        )
        text = generation.text.strip()
        if generation.finish_reason != "stop":
            verdict = JudgeVerdict.INCONCLUSIVE
            parse_status = "judge-non-stop-fallback"
        else:
            verdict, parse_status = _parse_verdict(text)
        return JudgeDecision(
            verdict,
            text,
            {
                **self.identity,
                "verdict": verdict.value,
                "verdict_parse_status": parse_status,
                "usage": generation.usage,
                "latency_seconds": generation.latency_seconds,
                "finish_reason": generation.finish_reason,
            },
        )
