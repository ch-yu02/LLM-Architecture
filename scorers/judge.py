from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from benchmark_core.schema import Generation, Problem, Score


class JudgeVerdict(StrEnum):
    YES = "Yes"
    NO = "No"
    INCONCLUSIVE = "Inconclusive"


@dataclass(frozen=True)
class JudgeDecision:
    verdict: JudgeVerdict
    rationale: str
    metadata: dict

    @property
    def correct(self) -> bool:
        return self.verdict is JudgeVerdict.YES


class JudgeBackend(Protocol):
    def judge(self, problem: Problem, candidate: str) -> JudgeDecision: ...


class JudgeRequiredError(RuntimeError):
    pass


class UMathJudgeScorer:
    def __init__(self, judge: JudgeBackend | None):
        self.judge = judge

    def score(self, problem: Problem, generation: Generation) -> Score:
        if self.judge is None:
            raise JudgeRequiredError(
                "U-MATH requires an explicit JudgeBackend; no string-match fallback is used"
            )
        decision = self.judge.judge(problem, generation.text)
        return Score.scored(
            decision.correct,
            details={
                "judge_verdict": decision.verdict.value,
                "judge_rationale": decision.rationale,
                "judge_metadata": decision.metadata,
            },
        )
