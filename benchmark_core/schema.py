from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


JsonObject = dict[str, Any]


class ScoreStatus(StrEnum):
    SCORED = "scored"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Problem:
    dataset: str
    id: str
    prompt: str
    reference_answer: Any | None = None
    metadata: JsonObject = field(default_factory=dict)
    answer_instruction: str = ""


@dataclass(frozen=True)
class Generation:
    text: str
    finish_reason: str = "stop"
    latency_seconds: float | None = None
    usage: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    status: ScoreStatus
    correct: bool | None = None
    value: float | None = None
    extracted_answer: Any | None = None
    details: JsonObject = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def scored(
        cls,
        correct: bool,
        *,
        extracted_answer: Any | None = None,
        details: JsonObject | None = None,
    ) -> "Score":
        return cls(
            status=ScoreStatus.SCORED,
            correct=correct,
            value=float(correct),
            extracted_answer=extracted_answer,
            details=details or {},
        )

    @classmethod
    def failed(cls, error: str, *, details: JsonObject | None = None) -> "Score":
        return cls(status=ScoreStatus.ERROR, error=error, details=details or {})


@dataclass(frozen=True)
class Experiment:
    id: str
    dataset: str
    method: str
    model: str
    revisions: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRecord:
    experiment: Experiment
    problem: Problem
    generation: Generation | None
    score: Score
    timing: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        value = asdict(self)
        value["score"]["status"] = self.score.status.value
        return value
