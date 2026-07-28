from __future__ import annotations

from typing import Any, Protocol, Sequence

from .schema import Generation, Problem


class ModelBackend(Protocol):
    def generate(
        self, messages: Sequence[dict[str, str]], *, config: dict[str, Any]
    ) -> Generation: ...


class EvaluationMethod(Protocol):
    name: str

    def run(
        self, problem: Problem, backend: ModelBackend, *, config: dict[str, Any]
    ) -> Generation: ...
