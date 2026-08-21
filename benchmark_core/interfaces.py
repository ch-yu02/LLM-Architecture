from __future__ import annotations

from typing import Any, Protocol, Sequence

from .schema import Generation, Problem


class MethodOutcomeError(RuntimeError):
    """A completed model attempt that did not produce a usable method output."""

    def __init__(
        self,
        generation: Generation,
        *,
        stage: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.generation = generation
        self.stage = stage
        self.reason = reason
        self.details = details or {}
        super().__init__(f"{stage}: {reason}")


class ModelBackend(Protocol):
    def generate(
        self, messages: Sequence[dict[str, str]], *, config: dict[str, Any]
    ) -> Generation: ...


class EvaluationMethod(Protocol):
    name: str

    def run(
        self, problem: Problem, backend: ModelBackend, *, config: dict[str, Any]
    ) -> Generation: ...
