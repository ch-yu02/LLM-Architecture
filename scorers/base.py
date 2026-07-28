from __future__ import annotations

from typing import Protocol

from benchmark_core.schema import Generation, Problem, Score


class Scorer(Protocol):
    def score(self, problem: Problem, generation: Generation) -> Score: ...
