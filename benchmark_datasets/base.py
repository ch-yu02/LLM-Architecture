from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from benchmark_core.schema import Problem
from scorers.base import Scorer
from scorers.judge import JudgeBackend


@dataclass(frozen=True)
class DatasetContext:
    data_root: Path
    bridge_python: dict[str, str] = field(default_factory=dict)
    judge: JudgeBackend | None = None

    def python_for(self, dataset: str) -> str:
        return self.bridge_python.get(dataset, sys.executable)


class DatasetPlugin(Protocol):
    name: str
    aliases: tuple[str, ...]
    answer_instruction: str

    def iter_problems(self, context: DatasetContext) -> Iterable[Problem]: ...

    def create_scorer(self, context: DatasetContext) -> Scorer: ...

    def problem_count(self, context: DatasetContext) -> int: ...
