"""Shared source validation for paper method adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from benchmark_core.interfaces import EvaluationMethod, ModelBackend
from benchmark_core.schema import Generation, Problem


class MethodAdapter(ABC, EvaluationMethod):
    name: str
    required_paths: tuple[str, ...]

    def __init__(self, source_path: Path | str) -> None:
        self.source_path = Path(source_path).resolve()

    def validate_source(self) -> None:
        if not self.source_path.is_dir():
            raise FileNotFoundError(f"Missing method source: {self.source_path}")
        missing = [
            relative
            for relative in self.required_paths
            if not (self.source_path / relative).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{self.name} source is incomplete; missing: {', '.join(missing)}"
            )

    @abstractmethod
    def run(
        self,
        problem: Problem,
        backend: ModelBackend,
        *,
        config: dict[str, Any],
    ) -> Generation:
        """Run one problem through the shared model backend."""
