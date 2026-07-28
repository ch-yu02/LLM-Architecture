"""Minimal source-level adapter contract.

Runtime integration is intentionally separate from source preparation: each method
has incompatible model clients and environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


class IntegrationPendingError(NotImplementedError):
    """Raised when source is present but runtime integration is not implemented."""


class MethodAdapter(ABC):
    name: str
    required_paths: tuple[str, ...]

    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path.resolve()

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
    def generate(
        self,
        problem: Mapping[str, Any],
        experiment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Run one problem and return a serializable generation record."""

