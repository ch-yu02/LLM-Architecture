from __future__ import annotations

from typing import Any, Mapping

from .base import IntegrationPendingError, MethodAdapter


class AFlowAdapter(MethodAdapter):
    name = "aflow"
    required_paths = ("run.py", "run_baseline.py", "config/config2.example.yaml")

    def generate(
        self,
        problem: Mapping[str, Any],
        experiment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        raise IntegrationPendingError(
            "AFlow source is ready; custom benchmark classes are not integrated yet"
        )

