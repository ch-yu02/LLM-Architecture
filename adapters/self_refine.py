from __future__ import annotations

from typing import Any, Mapping

from .base import IntegrationPendingError, MethodAdapter


class SelfRefineAdapter(MethodAdapter):
    name = "self_refine"
    required_paths = (
        "src/gsm/run.py",
        "src/gsm/feedback.py",
        "src/gsm/task_init.py",
    )

    def generate(
        self,
        problem: Mapping[str, Any],
        experiment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        raise IntegrationPendingError(
            "Self-Refine source is ready; the shared model backend is not integrated yet"
        )

