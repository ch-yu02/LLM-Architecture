from __future__ import annotations

from typing import Any, Mapping

from .base import IntegrationPendingError, MethodAdapter


class PALAdapter(MethodAdapter):
    name = "pal"
    required_paths = ("pal/core/interface.py", "pal/prompt/math_prompts.py")

    def generate(
        self,
        problem: Mapping[str, Any],
        experiment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        raise IntegrationPendingError(
            "PAL source is ready; sandboxed program execution is not integrated yet"
        )

