from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


class BridgeError(RuntimeError):
    pass


def _json_with_unlimited_trusted_integers(
    operation: Callable[[], Any],
) -> Any:
    if not hasattr(sys, "set_int_max_str_digits"):
        return operation()
    previous_limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        return operation()
    finally:
        sys.set_int_max_str_digits(previous_limit)


class JsonSubprocessBridge:
    """Isolates official checker dependencies from the evaluation process."""

    def __init__(
        self,
        script: Path,
        source_root: Path,
        *,
        python: str | None = None,
        timeout_seconds: float = 30,
    ):
        self.script = script
        self.source_root = source_root
        self.python = python or sys.executable
        self.timeout_seconds = timeout_seconds

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        # MathConstruct contains trusted reference integers longer than CPython's
        # default conversion guard. They must survive a JSON round trip unchanged.
        request_json = _json_with_unlimited_trusted_integers(
            lambda: json.dumps(request, ensure_ascii=False)
        )
        try:
            result = subprocess.run(
                [self.python, str(self.script), str(self.source_root)],
                input=request_json,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BridgeError(f"checker process failed: {exc}") from exc
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise BridgeError(stderr or f"checker exited {result.returncode}")
        try:
            response = _json_with_unlimited_trusted_integers(
                lambda: json.loads(result.stdout)
            )
        except json.JSONDecodeError as exc:
            raise BridgeError(f"checker returned invalid JSON: {result.stdout[:200]}") from exc
        if response.get("error"):
            raise BridgeError(str(response["error"]))
        return response
