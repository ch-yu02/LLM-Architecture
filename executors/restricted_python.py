from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProgramExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    value: Any
    value_type: str


class RestrictedPythonExecutor:
    """Execute arithmetic-only generated Python in an isolated subprocess."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 3,
        cpu_seconds: int = 2,
        memory_mb: int = 256,
        python: str | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.cpu_seconds = cpu_seconds
        self.memory_mb = memory_mb
        self.python = python or sys.executable
        self.worker = Path(__file__).with_name("_worker.py")

    def execute(self, code: str) -> ExecutionResult:
        request = {
            "code": code,
            "cpu_seconds": self.cpu_seconds,
            "memory_mb": self.memory_mb,
        }
        try:
            with tempfile.TemporaryDirectory(prefix="pal-exec-") as directory:
                result = subprocess.run(
                    [self.python, "-I", "-S", str(self.worker)],
                    input=json.dumps(request),
                    text=True,
                    capture_output=True,
                    cwd=directory,
                    env={"PYTHONHASHSEED": "0"},
                    timeout=self.timeout_seconds,
                    check=False,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProgramExecutionError(f"program process failed: {exc}") from exc
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ProgramExecutionError(
                f"program returned invalid result (exit {result.returncode})"
            ) from exc
        if result.returncode != 0 or not payload.get("ok"):
            raise ProgramExecutionError(payload.get("error", "program failed"))
        return ExecutionResult(payload.get("value"), payload["value_type"])
