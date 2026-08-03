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
    """Execute generated mathematical Python in an isolated subprocess."""

    protocol = "pal-python-v4"

    def __init__(
        self,
        *,
        timeout_seconds: float = 10,
        cpu_seconds: int = 8,
        memory_mb: int = 1024,
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
                    [self.python, "-I", str(self.worker)],
                    input=json.dumps(request),
                    text=True,
                    capture_output=True,
                    cwd=directory,
                    env={
                        "PYTHONHASHSEED": "0",
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1",
                    },
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

    def validate_environment(self) -> None:
        """Fail before an experiment if the declared PAL runtime is incomplete."""

        result = self.execute(
            "import numpy as np\n"
            "import scipy\n"
            "import sympy\n"
            "from itertools import combinations\n"
            "def solution():\n"
            "    print('PAL runtime preflight')\n"
            "    _, value = (0, 2)\n"
            "    return int(np.array([value]).sum())"
        )
        if result.value != 2:
            raise ProgramExecutionError(
                f"PAL runtime preflight returned unexpected value: {result.value!r}"
            )
