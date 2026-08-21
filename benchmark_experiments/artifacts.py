from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class ExperimentArtifactError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(configuration).encode("utf-8")).hexdigest()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "unnamed"


def experiment_directory(
    output_root: Path,
    *,
    model: str,
    method: str,
    dataset: str,
    repeat: int,
    fingerprint: str,
    created_at: str,
    run_tag: str = "",
    claim: bool = False,
) -> Path:
    method_root = (
        output_root / safe_slug(model) / safe_slug(method)
    )
    components = [
        safe_slug(dataset),
        f"r{repeat:03d}",
    ]
    if run_tag.strip():
        components.append(safe_slug(run_tag))
    return timestamped_experiment_directory(
        method_root,
        components=components,
        fingerprint=fingerprint,
        created_at=created_at,
        claim=claim,
    )


def timestamped_experiment_directory(
    output_root: Path,
    *,
    components: Iterable[str],
    fingerprint: str,
    created_at: str,
    claim: bool = False,
) -> Path:
    prefix = "__".join(safe_slug(component) for component in components)
    fingerprint_component = fingerprint[:16]
    pattern = re.compile(
        rf"^{re.escape(prefix)}__(\d{{8}}T\d{{6}}Z)__"
        rf"{re.escape(fingerprint_component)}$"
    )
    try:
        timestamp = datetime.fromisoformat(
            created_at.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ExperimentArtifactError(
            f"Invalid experiment creation time: {created_at!r}"
        ) from exc
    if timestamp.tzinfo is None:
        raise ExperimentArtifactError(
            f"Experiment creation time must include a timezone: {created_at!r}"
        )
    timestamp = timestamp.astimezone(timezone.utc)
    timestamp_component = timestamp.strftime("%Y%m%dT%H%M%SZ")
    proposed = output_root / (
        f"{prefix}__{timestamp_component}__{fingerprint_component}"
    )

    def find_existing() -> Path | None:
        matches = (
            sorted(
                path
                for path in output_root.iterdir()
                if path.is_dir() and pattern.fullmatch(path.name)
            )
            if output_root.is_dir()
            else []
        )
        if len(matches) > 1:
            raise ExperimentArtifactError(
                "Multiple experiment directories have the same configuration "
                f"fingerprint: {', '.join(str(path) for path in matches)}"
            )
        return matches[0] if matches else None

    existing = find_existing()
    if existing is not None or not claim:
        return existing or proposed

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".experiment-directory.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = find_existing()
        if existing is not None:
            return existing
        proposed.mkdir()
        return proposed


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Durably append one JSON object for sample/fatal error diagnostics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def ensure_experiment_manifest(
    path: Path,
    *,
    configuration: dict[str, Any],
    fingerprint: str,
    created_at: str,
) -> None:
    expected = {
        "fingerprint": fingerprint,
        "configuration": configuration,
    }
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ExperimentArtifactError(f"Invalid experiment manifest: {path}") from exc
        comparable = {
            "fingerprint": existing.get("fingerprint"),
            "configuration": existing.get("configuration"),
        }
        if comparable != expected:
            raise ExperimentArtifactError(
                f"Experiment directory contains a different configuration: {path}"
            )
        return
    atomic_write_json(path, {**expected, "created_at": created_at})


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def clean_git_repository_state(path: Path) -> dict[str, str]:
    """Return reproducible Git state and reject nested, dirty, or invalid roots."""

    resolved = path.resolve()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(resolved), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ExperimentArtifactError(
                f"Git inspection failed for {resolved}: {detail}"
            )
        return result.stdout.strip()

    repository_root = Path(git("rev-parse", "--show-toplevel")).resolve()
    if repository_root != resolved:
        raise ExperimentArtifactError(
            f"Expected an isolated Git repository at {resolved}, "
            f"but its root is {repository_root}"
        )
    status = git("status", "--porcelain", "--untracked-files=normal")
    if status:
        raise ExperimentArtifactError(
            f"Reproducibility requires a clean external repository: {resolved}"
        )
    return {
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current") or "(detached)",
    }


def python_environment_state(python: Path) -> dict[str, Any]:
    """Inspect the exact interpreter and scoring/runtime package versions."""

    distributions = (
        "antlr4-python3-runtime",
        "lark",
        "loguru",
        "numpy",
        "openai",
        "pydantic",
        "pyparsing",
        "python-dotenv",
        "regex",
        "rich",
        "scipy",
        "sympy",
    )
    inspection = """
import importlib.metadata
import json
import platform
import sys

names = json.loads(sys.argv[1])
versions = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "packages": versions,
}, sort_keys=True))
"""
    result = subprocess.run(
        [str(python), "-c", inspection, json.dumps(distributions)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ExperimentArtifactError(
            f"Cannot inspect Python environment {python}: {detail}"
        )
    try:
        state = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ExperimentArtifactError(
            f"Python environment returned invalid metadata: {python}"
        ) from exc
    missing = [
        name for name, version in state["packages"].items() if version is None
    ]
    if missing:
        raise ExperimentArtifactError(
            f"Python environment {python} is missing required packages: {missing}"
        )
    if state["prefix"] == state["base_prefix"]:
        raise ExperimentArtifactError(
            f"Python environment {python} is not an isolated virtual environment"
        )
    return state


def tree_fingerprint(root: Path) -> str:
    """Hash runner source/config content, including uncommitted and untracked files."""

    included_roots = (
        "adapters",
        "benchmark_aflow",
        "benchmark_core",
        "benchmark_datasets",
        "benchmark_experiments",
        "benchmark_methods",
        "bridges",
        "configs",
        "executors",
        "model_backends",
        "requirements",
        "scorers",
        "scripts",
    )
    files: list[Path] = []
    for relative_root in included_roots:
        directory = root / relative_root
        if directory.is_dir():
            files.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and not path.name.endswith((".pyc", ".pyo"))
            )
    for name in ("pyproject.toml", "README.md"):
        path = root / name
        if path.is_file():
            files.append(path)
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(records)
    scored = [item for item in items if item["score"]["status"] == "scored"]
    method_failed = [
        item for item in items if item["score"]["status"] == "method_failed"
    ]
    correct = sum(bool(item["score"].get("correct")) for item in scored)
    errors = sum(item["score"]["status"] == "error" for item in items)
    processing = [
        float(item.get("timing", {}).get("processing_seconds"))
        for item in items
        if item.get("timing", {}).get("processing_seconds") is not None
    ]
    generation = [
        float(item.get("timing", {}).get("generation_seconds"))
        for item in items
        if item.get("timing", {}).get("generation_seconds") is not None
    ]
    usage: dict[str, int | float] = {}
    for item in items:
        for key, value in (item.get("generation") or {}).get("usage", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
    return {
        "records": len(items),
        "scored": len(scored),
        "method_failed": len(method_failed),
        "correct": correct,
        "errors": errors,
        "accuracy": (
            correct / (len(scored) + len(method_failed))
            if scored or method_failed
            else None
        ),
        "usage": usage,
        "processing_seconds": {
            "total": sum(processing),
            "mean": sum(processing) / len(processing) if processing else None,
            "p50": _percentile(processing, 0.50),
            "p95": _percentile(processing, 0.95),
        },
        "generation_seconds": {
            "total": sum(generation),
            "mean": sum(generation) / len(generation) if generation else None,
            "p50": _percentile(generation, 0.50),
            "p95": _percentile(generation, 0.95),
        },
    }


def summarize_api_calls(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "attempts": 0,
            "successes": 0,
            "errors": 0,
            "usage": {},
            "latency_seconds": {},
        }
    calls = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    successes = [call for call in calls if call["status"] == "success"]
    latencies = [float(call["latency_seconds"]) for call in successes]
    usage: dict[str, int | float] = {}
    for call in successes:
        for key, value in call.get("usage", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
    return {
        "attempts": len(calls),
        "successes": len(successes),
        "errors": len(calls) - len(successes),
        "retryable_errors": sum(
            call["status"] == "error" and bool(call.get("retryable"))
            for call in calls
        ),
        "usage": usage,
        "latency_seconds": {
            "total": sum(latencies),
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
    }
