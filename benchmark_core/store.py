from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from .schema import EvaluationRecord, Experiment


class ExperimentConflictError(ValueError):
    pass


def _definition_hash(experiment: Experiment) -> str:
    encoded = json.dumps(
        asdict(experiment),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compact_method_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "provider",
            "profile",
            "model",
            "request_id",
            "attempt_count",
            "retry_wait_seconds",
            "api_latency_seconds",
            "evaluation_control",
            "workflow",
            "workflow_artifact",
            "refinement_trace",
            "workflow_trace",
        }
    }
    refinement_trace = metadata.get("refinement_trace")
    if isinstance(refinement_trace, list):
        compact["refinement_iterations"] = sum(
            item.get("stage") == "refinement"
            for item in refinement_trace
            if isinstance(item, dict)
        )
    workflow_trace = metadata.get("workflow_trace")
    if isinstance(workflow_trace, list):
        steps = []
        for item in workflow_trace:
            if not isinstance(item, dict):
                continue
            step = {
                "id": item.get("id"),
                "operator": item.get("operator"),
            }
            if isinstance(item.get("attempts"), list):
                step["attempts"] = [
                    {
                        key: attempt[key]
                        for key in ("attempt", "status", "error")
                        if key in attempt
                    }
                    for attempt in item["attempts"]
                    if isinstance(attempt, dict)
                ]
            steps.append(step)
        compact["workflow_steps"] = steps
    return compact


def _compact_score_details(details: dict[str, Any]) -> dict[str, Any]:
    compact = dict(details)
    compact.pop("traceback", None)
    control = compact.get("evaluation_control")
    if isinstance(control, dict):
        compact["evaluation_control"] = {
            "model_calls": control.get("model_calls")
        }
    judge = compact.get("judge_metadata")
    if isinstance(judge, dict):
        compact["judge_metadata"] = {
            key: judge[key]
            for key in (
                "verdict",
                "verdict_parse_status",
                "usage",
                "latency_seconds",
            )
            if key in judge
        }
    return compact


def _compact_record(record: EvaluationRecord) -> dict[str, Any]:
    generation = record.generation
    control = (
        generation.metadata.get("evaluation_control", {})
        if generation is not None
        else {}
    )
    if generation is not None:
        compact_generation: dict[str, Any] | None = {
            "text": generation.text,
            "finish_reason": generation.finish_reason,
            "usage": generation.usage,
        }
        if generation.latency_seconds is not None:
            compact_generation["latency_seconds"] = generation.latency_seconds
        if control.get("model_calls") is not None:
            compact_generation["model_calls"] = control["model_calls"]
        method_details = _compact_method_metadata(generation.metadata)
        if method_details:
            compact_generation["method_details"] = method_details
    else:
        compact_generation = None
    score: dict[str, Any] = {"status": record.score.status.value}
    if record.score.correct is not None:
        score["correct"] = record.score.correct
    if record.score.extracted_answer is not None:
        score["extracted_answer"] = record.score.extracted_answer
    details = _compact_score_details(record.score.details)
    if details:
        score["details"] = details
    if record.score.error is not None:
        score["error"] = record.score.error
    return {
        "record_type": "sample",
        "problem": {
            "id": record.problem.id,
            "prompt": record.problem.prompt,
            "reference_answer": record.problem.reference_answer,
            "metadata": record.problem.metadata,
        },
        "generation": compact_generation,
        "score": score,
        "timing": {
            key: value
            for key, value in record.timing.items()
            if value is not None
        },
    }


class JsonlResultStore:
    """Compact append-only result store with sample-level resume support."""

    format_version = 2

    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def exclusive_run(self) -> Iterator[None]:
        """Prevent concurrent runners from selecting the same unfinished samples."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        with lock_path.open("a", encoding="utf-8") as lock_stream:
            try:
                fcntl.flock(
                    lock_stream.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError as exc:
                raise ExperimentConflictError(
                    f"Another process is already running this experiment: "
                    f"{self.path}"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _header(experiment: Experiment) -> dict[str, Any]:
        return {
            "record_type": "experiment",
            "format_version": JsonlResultStore.format_version,
            "experiment_id": experiment.id,
            "dataset": experiment.dataset,
            "method": experiment.method,
            "model": experiment.model,
            "definition_hash": _definition_hash(experiment),
        }

    def completed_problem_ids(self, experiment: Experiment) -> set[str]:
        if not self.path.exists():
            return set()
        expected_hash = _definition_hash(experiment)
        completed: set[str] = set()
        definition_confirmed = False
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    record_type = item.get("record_type")
                    if record_type == "experiment":
                        if (
                            item.get("format_version") != self.format_version
                            or item.get("experiment_id") != experiment.id
                            or item.get("dataset") != experiment.dataset
                            or item.get("method") != experiment.method
                            or item.get("model") != experiment.model
                            or item.get("definition_hash") != expected_hash
                        ):
                            raise ExperimentConflictError(
                                f"Result file contains a different experiment: "
                                f"{self.path}"
                            )
                        definition_confirmed = True
                        continue
                    if record_type == "sample":
                        if not definition_confirmed:
                            raise ValueError(
                                "compact sample appears before experiment header"
                            )
                        completed.add(str(item["problem"]["id"]))
                        continue

                    raise ValueError(
                        f"unsupported record_type {record_type!r}; "
                        f"expected format version {self.format_version}"
                    )
                except ExperimentConflictError:
                    raise
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid result at {self.path}:{line_number}: {exc}"
                    ) from exc
        return completed

    def append(self, record: EvaluationRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        lines = []
        if write_header:
            lines.append(
                json.dumps(
                    self._header(record.experiment),
                    ensure_ascii=False,
                )
            )
        lines.append(json.dumps(_compact_record(record), ensure_ascii=False))
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def read(self) -> Iterable[dict]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("record_type") == "experiment":
                    continue
                if item.get("record_type") != "sample":
                    raise ValueError(
                        f"Unsupported result format in {self.path}"
                    )
                yield item
