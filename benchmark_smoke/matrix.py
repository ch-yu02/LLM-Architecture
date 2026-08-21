from __future__ import annotations

import json
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from benchmark_core.runner import EvaluationRunner
from benchmark_core.schema import Experiment, Generation, Problem
from benchmark_core.store import JsonlResultStore
from benchmark_datasets import get_dataset
from benchmark_datasets.base import DatasetContext, DatasetPlugin
from benchmark_experiments.checker_preflight import (
    validate_checker_capabilities,
)
from benchmark_methods import (
    load_method_config,
    method_run_settings,
)
from scorers.judge import JudgeDecision, JudgeVerdict


def _boxed_answer(answer: Any) -> str:
    return rf"\boxed{{{_plain_answer(answer)}}}"


def _plain_answer(answer: Any) -> str:
    text = str(answer).strip()
    if len(text) >= 2 and text.startswith("$") and text.endswith("$"):
        return text[1:-1].strip()
    return text


def _program_for(answer: Any, *, plain_answer: bool = False) -> str:
    value = _plain_answer(answer) if plain_answer else str(answer)
    return f"def solution():\n    return {value!r}"


class CannedSmokeBackend:
    """Positive-path backend for plumbing checks; never a performance baseline."""

    def __init__(self, method: str, dataset: str, answer: Any):
        program = _program_for(
            answer,
            plain_answer=(
                method == "pal"
                and dataset
                in {
                    "math-perturb",
                    "harp",
                    "harp-small",
                    "u-math-text-only",
                }
            ),
        )
        if method == "self_refine":
            if dataset in {
                "math-perturb",
                "harp",
                "harp-small",
                "u-math-text-only",
            }:
                answer_text = _boxed_answer(answer)
                if dataset in {"harp", "harp-small"}:
                    answer_text = f"Answer: {answer_text}"
                self.responses = [
                    answer_text,
                    (
                        "<status>correct</status>\n"
                        "<feedback>The answer is correct.</feedback>\n"
                        f"<revised_solution>{answer_text}</revised_solution>"
                    ),
                ]
            else:
                self.responses = [
                    program,
                    "There is no error in the code.\n# VERDICT: CORRECT",
                ]
        elif method == "pal":
            self.responses = [f"```python\n{program}\n```"]
        else:
            self.responses = [_boxed_answer(answer)]
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        config: dict[str, Any],
    ) -> Generation:
        if not self.responses:
            raise RuntimeError("Canned smoke backend received an unexpected extra call")
        messages_list = list(messages)
        self.calls.append(
            {
                "message_count": len(messages_list),
                "config": dict(config),
            }
        )
        return Generation(
            self.responses.pop(0),
            latency_seconds=0.001,
            usage={
                "input_tokens": sum(
                    len(message["content"].split()) for message in messages_list
                ),
                "output_tokens": 8,
            },
            metadata={"backend": "canned-smoke"},
        )


class ReferenceContainmentSmokeJudge:
    identity = {
        "name": "reference-containment-smoke-judge",
        "purpose": "plumbing-only",
    }

    def judge(self, problem: Problem, candidate: str) -> JudgeDecision:
        reference = str(problem.reference_answer).strip()
        correct = bool(reference) and reference in candidate
        return JudgeDecision(
            JudgeVerdict.YES if correct else JudgeVerdict.NO,
            "Smoke judge only checks that the canned reference survives the pipeline.",
            dict(self.identity),
        )


class SelectedProblemPlugin:
    aliases: tuple[str, ...] = ()

    def __init__(self, delegate: DatasetPlugin, problem: Problem):
        self.delegate = delegate
        self.problem = problem
        self.name = delegate.name

    def iter_problems(self, context: DatasetContext):
        yield self.problem

    def create_scorer(self, context: DatasetContext):
        return self.delegate.create_scorer(context)


@dataclass(frozen=True)
class SmokeCell:
    dataset: str
    method: str
    sample_id: str
    status: str
    correct: bool | None
    score_status: str | None
    model_calls: int
    duration_seconds: float
    output_preview: str | None = None
    error: str | None = None


class SmokeMatrixRunner:
    def __init__(
        self,
        *,
        runner_root: Path,
        data_root: Path,
        checker_python: str,
    ):
        self.runner_root = runner_root.resolve()
        self.data_root = data_root.resolve()
        self.checker_python = checker_python

    @staticmethod
    def load_matrix(path: Path) -> dict[str, Any]:
        with path.open("rb") as stream:
            matrix = tomllib.load(stream)
        methods = matrix.get("methods")
        datasets = matrix.get("datasets")
        if not isinstance(methods, list) or not methods:
            raise ValueError("Smoke matrix requires methods")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("Smoke matrix requires datasets")
        if len(set(methods)) != len(methods):
            raise ValueError("Smoke matrix method names must be unique")
        dataset_names = [item.get("name") for item in datasets]
        if len(set(dataset_names)) != len(dataset_names):
            raise ValueError("Smoke matrix dataset names must be unique")
        if any(not item.get("sample_id") for item in datasets):
            raise ValueError("Every smoke dataset requires a stable sample_id")
        if matrix.get("samples_per_cell") != 1:
            raise ValueError("Stage-8 smoke matrix requires samples_per_cell = 1")
        return matrix

    def _context(self) -> DatasetContext:
        return DatasetContext(
            self.data_root,
            bridge_python={
                "math-perturb": self.checker_python,
                "harp": self.checker_python,
                "harp-small": self.checker_python,
                "mathconstruct": self.checker_python,
            },
            judge=ReferenceContainmentSmokeJudge(),
        )

    def _selected_plugins(
        self,
        matrix: dict[str, Any],
        context: DatasetContext,
    ) -> dict[str, SelectedProblemPlugin]:
        selected: dict[str, SelectedProblemPlugin] = {}
        for item in matrix["datasets"]:
            plugin = get_dataset(item["name"])
            sample_id = item["sample_id"]
            problem = next(
                (
                    problem
                    for problem in plugin.iter_problems(context)
                    if problem.id == sample_id
                ),
                None,
            )
            if problem is None:
                raise ValueError(
                    f"Smoke sample {sample_id!r} not found in {plugin.name}"
                )
            selected[plugin.name] = SelectedProblemPlugin(plugin, problem)
        return selected

    def _load_methods(self, names: list[str]):
        methods = {}
        for name in names:
            config_path = self.runner_root / "configs" / "methods" / f"{name}.toml"
            method, tracked = load_method_config(config_path)
            method_config, fairness = method_run_settings(tracked)
            methods[name] = (method, method_config, fairness)
        return methods

    def run(self, matrix_path: Path) -> dict[str, Any]:
        matrix = self.load_matrix(matrix_path)
        validate_checker_capabilities(
            self.checker_python,
            self.data_root,
            (item["name"] for item in matrix["datasets"]),
        )
        context = self._context()
        plugins = self._selected_plugins(matrix, context)
        methods = self._load_methods(matrix["methods"])
        cells: list[SmokeCell] = []
        started = time.monotonic()

        for dataset_item in matrix["datasets"]:
            dataset_name = dataset_item["name"]
            plugin = plugins[dataset_name]
            problem = plugin.problem
            for method_name in matrix["methods"]:
                method, method_config, fairness = methods[method_name]
                backend = CannedSmokeBackend(
                    method_name,
                    dataset_name,
                    problem.reference_answer,
                )
                cell_started = time.monotonic()
                try:
                    from tempfile import TemporaryDirectory

                    with TemporaryDirectory(prefix="smoke-cell-") as directory:
                        store = JsonlResultStore(Path(directory) / "result.jsonl")
                        summary = EvaluationRunner().run(
                            experiment=Experiment(
                                id=f"smoke::{dataset_name}::{method_name}",
                                dataset=dataset_name,
                                method=method_name,
                                model="canned-smoke-backend",
                                metadata={"purpose": "plumbing-only"},
                            ),
                            plugin=plugin,
                            context=context,
                            method=method,
                            backend=backend,
                            store=store,
                            method_config=method_config,
                            inference_config=matrix.get("inference", {}),
                            fairness_policy=fairness,
                            batch_size=1,
                        )
                        record = list(store.read())[0]
                    score = record["score"]
                    generation = record.get("generation")
                    passed = (
                        summary.attempted == 1
                        and summary.scored == 1
                        and summary.correct == 1
                        and summary.method_failed == 0
                        and summary.errors == 0
                    )
                    cells.append(
                        SmokeCell(
                            dataset=dataset_name,
                            method=method_name,
                            sample_id=problem.id,
                            status="passed" if passed else "failed",
                            correct=score.get("correct"),
                            score_status=score.get("status"),
                            model_calls=len(backend.calls),
                            duration_seconds=round(
                                time.monotonic() - cell_started, 6
                            ),
                            output_preview=(
                                generation["text"][:200] if generation else None
                            ),
                            error=None if passed else score.get("error"),
                        )
                    )
                except Exception as exc:
                    cells.append(
                        SmokeCell(
                            dataset=dataset_name,
                            method=method_name,
                            sample_id=problem.id,
                            status="failed",
                            correct=None,
                            score_status=None,
                            model_calls=len(backend.calls),
                            duration_seconds=round(
                                time.monotonic() - cell_started, 6
                            ),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )

        passed = sum(cell.status == "passed" for cell in cells)
        return {
            "matrix": matrix["name"],
            "purpose": "plumbing-only; not a model performance result",
            "expected_cells": len(matrix["datasets"]) * len(matrix["methods"]),
            "passed": passed,
            "failed": len(cells) - passed,
            "duration_seconds": round(time.monotonic() - started, 6),
            "cells": [asdict(cell) for cell in cells],
        }

    @staticmethod
    def write_report(report: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
