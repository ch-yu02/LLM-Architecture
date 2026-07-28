from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, replace
from typing import Any, Callable

from benchmark_datasets.base import DatasetContext, DatasetPlugin

from .fairness import (
    ControlledModelBackend,
    FairnessPolicy,
    evaluation_control_metadata,
)
from .interfaces import EvaluationMethod, ModelBackend
from .schema import EvaluationRecord, Experiment, Generation, Score
from .store import JsonlResultStore


@dataclass(frozen=True)
class RunSummary:
    attempted: int
    resumed: int
    scored: int
    correct: int
    errors: int

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.scored if self.scored else None


class EvaluationRunner:
    """Dataset-agnostic orchestration; dataset behavior lives in plugins."""

    def run(
        self,
        *,
        experiment: Experiment,
        plugin: DatasetPlugin,
        context: DatasetContext,
        method: EvaluationMethod,
        backend: ModelBackend,
        store: JsonlResultStore,
        method_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
        fairness_policy: FairnessPolicy | None = None,
        batch_size: int | None = None,
        record_callback: Callable[[EvaluationRecord], None] | None = None,
    ) -> RunSummary:
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        if experiment.dataset != plugin.name:
            raise ValueError(
                f"Experiment dataset {experiment.dataset!r} != plugin {plugin.name!r}"
            )
        if experiment.method != method.name:
            raise ValueError(
                f"Experiment method {experiment.method!r} != method {method.name!r}"
            )
        policy = fairness_policy or FairnessPolicy()
        resolved_method_config = method_config or {}
        control = evaluation_control_metadata(
            inference_config or {}, resolved_method_config, policy
        )
        existing_control = experiment.metadata.get("evaluation_control")
        if existing_control is not None and existing_control != control:
            raise ValueError("Experiment contains conflicting evaluation_control metadata")
        resolved_experiment = replace(
            experiment,
            metadata={**experiment.metadata, "evaluation_control": control},
        )
        with store.exclusive_run():
            return self._run_exclusive(
                experiment=resolved_experiment,
                plugin=plugin,
                context=context,
                method=method,
                backend=backend,
                store=store,
                method_config=resolved_method_config,
                inference_config=inference_config or {},
                fairness_policy=policy,
                batch_size=batch_size,
                record_callback=record_callback,
            )

    @staticmethod
    def _run_exclusive(
        *,
        experiment: Experiment,
        plugin: DatasetPlugin,
        context: DatasetContext,
        method: EvaluationMethod,
        backend: ModelBackend,
        store: JsonlResultStore,
        method_config: dict[str, Any],
        inference_config: dict[str, Any],
        fairness_policy: FairnessPolicy,
        batch_size: int | None,
        record_callback: Callable[[EvaluationRecord], None] | None,
    ) -> RunSummary:
        completed = store.completed_problem_ids(experiment)
        scorer = plugin.create_scorer(context)
        attempted = resumed = scored = correct = errors = 0
        for problem in plugin.iter_problems(context):
            if problem.id in completed:
                resumed += 1
                continue
            if batch_size is not None and attempted >= batch_size:
                break
            attempted += 1
            generation: Generation | None = None
            controlled_backend: ControlledModelBackend | None = None
            sample_started = time.perf_counter()
            scoring_seconds: float | None = None
            failure_stage = "method"
            try:
                controlled_backend = ControlledModelBackend(
                    backend,
                    inference_config=inference_config,
                    policy=fairness_policy,
                )
                generation = method.run(
                    problem, controlled_backend, config=method_config
                )
                generation = controlled_backend.finalize(generation)
                failure_stage = "scoring"
                scoring_started = time.perf_counter()
                score = scorer.score(problem, generation)
                scoring_seconds = time.perf_counter() - scoring_started
            except Exception as exc:  # one bad sample must not abort a long run
                details = {
                    "failure_stage": failure_stage,
                    "exception_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }
                if controlled_backend is not None:
                    details["evaluation_control"] = {
                        "model_calls": controlled_backend.model_calls
                    }
                score = Score.failed(
                    f"{type(exc).__name__}: {exc}", details=details
                )
            if score.status.value == "scored":
                scored += 1
                correct += int(bool(score.correct))
            elif score.status.value == "error":
                errors += 1
            record = EvaluationRecord(
                experiment=experiment,
                problem=problem,
                generation=generation,
                score=score,
                timing={
                    "processing_seconds": time.perf_counter() - sample_started,
                    "generation_seconds": (
                        generation.latency_seconds
                        if generation is not None
                        else None
                    ),
                    "scoring_seconds": scoring_seconds,
                },
            )
            store.append(record)
            if record_callback is not None:
                record_callback(record)
        return RunSummary(attempted, resumed, scored, correct, errors)
