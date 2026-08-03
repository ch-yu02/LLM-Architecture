#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.aflow import AFlowAdapter, workflow_call_bounds  # noqa: E402
from benchmark_aflow import (  # noqa: E402
    SEARCH_PROTOCOL,
    build_optimizer_prompt,
    candidate_rank,
    parse_workflow_proposal,
    workflow_digest,
)
from benchmark_core.fairness import (  # noqa: E402
    FairnessPolicy,
    evaluation_control_metadata,
)
from benchmark_core.runner import (  # noqa: E402
    EvaluationRunner,
    SampleEvaluationError,
)
from benchmark_core.schema import Experiment, Problem  # noqa: E402
from benchmark_core.store import JsonlResultStore  # noqa: E402
from benchmark_datasets import get_dataset  # noqa: E402
from benchmark_datasets.base import DatasetContext  # noqa: E402
from benchmark_datasets.jsonl import iter_unified_jsonl  # noqa: E402
from benchmark_experiments.artifacts import (  # noqa: E402
    ExperimentArtifactError,
    append_jsonl,
    atomic_write_json,
    clean_git_repository_state,
    configuration_fingerprint,
    ensure_experiment_manifest,
    experiment_directory,
    python_environment_state,
    safe_slug,
    summarize_api_calls,
    summarize_records,
    timestamped_experiment_directory,
)
from benchmark_methods import load_method_config, method_run_settings  # noqa: E402
from model_backends import (  # noqa: E402
    ModelConfigurationError,
    OpenAICompatibleBackend,
    load_model_profile,
    validate_inference_config,
)
from scorers.model_judge import ModelJudgeBackend, judge_protocol_metadata  # noqa: E402
from scripts.run_experiments import _load_locked_u_math_judge  # noqa: E402


AFLOW_DATASETS = (
    "gsm1k",
    "math-perturb",
    "harp",
    "u-math-text-only",
)
VALIDATION_FILES = {
    "gsm1k": "gsm8k_validation.jsonl",
    "math-perturb": "math_perturb_validation.jsonl",
    "harp": "harp_validation.jsonl",
    "u-math-text-only": "u_math_text_only_validation.jsonl",
}
WORKFLOW_FORMAT_VERSION = 1


class ValidationPlugin:
    aliases: tuple[str, ...] = ()

    def __init__(self, delegate, problems: list[Problem], filename: str):
        self.delegate = delegate
        self.problems = problems
        self.name = delegate.name
        self.answer_instruction = delegate.answer_instruction
        self.evaluation_scope = f"validation:{filename}"

    def iter_problems(self, context: DatasetContext) -> Iterable[Problem]:
        yield from self.problems

    def problem_count(self, context: DatasetContext) -> int:
        return len(self.problems)

    def create_scorer(self, context: DatasetContext):
        return self.delegate.create_scorer(context)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _count_or_all(value: str) -> int | None:
    if value.strip().lower() == "all":
        return None
    return _positive_int(value)


def _resolve_model_path(value: str) -> Path:
    provided = Path(value)
    if provided.is_file():
        return provided.resolve()
    candidate = ROOT / "configs" / "models" / f"{value}.toml"
    if candidate.is_file():
        return candidate
    for path in (ROOT / "configs" / "models").glob("*.toml"):
        profile = load_model_profile(path)
        if profile.profile == value or profile.model == value:
            return path
    raise ModelConfigurationError(f"Unknown model profile or config path: {value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_paths(plugin, *, validation_path: Path | None = None) -> tuple[Path, ...]:
    paths = []
    if validation_path is not None:
        paths.append(validation_path)
    else:
        filename = getattr(plugin, "filename", None)
        if isinstance(filename, str) and filename:
            paths.append(Path("data") / "processed" / filename)
    paths.extend(Path(value) for value in getattr(plugin, "revision_paths", ()))
    return tuple(paths)


def _git_artifact_ids(
    data_root: Path,
    *,
    revision: str,
    paths: tuple[Path, ...],
) -> dict[str, str]:
    artifacts = {}
    for path in paths:
        relative = (
            path.relative_to(data_root) if path.is_absolute() else path
        )
        result = subprocess.run(
            [
                "git",
                "-C",
                str(data_root),
                "rev-parse",
                f"{revision}:{relative.as_posix()}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        object_id = result.stdout.strip()
        if result.returncode != 0 or not object_id:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ExperimentArtifactError(
                f"Cannot resolve dataset artifact {relative}: {detail}"
            )
        artifacts[relative.as_posix()] = object_id
    return artifacts


def _configuration_identity(configuration: dict[str, Any]) -> dict[str, Any]:
    identity = json.loads(json.dumps(configuration, ensure_ascii=False))
    identity.pop("audit_revisions", None)
    return identity


def _manifest_configuration(
    path: Path,
    *,
    current: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    if not path.is_file():
        return current
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        stored = manifest["configuration"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ExperimentArtifactError(f"Invalid experiment manifest: {path}") from exc
    if manifest.get("fingerprint") != fingerprint or not isinstance(stored, dict):
        raise ExperimentArtifactError(f"Experiment fingerprint mismatch: {path}")
    if _configuration_identity(stored) != _configuration_identity(current):
        raise ExperimentArtifactError(f"Experiment configuration mismatch: {path}")
    return stored


def _find_artifact_directory(
    output_root: Path,
    *,
    manifest_name: str,
    fingerprint: str,
    identity: dict[str, Any],
) -> Path | None:
    if not output_root.is_dir():
        return None
    matches: list[Path] = []
    for path in sorted(output_root.rglob(manifest_name)):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            stored = manifest["configuration"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        if (
            manifest.get("fingerprint") == fingerprint
            and isinstance(stored, dict)
            and _configuration_identity(stored) == identity
        ):
            matches.append(path.parent)
    if len(matches) > 1:
        raise ExperimentArtifactError(
            "Multiple AFlow artifact directories match the same configuration: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0] if matches else None


def _load_aflow_method() -> tuple[AFlowAdapter, dict[str, Any]]:
    method, tracked = load_method_config(
        ROOT / "configs" / "methods" / "aflow.toml"
    )
    if not isinstance(method, AFlowAdapter):
        raise TypeError("Tracked AFlow config did not construct AFlowAdapter")
    method_config, _ = method_run_settings(tracked)
    return method, method_config


def _method_source_state() -> dict[str, str]:
    config_path = ROOT / "configs" / "methods" / "aflow.toml"
    with config_path.open("rb") as stream:
        import tomllib

        tracked = tomllib.load(stream)
    source = (config_path.parent / tracked["source_path"]).resolve()
    return clean_git_repository_state(source)


def _inference_config(profile, args: argparse.Namespace) -> dict[str, Any]:
    inference = profile.inference_defaults()
    for key, value in (
        ("temperature", args.temperature),
        ("top_p", args.top_p),
        ("max_output_tokens", args.max_output_tokens),
        ("seed", args.seed),
    ):
        if value is not None:
            inference[key] = value
    validate_inference_config(inference)
    return inference


def _optimizer_inference_config(profile, args: argparse.Namespace) -> dict[str, Any]:
    inference = profile.inference_defaults()
    for key, value in (
        ("temperature", args.optimizer_temperature),
        ("max_output_tokens", args.optimizer_max_output_tokens),
        ("seed", args.optimizer_seed),
    ):
        if value is not None:
            inference[key] = value
    validate_inference_config(inference)
    return inference


def _repeat_inference(
    inference: dict[str, Any], repeat: int, seed_mode: str
) -> dict[str, Any]:
    resolved = dict(inference)
    if seed_mode == "increment":
        if "seed" not in resolved:
            raise ModelConfigurationError(
                "--seed-mode increment requires --seed or a profile seed"
            )
        resolved["seed"] += repeat - 1
    validate_inference_config(resolved)
    return resolved


def _public_judge(profile) -> dict[str, Any]:
    return {
        **judge_protocol_metadata(),
        "profile": profile.public_dict(),
        "model": profile.model,
        "inference_config": profile.inference_defaults(),
    }


def _judge_backend(profile, public: dict[str, Any], telemetry: Path):
    return ModelJudgeBackend(
        OpenAICompatibleBackend(profile, telemetry_path=telemetry),
        model_name=profile.model,
        inference_config=public["inference_config"],
    )


def _validation_plugin(
    dataset: str,
    data_root: Path,
    validation_size: int | None,
) -> tuple[ValidationPlugin, Path]:
    delegate = get_dataset(dataset)
    filename = VALIDATION_FILES[dataset]
    path = data_root / "data" / "processed" / filename
    problems = []
    for problem in iter_unified_jsonl(path, dataset):
        problems.append(
            replace(problem, answer_instruction=delegate.answer_instruction)
        )
        if validation_size is not None and len(problems) >= validation_size:
            break
    if not problems:
        raise ValueError(f"Validation split is empty: {path}")
    return ValidationPlugin(delegate, problems, filename), path


def _workflow_payload(path: Path, dataset: str) -> dict[str, Any]:
    if path.is_dir():
        path = path / "workflow.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read frozen workflow {path}: {exc}") from exc
    if payload.get("artifact_type") != "aflow-frozen-workflow":
        raise ValueError("--workflow must point to an AFlow frozen workflow artifact")
    if payload.get("format_version") != WORKFLOW_FORMAT_VERSION:
        raise ValueError("Unsupported AFlow workflow artifact version")
    if payload.get("dataset") != dataset:
        raise ValueError(
            f"Workflow dataset {payload.get('dataset')!r} does not match {dataset!r}"
        )
    workflow = payload.get("workflow")
    artifact = payload.get("workflow_artifact")
    if not isinstance(workflow, dict) or not isinstance(artifact, dict):
        raise ValueError("Frozen workflow artifact is incomplete")
    AFlowAdapter.validate_workflow(workflow)
    if not artifact.get("frozen") or artifact.get("evaluation_data_used") is not False:
        raise ValueError("AFlow test requires a frozen validation-only workflow")
    return {**payload, "path": str(path.resolve()), "sha256": _sha256(path)}


def _failure_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for record in records:
        if record["score"].get("correct") is not False:
            continue
        problem = record["problem"]
        generation = record.get("generation") or {}
        failures.append(
            {
                "problem": problem.get("prompt", "")[:800],
                "reference_answer": str(problem.get("reference_answer", ""))[:400],
                "candidate": generation.get("text", "")[:600],
            }
        )
        if len(failures) >= 3:
            break
    return failures


def _candidate_policy(workflow: dict[str, Any]) -> FairnessPolicy:
    minimum, maximum = workflow_call_bounds(workflow)
    return FairnessPolicy(
        min_model_calls_per_problem=minimum,
        max_model_calls_per_problem=maximum,
    )


def _candidate_artifact(
    *, fingerprint: str, validation_path: Path, validation_sha: str
) -> dict[str, Any]:
    return {
        "source": f"{SEARCH_PROTOCOL}:{fingerprint}",
        "optimization_split": (
            f"validation:{validation_path.name}:{validation_sha[:16]}"
        ),
        "optimization_cost": None,
        "optimization_cost_status": "provider-cost-unavailable",
        "evaluation_data_used": False,
        "frozen": True,
    }


def _evaluate_candidate(
    *,
    method: AFlowAdapter,
    candidate: dict[str, Any],
    search_directory: Path,
    search_fingerprint: str,
    plugin: ValidationPlugin,
    base_context: DatasetContext,
    profile,
    inference: dict[str, Any],
    judge_profile,
    public_judge: dict[str, Any] | None,
    artifact: dict[str, Any],
    concurrency: int,
    progress,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    round_number = candidate["round"]
    directory = search_directory / "candidates" / f"round-{round_number:03d}"
    records_path = directory / "records.jsonl"
    api_path = directory / "api_calls.jsonl"
    error_path = directory / "errors.jsonl"
    workflow = candidate["workflow"]
    policy = _candidate_policy(workflow)
    method_config = {"workflow": workflow, "workflow_artifact": artifact}
    experiment = Experiment(
        id=(
            f"aflow-search::{search_fingerprint}::round-{round_number:03d}::"
            f"{workflow_digest(workflow)}"
        ),
        dataset=plugin.name,
        method="aflow",
        model=profile.model,
        revisions={"search_protocol": SEARCH_PROTOCOL},
        metadata={
            "search_fingerprint": search_fingerprint,
            "search_round": round_number,
            "evaluation_control": evaluation_control_metadata(
                inference, method_config, policy
            ),
        },
    )
    store = JsonlResultStore(records_path)
    completed = store.completed_problem_ids(experiment)
    target = len(plugin.problems) - len(completed)
    task = progress.add_task(
        f"search r{round_number} {plugin.name}",
        total=target,
        latency="lat —",
        tokens="tok 0",
    )
    usage: dict[str, int | float] = {}
    latencies: list[float] = []

    def on_record(record) -> None:
        generation = record.generation
        if generation is not None:
            for key, value in generation.usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] = usage.get(key, 0) + value
        processing = record.timing.get("processing_seconds")
        if processing is not None:
            latencies.append(float(processing))
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        progress.update(
            task,
            advance=1,
            latency=f"lat {mean_latency:.2f}s",
            tokens=f"tok {int(usage.get('total_tokens', 0)):,}",
        )
        if record.score.status.value == "error":
            append_jsonl(
                error_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "problem_id": record.problem.id,
                    "error": record.score.error,
                    "details": record.score.details,
                },
            )

    backend = OpenAICompatibleBackend(profile, telemetry_path=api_path)
    judge = (
        _judge_backend(judge_profile, public_judge, api_path)
        if public_judge is not None
        else None
    )
    context = DatasetContext(
        base_context.data_root,
        bridge_python=base_context.bridge_python,
        judge=judge,
    )
    try:
        EvaluationRunner().run(
            experiment=experiment,
            plugin=plugin,
            context=context,
            method=method,
            backend=backend,
            store=store,
            method_config=method_config,
            inference_config=inference,
            fairness_policy=policy,
            batch_size=None,
            concurrency=concurrency,
            record_callback=on_record,
        )
    finally:
        progress.remove_task(task)
    records = list(store.read())
    summary = summarize_records(records)
    if summary["errors"]:
        raise RuntimeError(f"AFlow candidate round {round_number} has score errors")
    candidate = {
        **candidate,
        "accuracy": summary["accuracy"],
        "scored": summary["scored"],
        "correct": summary["correct"],
        "usage": summary["usage"],
        "records": str(records_path.relative_to(search_directory)),
        "api": summarize_api_calls(api_path),
    }
    return candidate, records


def _search(args, console, components) -> int:
    (
        method,
        default_method_config,
        data_state,
        method_state,
        environment_state,
        profile,
        inference,
        optimizer_profile,
        optimizer_inference,
        judge_profile,
        public_judge,
        base_context,
        progress_factory,
        confirm,
    ) = components
    plugin, validation_path = _validation_plugin(
        args.dataset, args.data_root.resolve(), args.validation_size
    )
    validation_sha = _sha256(validation_path)
    dataset_artifacts = _git_artifact_ids(
        args.data_root.resolve(),
        revision=data_state["revision"],
        paths=_dataset_paths(plugin.delegate, validation_path=validation_path),
    )
    configuration = {
        "mode": "search",
        "protocol": SEARCH_PROTOCOL,
        "dataset": args.dataset,
        "dataset_scope": plugin.evaluation_scope,
        "validation_file": validation_path.relative_to(args.data_root.resolve()).as_posix(),
        "validation_sha256": validation_sha,
        "dataset_artifacts": dataset_artifacts,
        "validation_problem_ids": [problem.id for problem in plugin.problems],
        "search_rounds": args.search_rounds,
        "execution_model": profile.public_dict(),
        "execution_inference": inference,
        "optimizer_model": optimizer_profile.public_dict(),
        "optimizer_inference": optimizer_inference,
        "judge": public_judge,
        "answer_instruction": plugin.answer_instruction,
        "environment": environment_state,
        "audit_revisions": {
            "data_repository": data_state,
            "method_source": method_state,
        },
        "run_tag": args.run_tag,
    }
    identity = _configuration_identity(configuration)
    fingerprint = configuration_fingerprint(identity)
    created_at = datetime.now(timezone.utc).isoformat()
    method_root = (
        args.output_root.resolve() / safe_slug(profile.profile) / "aflow"
    )
    search_components = (
        "search",
        args.dataset,
        *(
            (f"optimizer-{safe_slug(optimizer_profile.profile)}",)
            if optimizer_profile.profile != profile.profile
            else ()
        ),
        *(tuple([args.run_tag]) if args.run_tag.strip() else ()),
    )
    search_directory = _find_artifact_directory(
        args.output_root.resolve(),
        manifest_name="search.json",
        fingerprint=fingerprint,
        identity=identity,
    ) or timestamped_experiment_directory(
        method_root,
        components=search_components,
        fingerprint=fingerprint,
        created_at=created_at,
        claim=False,
    )
    console.print(
        f"[bold]AFlow search[/bold] · dataset={args.dataset} · "
        f"validation={len(plugin.problems)} · rounds={args.search_rounds}"
    )
    console.print(f"Execution model: {profile.profile} → {profile.model}")
    console.print(
        f"Optimizer model: {optimizer_profile.profile} → {optimizer_profile.model}"
    )
    console.print(f"Output: {search_directory}")
    if args.dry_run:
        console.print("[yellow]Dry run complete; no API calls or artifacts.[/yellow]")
        return 0
    if not args.yes and not confirm.ask("Start AFlow search?", default=False):
        return 0

    search_directory = _find_artifact_directory(
        args.output_root.resolve(),
        manifest_name="search.json",
        fingerprint=fingerprint,
        identity=identity,
    ) or timestamped_experiment_directory(
        method_root,
        components=search_components,
        fingerprint=fingerprint,
        created_at=created_at,
        claim=True,
    )

    manifest_path = search_directory / "search.json"
    manifest_configuration = _manifest_configuration(
        manifest_path,
        current=configuration,
        fingerprint=fingerprint,
    )
    ensure_experiment_manifest(
        manifest_path,
        configuration=manifest_configuration,
        fingerprint=fingerprint,
        created_at=created_at,
    )
    state_path = search_directory / "search_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise ExperimentArtifactError("Search state fingerprint mismatch")
    else:
        state = {
            "format_version": 1,
            "fingerprint": fingerprint,
            "status": "running",
            "candidates": [
                {
                    "round": 0,
                    "rationale": "tracked initial workflow",
                    "workflow": default_method_config["workflow"],
                    "accuracy": None,
                }
            ],
        }
        atomic_write_json(state_path, state)

    artifact = _candidate_artifact(
        fingerprint=fingerprint,
        validation_path=validation_path,
        validation_sha=validation_sha,
    )
    progress = progress_factory(console)
    optimizer_backend = OpenAICompatibleBackend(
        optimizer_profile,
        telemetry_path=search_directory / "optimizer_api_calls.jsonl",
    )
    records_by_round: dict[int, list[dict[str, Any]]] = {}
    with progress:
        for round_number in range(args.search_rounds + 1):
            candidate = next(
                (
                    item
                    for item in state["candidates"]
                    if item["round"] == round_number
                ),
                None,
            )
            if candidate is None:
                evaluated = [
                    item
                    for item in state["candidates"]
                    if item.get("accuracy") is not None
                ]
                best = max(evaluated, key=candidate_rank)
                best_records = records_by_round.get(best["round"])
                if best_records is None:
                    best_records = list(
                        JsonlResultStore(
                            search_directory / best["records"]
                        ).read()
                    )
                prompt = build_optimizer_prompt(
                    dataset=args.dataset,
                    answer_instruction=plugin.answer_instruction,
                    best_candidate=best,
                    history=evaluated,
                    failure_examples=_failure_examples(best_records),
                    round_number=round_number,
                )
                proposal = optimizer_backend.generate(
                    [{"role": "user", "content": prompt}],
                    config=optimizer_inference,
                )
                try:
                    workflow, rationale = parse_workflow_proposal(proposal.text)
                except ValueError as exc:
                    append_jsonl(
                        search_directory / "errors.jsonl",
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "stage": "optimizer_proposal",
                            "round": round_number,
                            "error": str(exc),
                            "raw_output": proposal.text,
                            "finish_reason": proposal.finish_reason,
                            "usage": proposal.usage,
                        },
                    )
                    raise
                candidate = {
                    "round": round_number,
                    "rationale": rationale,
                    "workflow": workflow,
                    "accuracy": None,
                    "optimizer_generation": {
                        "finish_reason": proposal.finish_reason,
                        "usage": proposal.usage,
                        "latency_seconds": proposal.latency_seconds,
                    },
                }
                state["candidates"].append(candidate)
                atomic_write_json(state_path, state)
            if candidate.get("accuracy") is None:
                candidate, records = _evaluate_candidate(
                    method=method,
                    candidate=candidate,
                    search_directory=search_directory,
                    search_fingerprint=fingerprint,
                    plugin=plugin,
                    base_context=base_context,
                    profile=profile,
                    inference=inference,
                    judge_profile=judge_profile,
                    public_judge=public_judge,
                    artifact=artifact,
                    concurrency=args.concurrency,
                    progress=progress,
                )
                records_by_round[round_number] = records
                state["candidates"] = [
                    candidate if item["round"] == round_number else item
                    for item in state["candidates"]
                ]
                atomic_write_json(state_path, state)
                console.print(
                    f"round {round_number}: accuracy={candidate['accuracy']:.4f} · "
                    f"workflow={workflow_digest(candidate['workflow'])[:12]}"
                )

    best = max(state["candidates"], key=candidate_rank)
    state["status"] = "complete"
    state["selected_round"] = best["round"]
    atomic_write_json(state_path, state)
    evaluation_usage: dict[str, int | float] = {}
    for candidate in state["candidates"]:
        for key, value in candidate.get("api", {}).get("usage", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                evaluation_usage[key] = evaluation_usage.get(key, 0) + value
    optimizer_api = summarize_api_calls(
        search_directory / "optimizer_api_calls.jsonl"
    )
    frozen = {
        "artifact_type": "aflow-frozen-workflow",
        "format_version": WORKFLOW_FORMAT_VERSION,
        "dataset": args.dataset,
        "workflow": best["workflow"],
        "workflow_artifact": artifact,
        "search": {
            "protocol": SEARCH_PROTOCOL,
            "fingerprint": fingerprint,
            "selected_round": best["round"],
            "validation_accuracy": best["accuracy"],
            "validation_samples": len(plugin.problems),
            "execution_model": profile.public_dict(),
            "optimizer_model": optimizer_profile.public_dict(),
            "optimization_usage": {
                "optimizer": optimizer_api.get("usage", {}),
                "candidate_evaluation": evaluation_usage,
            },
            "candidates": [
                {
                    key: item[key]
                    for key in (
                        "round",
                        "rationale",
                        "accuracy",
                        "scored",
                        "correct",
                        "usage",
                    )
                    if key in item
                }
                for item in state["candidates"]
            ],
        },
    }
    workflow_path = search_directory / "workflow.json"
    atomic_write_json(workflow_path, frozen)
    console.print(
        f"[green]Search complete[/green] · selected round {best['round']} · "
        f"validation accuracy {best['accuracy']:.4f}"
    )
    console.print(f"Frozen workflow: {workflow_path}")
    return 0


def _test(args, console, components) -> int:
    (
        method,
        _,
        data_state,
        method_state,
        environment_state,
        profile,
        inference,
        _,
        _,
        judge_profile,
        public_judge,
        base_context,
        progress_factory,
        confirm,
    ) = components
    payload = _workflow_payload(args.workflow.resolve(), args.dataset)
    workflow = payload["workflow"]
    artifact = payload["workflow_artifact"]
    policy = _candidate_policy(workflow)
    method_config = {"workflow": workflow, "workflow_artifact": artifact}
    plugin = get_dataset(args.dataset)
    total = plugin.problem_count(base_context)
    dataset_artifacts = _git_artifact_ids(
        args.data_root.resolve(),
        revision=data_state["revision"],
        paths=_dataset_paths(plugin),
    )
    test_root = args.output_root.resolve()
    console.print(
        f"[bold]AFlow test[/bold] · dataset={args.dataset} · total={total} · "
        f"workflow={payload['sha256'][:12]}"
    )
    console.print(f"Execution model: {profile.profile} → {profile.model}")
    console.print(
        "Output: "
        f"{test_root / safe_slug(profile.profile) / 'aflow'}"
    )
    if args.dry_run:
        console.print("[yellow]Dry run complete; no API calls or artifacts.[/yellow]")
        return 0
    if not args.yes and not confirm.ask("Start AFlow test?", default=False):
        return 0

    progress = progress_factory(console)
    with progress:
        for repeat in range(1, args.repeats + 1):
            repeat_inference = _repeat_inference(
                inference, repeat, args.seed_mode
            )
            configuration = {
                "mode": "test",
                "method": "aflow",
                "dataset": args.dataset,
                "dataset_scope": getattr(plugin, "evaluation_scope", "test"),
                "model": profile.public_dict(),
                "inference_config": repeat_inference,
                "repeat": repeat,
                "seed_mode": args.seed_mode,
                "workflow_sha256": payload["sha256"],
                "workflow": workflow,
                "workflow_artifact": artifact,
                "fairness_policy": policy.to_dict(),
                "judge": public_judge,
                "answer_instruction": plugin.answer_instruction,
                "dataset_artifacts": dataset_artifacts,
                "environment": environment_state,
                "audit_revisions": {
                    "data_repository": data_state,
                    "method_source": method_state,
                },
                "run_tag": args.run_tag,
            }
            identity = _configuration_identity(configuration)
            fingerprint = configuration_fingerprint(identity)
            created_at = datetime.now(timezone.utc).isoformat()
            directory = _find_artifact_directory(
                test_root,
                manifest_name="experiment.json",
                fingerprint=fingerprint,
                identity=identity,
            ) or experiment_directory(
                test_root,
                model=profile.profile,
                method="aflow",
                dataset=args.dataset,
                repeat=repeat,
                fingerprint=fingerprint,
                created_at=created_at,
                run_tag=args.run_tag,
                claim=True,
            )
            manifest_path = directory / "experiment.json"
            manifest_configuration = _manifest_configuration(
                manifest_path,
                current=configuration,
                fingerprint=fingerprint,
            )
            ensure_experiment_manifest(
                manifest_path,
                configuration=manifest_configuration,
                fingerprint=fingerprint,
                created_at=created_at,
            )
            api_path = directory / "api_calls.jsonl"
            error_path = directory / "errors.jsonl"
            judge = (
                _judge_backend(judge_profile, public_judge, api_path)
                if public_judge is not None
                else None
            )
            context = DatasetContext(
                base_context.data_root,
                bridge_python=base_context.bridge_python,
                judge=judge,
            )
            experiment = Experiment(
                id=f"aflow-test::{fingerprint}",
                dataset=args.dataset,
                method="aflow",
                model=profile.model,
                revisions=manifest_configuration["audit_revisions"],
                metadata={
                    "fingerprint": fingerprint,
                    "repeat": repeat,
                    "workflow_sha256": payload["sha256"],
                    "evaluation_control": evaluation_control_metadata(
                        repeat_inference, method_config, policy
                    ),
                },
            )
            store = JsonlResultStore(directory / "records.jsonl")
            completed = store.completed_problem_ids(experiment)
            remaining = total - len(completed)
            target = (
                remaining
                if args.batch_size is None
                else min(args.batch_size, remaining)
            )
            task = progress.add_task(
                f"r{repeat} aflow × {args.dataset}",
                total=target,
                latency="lat —",
                tokens="tok 0",
            )
            usage: dict[str, int | float] = {}
            latencies: list[float] = []

            def on_record(record) -> None:
                generation = record.generation
                if generation is not None:
                    for key, value in generation.usage.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            usage[key] = usage.get(key, 0) + value
                processing = record.timing.get("processing_seconds")
                if processing is not None:
                    latencies.append(float(processing))
                mean_latency = sum(latencies) / len(latencies) if latencies else 0
                progress.update(
                    task,
                    advance=1,
                    latency=f"lat {mean_latency:.2f}s",
                    tokens=f"tok {int(usage.get('total_tokens', 0)):,}",
                )
                if record.score.status.value == "error":
                    append_jsonl(
                        error_path,
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "problem_id": record.problem.id,
                            "error": record.score.error,
                            "details": record.score.details,
                        },
                    )

            try:
                summary = EvaluationRunner().run(
                    experiment=experiment,
                    plugin=plugin,
                    context=context,
                    method=method,
                    backend=OpenAICompatibleBackend(profile, telemetry_path=api_path),
                    store=store,
                    method_config=method_config,
                    inference_config=repeat_inference,
                    fairness_policy=policy,
                    batch_size=args.batch_size,
                    concurrency=args.concurrency,
                    record_callback=on_record,
                )
            finally:
                progress.remove_task(task)
            aggregate = summarize_records(store.read())
            atomic_write_json(
                directory / "summary.json",
                {
                    "run": {
                        "attempted": summary.attempted,
                        "resumed": summary.resumed,
                        "scored": summary.scored,
                        "correct": summary.correct,
                    },
                    "aggregate": aggregate,
                    "api": summarize_api_calls(api_path),
                    "workflow_sha256": payload["sha256"],
                },
            )
            console.print(
                f"r{repeat} complete · new={summary.scored} · "
                f"resumed={summary.resumed} · accuracy={aggregate['accuracy']}"
            )
    return 0


def _progress_factory(console):
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[latency]}"),
        TextColumn("{task.fields[tokens]}"),
        console=console,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search and test frozen AFlow workflows"
    )
    parser.add_argument("--mode", choices=("search", "test"), required=True)
    parser.add_argument("--dataset", choices=AFLOW_DATASETS, required=True)
    parser.add_argument("--model", default="qwen35_flash")
    parser.add_argument("--model-id")
    parser.add_argument(
        "--optimizer-model",
        help="Optimizer model profile; defaults to --model",
    )
    parser.add_argument("--optimizer-model-id")
    parser.add_argument("--workflow", type=Path)
    parser.add_argument("--search-rounds", type=_positive_int, default=3)
    parser.add_argument(
        "--validation-size",
        type=_count_or_all,
        default=None,
        metavar="N|all",
    )
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument(
        "--batch-size", type=_count_or_all, default=1, metavar="N|all"
    )
    parser.add_argument("--concurrency", type=_positive_int, default=1)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--seed-mode", choices=("fixed", "increment"), default="fixed"
    )
    parser.add_argument("--optimizer-temperature", type=float)
    parser.add_argument("--optimizer-max-output-tokens", type=int, default=4096)
    parser.add_argument("--optimizer-seed", type=int)
    parser.add_argument("--run-tag", default="")
    parser.add_argument(
        "--data-root", type=Path, default=ROOT.parent / "math-benchmark-data"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "results" / "aflow"
    )
    parser.add_argument(
        "--checker-python",
        type=Path,
        default=ROOT / ".venv-checkers" / "bin" / "python",
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        from dotenv import load_dotenv
        from rich.console import Console
        from rich.prompt import Confirm
    except ImportError:
        print("Install requirements/experiment-lock.txt first.", file=sys.stderr)
        return 2
    args = parse_args()
    console = Console()
    load_dotenv(args.env_file, override=False)
    if args.mode == "test" and args.workflow is None:
        console.print("[red]--workflow is required in test mode[/red]")
        return 2
    try:
        # Preserve the virtual-environment launcher path; resolving its symlink
        # to /usr/bin/python would lose the venv prefix and installed packages.
        checker = args.checker_python.absolute()
        if not checker.is_file():
            raise ValueError(f"Checker Python not found: {checker}")
        environment_state = python_environment_state(checker)
        data_state = clean_git_repository_state(args.data_root.resolve())
        method_state = _method_source_state()
        method, default_method_config = _load_aflow_method()
        profile = load_model_profile(_resolve_model_path(args.model))
        if args.model_id:
            profile = replace(profile, model=args.model_id)
        if args.mode == "search":
            optimizer_profile = load_model_profile(
                _resolve_model_path(args.optimizer_model or args.model)
            )
            if args.optimizer_model_id:
                optimizer_profile = replace(
                    optimizer_profile, model=args.optimizer_model_id
                )
            optimizer_inference = _optimizer_inference_config(
                optimizer_profile, args
            )
        else:
            optimizer_profile = None
            optimizer_inference = None
        inference = _inference_config(profile, args)
        judge_profile = (
            _load_locked_u_math_judge()
            if args.dataset == "u-math-text-only"
            else None
        )
        public_judge = (
            _public_judge(judge_profile) if judge_profile is not None else None
        )
        base_context = DatasetContext(
            args.data_root.resolve(),
            bridge_python={
                "math-perturb": str(checker),
                "harp": str(checker),
            },
        )
    except Exception as exc:
        console.print(f"[red]Configuration error:[/red] {type(exc).__name__}: {exc}")
        if args.debug:
            console.print_exception(show_locals=False)
        return 2

    if not args.dry_run and not args.skip_preflight:
        try:
            seen = set()
            preflights = [profile]
            if args.mode == "search":
                preflights.append(optimizer_profile)
            if judge_profile is not None:
                preflights.append(judge_profile)
            for preflight_profile in preflights:
                identity = (
                    preflight_profile.base_url,
                    preflight_profile.model,
                    preflight_profile.api_key_env,
                )
                if identity in seen:
                    continue
                seen.add(identity)
                response = OpenAICompatibleBackend(preflight_profile).generate(
                    [{"role": "user", "content": "Reply with exactly: OK"}],
                    config={
                        **preflight_profile.inference_defaults(),
                        "max_output_tokens": 16,
                    },
                )
                console.print(
                    f"[green]API ready[/green] · {preflight_profile.model} · "
                    f"{response.latency_seconds:.2f}s"
                )
        except Exception as exc:
            console.print(f"[red]API preflight failed:[/red] {type(exc).__name__}: {exc}")
            if args.debug:
                console.print_exception(show_locals=False)
            return 2

    components = (
        method,
        default_method_config,
        data_state,
        method_state,
        environment_state,
        profile,
        inference,
        optimizer_profile,
        optimizer_inference,
        judge_profile,
        public_judge,
        base_context,
        _progress_factory,
        Confirm,
    )
    try:
        return (
            _search(args, console, components)
            if args.mode == "search"
            else _test(args, console, components)
        )
    except SampleEvaluationError as exc:
        console.print(f"[red]Experiment stopped:[/red] {exc}")
        if args.debug:
            console.print(traceback.format_exc())
        return 1
    except Exception as exc:
        console.print(f"[red]AFlow {args.mode} failed:[/red] {type(exc).__name__}: {exc}")
        if args.debug:
            console.print_exception(show_locals=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
