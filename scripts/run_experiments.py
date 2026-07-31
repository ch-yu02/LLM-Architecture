#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import tomllib
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_core.fairness import (  # noqa: E402
    FairnessPolicy,
    FairnessViolation,
    evaluation_control_metadata,
)
from benchmark_core.runner import (  # noqa: E402
    EvaluationRunner,
    SampleEvaluationError,
)
from benchmark_core.schema import Experiment  # noqa: E402
from benchmark_core.store import JsonlResultStore  # noqa: E402
from benchmark_datasets import get_dataset  # noqa: E402
from benchmark_datasets.base import DatasetContext  # noqa: E402
from benchmark_experiments.artifacts import (  # noqa: E402
    ExperimentArtifactError,
    append_jsonl,
    atomic_write_json,
    clean_git_repository_state,
    configuration_fingerprint,
    ensure_experiment_manifest,
    experiment_directory,
    git_revision,
    python_environment_state,
    safe_slug,
    summarize_api_calls,
    summarize_records,
    tree_fingerprint,
)
from benchmark_methods import (  # noqa: E402
    get_method,
    load_method_config,
    method_run_settings,
)
from model_backends import (  # noqa: E402
    ModelConfigurationError,
    OpenAICompatibleBackend,
    load_model_profile,
    validate_inference_config,
)
from scorers.model_judge import (  # noqa: E402
    ModelJudgeBackend,
    judge_protocol_metadata,
)


PAPER_METHODS = ("pal", "self_refine", "aflow")
ALL_METHODS = ("direct", "zero_shot_cot", *PAPER_METHODS)
PRIMARY_DATASETS = (
    "gsm1k",
    "math-perturb",
    "harp",
    "u-math-text-only",
    "mathconstruct",
)
SELECTABLE_DATASETS = (*PRIMARY_DATASETS, "harp-small")
U_MATH_JUDGE_LOCK = ROOT / "configs" / "judges" / "u_math.lock.toml"


def _csv_selection(value: str, available: tuple[str, ...], label: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(available)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown {label}: {sorted(unknown)}")
    if not selected:
        raise ValueError(f"No {label} selected")
    return selected


def _dataset_selection(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(PRIMARY_DATASETS)
    return _csv_selection(value, SELECTABLE_DATASETS, "datasets")


def _normalize_profile_metadata(profile: dict[str, Any]) -> None:
    if profile.get("thinking_type") is None:
        profile.pop("thinking_type", None)
    if profile.get("reasoning_effort") is None:
        profile.pop("reasoning_effort", None)
    if profile.get("omit_sampling_parameters") is False:
        profile.pop("omit_sampling_parameters", None)


def _experiment_identity(
    configuration: dict[str, Any],
    *,
    dataset_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    identity = copy.deepcopy(configuration)
    revisions = identity.get("revisions")
    if isinstance(revisions, dict):
        revisions.pop("runner_git", None)
        revisions.pop("runner_tree", None)
        if dataset_artifacts is not None:
            revisions.pop("data", None)
        if not revisions:
            identity.pop("revisions")
    if dataset_artifacts is not None:
        identity["dataset_artifacts"] = dataset_artifacts
    model = identity.get("model")
    if isinstance(model, dict):
        _normalize_profile_metadata(model)
    judge = identity.get("judge")
    if isinstance(judge, dict):
        judge_profile = judge.get("profile")
        if isinstance(judge_profile, dict):
            _normalize_profile_metadata(judge_profile)
    return identity


def _dataset_revision_paths(plugin) -> tuple[Path, ...]:
    filename = getattr(plugin, "filename", None)
    if not isinstance(filename, str) or not filename:
        return ()
    paths = [Path("data") / "processed" / filename]
    paths.extend(
        Path(path) for path in getattr(plugin, "revision_paths", ())
    )
    return tuple(paths)


def _git_object_ids(
    data_root: Path,
    *,
    revision: str,
    paths: tuple[Path, ...],
) -> dict[str, str]:
    if not paths:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(data_root),
                "rev-parse",
                f"{revision}^{{tree}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        object_id = result.stdout.strip()
        if result.returncode != 0 or not object_id:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ExperimentArtifactError(
                f"Cannot resolve dataset repository tree {revision}: {detail}"
            )
        return {"__repository_tree__": object_id}
    objects: dict[str, str] = {}
    for path in paths:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(data_root),
                "rev-parse",
                f"{revision}:{path.as_posix()}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        object_id = result.stdout.strip()
        if result.returncode != 0 or not object_id:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ExperimentArtifactError(
                f"Cannot resolve dataset artifact {revision}:{path}: {detail}"
            )
        objects[path.as_posix()] = object_id
    return objects


def _result_problem_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    header_seen = False
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if item.get("record_type") == "experiment":
                    if header_seen or completed:
                        raise ValueError("duplicate or misplaced header")
                    header_seen = True
                elif item.get("record_type") == "sample" and header_seen:
                    problem_id = str(item["problem"]["id"])
                    if problem_id in completed:
                        raise ValueError(f"duplicate problem ID {problem_id}")
                    completed.add(problem_id)
                else:
                    raise ValueError("invalid record order or type")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ExperimentArtifactError(
                    f"Invalid result at {path}:{line_number}: {exc}"
                ) from exc
    return completed


def _find_resume_experiment(
    output_root: Path,
    *,
    components: tuple[str, ...],
    identity: dict[str, Any],
    data_root: Path,
    dataset_paths: tuple[Path, ...],
) -> tuple[Path, str, dict[str, Any]] | None:
    if not output_root.is_dir():
        return None
    prefix = "__".join(safe_slug(component) for component in components) + "__"
    matches: list[tuple[Path, str, dict[str, Any], set[str]]] = []
    for directory in sorted(output_root.iterdir()):
        if not directory.is_dir() or not directory.name.startswith(prefix):
            continue
        manifest_path = directory / "experiment.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fingerprint = manifest["fingerprint"]
            stored_configuration = manifest["configuration"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ExperimentArtifactError(
                f"Invalid experiment manifest: {manifest_path}"
            ) from exc
        if not isinstance(fingerprint, str) or not isinstance(
            stored_configuration, dict
        ):
            raise ExperimentArtifactError(
                f"Invalid experiment manifest: {manifest_path}"
            )
        stored_artifacts = stored_configuration.get("dataset_artifacts")
        if not isinstance(stored_artifacts, dict):
            try:
                stored_revision = stored_configuration["revisions"]["data"][
                    "revision"
                ]
            except (KeyError, TypeError) as exc:
                raise ExperimentArtifactError(
                    f"Manifest does not identify its dataset: {manifest_path}"
                ) from exc
            stored_artifacts = _git_object_ids(
                data_root,
                revision=str(stored_revision),
                paths=dataset_paths,
            )
        if (
            _experiment_identity(
                stored_configuration,
                dataset_artifacts=stored_artifacts,
            )
            == identity
        ):
            matches.append(
                (
                    directory,
                    fingerprint,
                    stored_configuration,
                    _result_problem_ids(directory / "records.jsonl"),
                )
            )
    if not matches:
        return None
    matches.sort(key=lambda item: len(item[3]), reverse=True)
    best = matches[0]
    if any(not item[3].issubset(best[3]) for item in matches[1:]):
        raise ExperimentArtifactError(
            "Multiple matching experiment directories contain divergent "
            f"sample sets: {', '.join(str(item[0]) for item in matches)}"
        )
    if len(matches) > 1 and len(matches[1][3]) == len(best[3]):
        raise ExperimentArtifactError(
            "Multiple matching experiment directories have equal progress: "
            f"{', '.join(str(item[0]) for item in matches)}"
        )
    return best[0], best[1], best[2]


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_batch_size(value: str) -> int | None:
    if value.strip().lower() == "all":
        return None
    try:
        size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--batch-size must be a positive integer or 'all'"
        ) from exc
    if size < 1:
        raise argparse.ArgumentTypeError(
            "--batch-size must be a positive integer or 'all'"
        )
    return size


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        )
    return parsed


def _repeat_inference_config(
    inference: dict[str, Any],
    *,
    repeat_index: int,
    seed_mode: str,
) -> dict[str, Any]:
    resolved = copy.deepcopy(inference)
    validate_inference_config(resolved)
    if repeat_index < 1:
        raise ModelConfigurationError("repeat_index must be at least 1")
    if seed_mode == "fixed":
        return resolved
    if seed_mode != "increment":
        raise ModelConfigurationError(f"Unknown seed mode: {seed_mode}")
    if "seed" not in resolved:
        raise ModelConfigurationError(
            "--seed-mode increment requires --seed or seed in the model profile"
        )
    resolved["seed"] += repeat_index - 1
    validate_inference_config(resolved)
    return resolved


def _deep_merge(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = target
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set nested parameter beneath {part!r}")
        current = child
    current[parts[-1]] = value


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".toml":
        import tomllib

        with path.open("rb") as stream:
            value = tomllib.load(stream)
    else:
        raise ValueError(f"Method config must be JSON or TOML: {path}")
    if not isinstance(value, dict):
        raise ValueError(f"Method config must contain an object/table: {path}")
    return value


def _method_overrides(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    overrides = {name: {} for name in ALL_METHODS}
    for specification in args.method_config:
        method, separator, path_value = specification.partition("=")
        if not separator or method not in overrides:
            raise ValueError(
                "--method-config expects METHOD=/path/to/config.{json,toml}"
            )
        _deep_merge(overrides[method], _load_mapping(Path(path_value).resolve()))
    for specification in args.method_param:
        left, separator, raw_value = specification.partition("=")
        method, dot, parameter = left.partition(".")
        if not separator or not dot or method not in overrides or not parameter:
            raise ValueError(
                "--method-param expects METHOD.PARAMETER=JSON_VALUE"
            )
        _set_dotted(overrides[method], parameter, _parse_value(raw_value))
    return overrides


def _method_and_settings(
    name: str,
    overrides: dict[str, dict[str, Any]],
):
    config_path = ROOT / "configs" / "methods" / f"{name}.toml"
    if config_path.exists():
        method, tracked = load_method_config(config_path)
        method_config, policy = method_run_settings(tracked)
    else:
        method = get_method(name)
        tracked = {"name": name, "status": "ready"}
        method_config = {}
        policy = FairnessPolicy()
    method_config = copy.deepcopy(method_config)
    _deep_merge(method_config, overrides[name])
    if name == "self_refine":
        max_refinements = int(method_config.get("max_refinements", 4))
        policy = replace(
            policy,
            min_model_calls_per_problem=2,
            max_model_calls_per_problem=1 + max_refinements,
        )
    elif name == "aflow":
        nodes = method_config["workflow"]["nodes"]
        policy = replace(
            policy,
            min_model_calls_per_problem=len(nodes),
            max_model_calls_per_problem=sum(
                node.get("max_attempts", 3)
                if node["operator"] == "programmer"
                else 1
                for node in nodes
            ),
        )
    return method, tracked, method_config, policy


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
    raise ValueError(f"Unknown model profile or config path: {value}")


def _load_locked_u_math_judge(lock_path: Path = U_MATH_JUDGE_LOCK):
    try:
        with lock_path.open("rb") as stream:
            lock = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ModelConfigurationError(
            f"Cannot read U-MATH judge lock {lock_path}: {exc}"
        ) from exc
    if lock.get("status") != "locked":
        raise ModelConfigurationError(
            "U-MATH judge is not locked yet. Evaluate candidate judges on "
            "µ-MATH, then set status='locked' and profile in "
            "configs/judges/u_math.lock.toml."
        )
    protocol = judge_protocol_metadata()
    expected_lock = {
        "protocol": protocol["name"],
        "protocol_version": protocol["version"],
        "verdict_extractor": "none",
        "inconclusive_treatment": protocol["inconclusive_treatment"],
    }
    mismatches = {
        key: {"expected": expected, "actual": lock.get(key)}
        for key, expected in expected_lock.items()
        if lock.get(key) != expected
    }
    if mismatches:
        raise ModelConfigurationError(
            "Locked U-MATH judge protocol does not match the runner: "
            f"{mismatches}"
        )
    profile_value = lock.get("profile")
    if not isinstance(profile_value, str) or not profile_value.strip():
        raise ModelConfigurationError(
            "Locked U-MATH judge must specify one profile path"
        )
    profile_path = (lock_path.parent / profile_value).resolve()
    try:
        profile_path.relative_to(lock_path.parent.resolve())
    except ValueError as exc:
        raise ModelConfigurationError(
            "U-MATH judge profile must stay inside configs/judges"
        ) from exc
    if not profile_path.is_file():
        raise ModelConfigurationError(
            f"Locked U-MATH judge profile not found: {profile_path}"
        )
    return load_model_profile(profile_path)


def _source_state(config_path: Path) -> dict[str, str] | None:
    if not config_path.is_file():
        return None
    try:
        with config_path.open("rb") as stream:
            tracked = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ExperimentArtifactError(
            f"Cannot read method repository configuration {config_path}: {exc}"
        ) from exc
    source = tracked.get("source_path")
    if not source:
        return None
    return clean_git_repository_state((config_path.parent / source).resolve())


def _count(plugin, context: DatasetContext) -> int:
    counter = getattr(plugin, "problem_count", None)
    if callable(counter):
        return int(counter(context))
    return sum(1 for _ in plugin.iter_problems(context))


def _sum_usage(target: dict[str, int | float], usage: dict[str, Any]) -> None:
    for key, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value


def _report_error(console, label: str, exc: Exception, *, debug: bool) -> None:
    console.print(f"[red]{label}:[/red] {type(exc).__name__}: {exc}")
    if debug:
        console.print_exception(show_locals=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable math benchmark experiment matrices"
    )
    parser.add_argument(
        "--model",
        default="qwen35_flash",
        help="Model profile name, model ID, or TOML path",
    )
    parser.add_argument("--model-id", help="Override model ID in the profile")
    parser.add_argument(
        "--methods",
        default="direct",
        help="Comma-separated methods or 'all'",
    )
    parser.add_argument(
        "--datasets",
        default="gsm1k",
        help="Comma-separated datasets or 'all'",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=_parse_batch_size,
        default="1",
        metavar="N|all",
        help="Run the next N unfinished samples per cell, or all remaining",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=1,
        help="Concurrent samples within each experiment cell (default: 1)",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Base model generation seed (0..2147483647); repeat handling "
            "is controlled by --seed-mode"
        ),
    )
    parser.add_argument(
        "--seed-mode",
        choices=("fixed", "increment"),
        default="fixed",
        help=(
            "Use the same seed for every repeat, or add repeat_index-1 "
            "(default: fixed)"
        ),
    )
    parser.add_argument(
        "--method-param",
        action="append",
        default=[],
        metavar="METHOD.KEY=VALUE",
    )
    parser.add_argument(
        "--method-config",
        action="append",
        default=[],
        metavar="METHOD=PATH",
    )
    parser.add_argument("--run-tag", default="")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT.parent / "math-benchmark-data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "experiments",
    )
    parser.add_argument(
        "--checker-python",
        type=Path,
        default=ROOT / ".venv-checkers" / "bin" / "python",
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full tracebacks in addition to durable error logs",
    )
    return parser.parse_args()


def main() -> int:
    try:
        from dotenv import load_dotenv
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
        from rich.prompt import Confirm
        from rich.table import Table
    except ImportError:
        print(
            "Experiment dependencies are missing. Run:\n"
            "  .venv-checkers/bin/pip install "
            "-r requirements/experiment-lock.txt\n"
            "  .venv-checkers/bin/pip install -e . --no-deps",
            file=sys.stderr,
        )
        return 2

    args = parse_args()
    console = Console()
    if args.repeats < 1:
        console.print("[red]--repeats must be at least 1[/red]")
        return 2
    load_dotenv(args.env_file, override=False)

    try:
        methods = _csv_selection(args.methods, ALL_METHODS, "methods")
        datasets = _dataset_selection(args.datasets)
        overrides = _method_overrides(args)
        model_path = _resolve_model_path(args.model)
        profile = load_model_profile(model_path)
        if args.model_id:
            profile = replace(profile, model=args.model_id)
        judge_profile = (
            _load_locked_u_math_judge()
            if "u-math-text-only" in datasets
            else None
        )
    except (ValueError, ModelConfigurationError) as exc:
        _report_error(console, "Configuration error", exc, debug=args.debug)
        return 2

    inference = profile.inference_defaults()
    for key, value in (
        ("temperature", args.temperature),
        ("top_p", args.top_p),
        ("max_output_tokens", args.max_output_tokens),
        ("seed", args.seed),
    ):
        if value is not None:
            inference[key] = value
    try:
        validate_inference_config(inference)
        repeat_inferences = {
            repeat_index: _repeat_inference_config(
                inference,
                repeat_index=repeat_index,
                seed_mode=args.seed_mode,
            )
            for repeat_index in range(1, args.repeats + 1)
        }
    except ModelConfigurationError as exc:
        _report_error(
            console,
            "Inference configuration error",
            exc,
            debug=args.debug,
        )
        return 2
    checker_python = args.checker_python.absolute()
    if not checker_python.is_file():
        console.print(f"[red]Checker Python not found:[/red] {checker_python}")
        return 2
    try:
        environment_state = python_environment_state(checker_python)
    except ExperimentArtifactError as exc:
        _report_error(
            console,
            "Environment validation failed",
            exc,
            debug=args.debug,
        )
        return 2

    try:
        data_root = args.data_root.resolve()
        data_state = clean_git_repository_state(data_root)
        method_source_states = {}
        for name in methods:
            config_path = ROOT / "configs" / "methods" / f"{name}.toml"
            method_source_states[name] = _source_state(config_path)
        dataset_paths = {
            name: _dataset_revision_paths(get_dataset(name))
            for name in datasets
        }
        dataset_artifacts = {
            name: _git_object_ids(
                data_root,
                revision=data_state["revision"],
                paths=dataset_paths[name],
            )
            for name in datasets
        }
    except ExperimentArtifactError as exc:
        _report_error(
            console,
            "Repository validation failed",
            exc,
            debug=args.debug,
        )
        return 2

    method_settings = {}
    try:
        for name in methods:
            method_settings[name] = _method_and_settings(name, overrides)
            _, _, method_config, policy = method_settings[name]
            evaluation_control_metadata(inference, method_config, policy)
    except (
        FairnessViolation,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        _report_error(
            console,
            "Method configuration error",
            exc,
            debug=args.debug,
        )
        return 2

    base_context = DatasetContext(
        args.data_root.resolve(),
        bridge_python={
            "math-perturb": str(checker_python),
            "harp": str(checker_python),
            "harp-small": str(checker_python),
            "mathconstruct": str(checker_python),
        },
    )
    dataset_totals = {}
    try:
        for name in datasets:
            plugin = get_dataset(name)
            dataset_totals[name] = _count(plugin, base_context)
    except Exception as exc:
        _report_error(
            console,
            "Dataset validation failed",
            exc,
            debug=args.debug,
        )
        return 2

    runner_tree = tree_fingerprint(ROOT)
    runner_git = git_revision(ROOT)
    output_root = args.output_root.resolve()
    cell_count = len(methods) * len(datasets) * args.repeats
    cell_plans: dict[tuple[int, str, str], dict[str, Any]] = {}
    sample_count = 0
    minimum_calls = 0
    maximum_calls = 0
    try:
        for repeat_index in range(1, args.repeats + 1):
            repeat_inference = repeat_inferences[repeat_index]
            for method_name in methods:
                _, _, method_config, policy = method_settings[method_name]
                method_source_state = method_source_states[method_name]
                for dataset_name in datasets:
                    plugin = get_dataset(dataset_name)
                    public_judge = (
                        {
                            **judge_protocol_metadata(),
                            "profile": judge_profile.public_dict(),
                            "model": judge_profile.model,
                            "inference_config": (
                                judge_profile.inference_defaults()
                            ),
                        }
                        if dataset_name == "u-math-text-only"
                        else None
                    )
                    configuration = {
                        "model": profile.public_dict(),
                        "inference_config": repeat_inference,
                        "repeat_seed": {
                            "mode": args.seed_mode,
                            "base_seed": inference.get("seed"),
                            "effective_seed": repeat_inference.get("seed"),
                        },
                        "method": method_name,
                        "method_config": method_config,
                        "fairness_policy": policy.to_dict(),
                        "dataset": dataset_name,
                        "dataset_scope": getattr(
                            plugin, "evaluation_scope", "test"
                        ),
                        "dataset_artifacts": dataset_artifacts[dataset_name],
                        "dataset_answer_instruction": getattr(
                            plugin, "answer_instruction", ""
                        ),
                        "repeat": repeat_index,
                        "run_tag": args.run_tag,
                        "judge": public_judge,
                        "environment": environment_state,
                        "revisions": {
                            "runner_git": runner_git,
                            "runner_tree": runner_tree,
                            "data": data_state,
                            "method_source": method_source_state,
                        },
                    }
                    identity = _experiment_identity(
                        configuration,
                        dataset_artifacts=dataset_artifacts[dataset_name],
                    )
                    components = (
                        profile.profile,
                        method_name,
                        dataset_name,
                        f"r{repeat_index:03d}",
                    )
                    if args.run_tag.strip():
                        components = (*components, args.run_tag)
                    resume = _find_resume_experiment(
                        output_root,
                        components=components,
                        identity=identity,
                        data_root=data_root,
                        dataset_paths=dataset_paths[dataset_name],
                    )
                    completed_count = (
                        len(_result_problem_ids(resume[0] / "records.jsonl"))
                        if resume is not None
                        else 0
                    )
                    remaining = max(
                        0,
                        dataset_totals[dataset_name] - completed_count,
                    )
                    target = (
                        remaining
                        if args.batch_size is None
                        else min(args.batch_size, remaining)
                    )
                    key = (repeat_index, method_name, dataset_name)
                    cell_plans[key] = {
                        "configuration": configuration,
                        "identity": identity,
                        "stable_fingerprint": configuration_fingerprint(
                            identity
                        ),
                        "components": components,
                        "resume": resume,
                        "public_judge": public_judge,
                    }
                    sample_count += target
                    minimum_calls += policy.min_model_calls_per_problem * target
                    maximum_calls += policy.max_model_calls_per_problem * target
                    if dataset_name == "u-math-text-only":
                        minimum_calls += target
                        maximum_calls += target
    except ExperimentArtifactError as exc:
        _report_error(
            console,
            "Resume validation failed",
            exc,
            debug=args.debug,
        )
        return 2

    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column()
    table.add_row("Model", f"{profile.profile} → {profile.model}")
    table.add_row("Endpoint", profile.base_url)
    table.add_row(
        "Python",
        (
            f"{environment_state['python_version']} · "
            f"{environment_state['executable']}"
        ),
    )
    table.add_row("Methods", ", ".join(methods))
    table.add_row("Datasets", ", ".join(datasets))
    if judge_profile is not None:
        table.add_row(
            "U-MATH judge",
            f"{judge_profile.profile} → {judge_profile.model} (fixed)",
        )
    table.add_row("Repeats", str(args.repeats))
    if "seed" not in inference:
        repeat_seed_description = "unset"
    elif args.seed_mode == "fixed":
        repeat_seed_description = f"{inference['seed']} (fixed)"
    elif args.repeats == 1:
        repeat_seed_description = f"{inference['seed']} (increment)"
    else:
        repeat_seed_description = (
            f"{inference['seed']}–"
            f"{repeat_inferences[args.repeats]['seed']} (increment)"
        )
    table.add_row("Repeat seeds", repeat_seed_description)
    table.add_row(
        "Batch size",
        "all remaining" if args.batch_size is None else str(args.batch_size),
    )
    table.add_row("Concurrency", str(args.concurrency))
    table.add_row(
        "Max new samples",
        f"{sample_count:,} across {cell_count} cells",
    )
    table.add_row(
        "Estimated API calls",
        f"{minimum_calls:,}"
        if minimum_calls == maximum_calls
        else f"{minimum_calls:,}–{maximum_calls:,}",
    )
    table.add_row("Inference", json.dumps(inference, ensure_ascii=False))
    table.add_row("Resume", "sample-level JSONL (enabled)")
    table.add_row("Output", str(args.output_root.resolve()))
    console.print(Panel(table, title="Math Benchmark Experiment Plan", border_style="blue"))

    if args.dry_run:
        console.print("[yellow]Dry run complete; no API calls were made.[/yellow]")
        return 0
    if not args.yes and not Confirm.ask("Start this experiment matrix?", default=False):
        console.print("Cancelled.")
        return 0

    try:
        if not args.skip_preflight:
            console.print("[cyan]Running API preflight…[/cyan]")
            preflight_backend = OpenAICompatibleBackend(
                profile,
                telemetry_path=(
                    args.output_root.resolve()
                    / f"_preflight__{safe_slug(profile.profile)}.jsonl"
                ),
            )
            preflight = preflight_backend.generate(
                [{"role": "user", "content": "Reply with exactly: OK"}],
                config={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_output_tokens": 16,
                    **(
                        {"seed": repeat_inferences[1]["seed"]}
                        if "seed" in repeat_inferences[1]
                        else {}
                    ),
                },
            )
            console.print(
                f"[green]API ready[/green] · {preflight.latency_seconds:.2f}s · "
                f"{preflight.usage.get('total_tokens', 0):,} tokens"
            )
            if judge_profile is not None:
                console.print("[cyan]Running fixed judge API preflight…[/cyan]")
                judge_preflight_backend = OpenAICompatibleBackend(
                    judge_profile,
                    telemetry_path=(
                        args.output_root.resolve()
                        / f"_preflight__{safe_slug(judge_profile.profile)}.jsonl"
                    ),
                )
                judge_preflight = judge_preflight_backend.generate(
                    [{"role": "user", "content": "Reply with exactly: OK"}],
                    config={
                        **judge_profile.inference_defaults(),
                        "max_output_tokens": 16,
                    },
                )
                console.print(
                    f"[green]Judge API ready[/green] · "
                    f"{judge_preflight.latency_seconds:.2f}s · "
                    f"{judge_preflight.usage.get('total_tokens', 0):,} tokens"
                )
    except Exception as exc:
        _report_error(console, "API preflight failed", exc, debug=args.debug)
        return 2

    overall_failed = 0
    matrix_started = time.perf_counter()
    result_rows: list[dict[str, Any]] = []

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[latency]}"),
        TextColumn("{task.fields[tokens]}"),
        console=console,
    )
    overall_task = progress.add_task(
        "Matrix",
        total=cell_count,
        latency="",
        tokens="",
    )

    try:
        with progress:
            for repeat_index in range(1, args.repeats + 1):
                repeat_inference = repeat_inferences[repeat_index]
                for method_name in methods:
                    method, _, method_config, policy = method_settings[
                        method_name
                    ]
                    for dataset_name in datasets:
                        plugin = get_dataset(dataset_name)
                        dataset_total = dataset_totals[dataset_name]
                        plan = cell_plans[
                            (repeat_index, method_name, dataset_name)
                        ]
                        configuration = plan["configuration"]
                        identity = plan["identity"]
                        stable_fingerprint = plan["stable_fingerprint"]
                        public_judge = plan["public_judge"]
                        started_at = datetime.now(timezone.utc).isoformat()
                        components = plan["components"]
                        resume = _find_resume_experiment(
                            output_root,
                            components=components,
                            identity=identity,
                            data_root=data_root,
                            dataset_paths=dataset_paths[dataset_name],
                        )
                        if resume is None:
                            fingerprint = stable_fingerprint
                            manifest_configuration = configuration
                            directory = experiment_directory(
                                output_root,
                                model=profile.profile,
                                method=method_name,
                                dataset=dataset_name,
                                repeat=repeat_index,
                                fingerprint=fingerprint,
                                created_at=started_at,
                                run_tag=args.run_tag,
                                claim=True,
                            )
                        else:
                            (
                                directory,
                                fingerprint,
                                manifest_configuration,
                            ) = resume
                        ensure_experiment_manifest(
                            directory / "experiment.json",
                            configuration=manifest_configuration,
                            fingerprint=fingerprint,
                            created_at=started_at,
                        )
                        api_path = directory / "api_calls.jsonl"
                        error_path = directory / "errors.jsonl"
                        backend = OpenAICompatibleBackend(
                            profile, telemetry_path=api_path
                        )
                        judge = (
                            ModelJudgeBackend(
                                OpenAICompatibleBackend(
                                    judge_profile,
                                    telemetry_path=api_path,
                                ),
                                model_name=judge_profile.model,
                                inference_config=public_judge[
                                    "inference_config"
                                ],
                            )
                            if public_judge
                            else None
                        )
                        context = DatasetContext(
                            args.data_root.resolve(),
                            bridge_python=base_context.bridge_python,
                            judge=judge,
                        )
                        store = JsonlResultStore(directory / "records.jsonl")
                        experiment = Experiment(
                            id=f"experiment::{fingerprint}",
                            dataset=dataset_name,
                            method=method_name,
                            model=profile.model,
                            revisions=manifest_configuration["revisions"],
                            metadata={
                                "fingerprint": fingerprint,
                                "repeat": repeat_index,
                                "run_tag": args.run_tag,
                                "judge": manifest_configuration["judge"],
                                "evaluation_control": (
                                    evaluation_control_metadata(
                                        repeat_inference,
                                        method_config,
                                        policy,
                                    )
                                ),
                            },
                        )
                        completed_before = store.completed_problem_ids(
                            experiment
                        )
                        remaining = max(
                            0,
                            dataset_total - len(completed_before),
                        )
                        invocation_target = (
                            remaining
                            if args.batch_size is None
                            else min(args.batch_size, remaining)
                        )
                        live_usage: dict[str, int | float] = {}
                        live_latencies: list[float] = []
                        sample_task = progress.add_task(
                            (
                                f"r{repeat_index} {method_name} × "
                                f"{dataset_name}"
                            ),
                            total=invocation_target,
                            latency="lat —",
                            tokens="tok 0",
                        )

                        def on_record(record) -> None:
                            generation = record.generation
                            if generation is not None:
                                _sum_usage(live_usage, generation.usage)
                            processing = record.timing.get(
                                "processing_seconds"
                            )
                            if processing is not None:
                                live_latencies.append(float(processing))
                            mean_latency = (
                                sum(live_latencies) / len(live_latencies)
                                if live_latencies
                                else 0.0
                            )
                            progress.update(
                                sample_task,
                                advance=1,
                                latency=f"lat {mean_latency:.2f}s",
                                tokens=(
                                    "tok "
                                    f"{int(live_usage.get('total_tokens', 0)):,}"
                                ),
                            )
                            if record.score.status.value == "error":
                                diagnostic = {
                                    "timestamp": datetime.now(
                                        timezone.utc
                                    ).isoformat(),
                                    "fingerprint": fingerprint,
                                    "model": profile.model,
                                    "method": method_name,
                                    "dataset": dataset_name,
                                    "repeat": repeat_index,
                                    "problem_id": record.problem.id,
                                    "error": record.score.error,
                                    "details": record.score.details,
                                    "timing": record.timing,
                                }
                                append_jsonl(error_path, diagnostic)
                                failure_stage = record.score.details.get(
                                    "failure_stage", "unknown"
                                )
                                progress.console.print(
                                    "[red]ERROR[/red] "
                                    f"{method_name} × {dataset_name} · "
                                    f"problem={record.problem.id} · "
                                    f"stage={failure_stage} · "
                                    f"{record.score.error}"
                                )
                                if args.debug:
                                    traceback_text = record.score.details.get(
                                        "traceback"
                                    )
                                    if traceback_text:
                                        progress.console.print(
                                            traceback_text,
                                            style="dim red",
                                            markup=False,
                                        )

                        cell_started = time.perf_counter()
                        run_summary = EvaluationRunner().run(
                            experiment=experiment,
                            plugin=plugin,
                            context=context,
                            method=method,
                            backend=backend,
                            store=store,
                            method_config=method_config,
                            inference_config=repeat_inference,
                            fairness_policy=policy,
                            batch_size=args.batch_size,
                            concurrency=args.concurrency,
                            record_callback=on_record,
                        )
                        aggregate = summarize_records(store.read())
                        api_summary = summarize_api_calls(api_path)
                        cell_elapsed = time.perf_counter() - cell_started
                        summary_document = {
                            "fingerprint": fingerprint,
                            "started_at": started_at,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "invocation_seconds": cell_elapsed,
                            "invocation": {
                                "batch_size": args.batch_size,
                                "concurrency": args.concurrency,
                                "completed_before": len(completed_before),
                                "target_new_samples": invocation_target,
                            },
                            "run": asdict(run_summary),
                            "aggregate": aggregate,
                            "api": api_summary,
                            "paths": {
                                "directory": str(directory),
                                "records": str(directory / "records.jsonl"),
                                "api_calls": str(api_path),
                                "errors": str(error_path),
                            },
                        }
                        atomic_write_json(
                            directory / "summary.json", summary_document
                        )
                        progress.update(
                            sample_task,
                            completed=invocation_target,
                        )
                        progress.remove_task(sample_task)
                        progress.advance(overall_task)
                        failed = aggregate["errors"]
                        overall_failed += int(failed)
                        result_rows.append(
                            {
                                "method": method_name,
                                "dataset": dataset_name,
                                "repeat": repeat_index,
                                "new": run_summary.attempted,
                                "records": aggregate["records"],
                                "accuracy": aggregate["accuracy"],
                                "errors": failed,
                                "tokens": api_summary["usage"].get(
                                    "total_tokens", 0
                                ),
                                "seconds": cell_elapsed,
                                "resumed": run_summary.resumed,
                                "path": str(directory),
                            }
                        )
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted. Completed samples are durable and will resume "
            "on the next identical command.[/yellow]"
        )
        return 130
    except SampleEvaluationError as exc:
        console.print(
            "\n[red]Experiment stopped at the first failed sample.[/red] "
            f"No result was recorded for {exc.record.problem.id}; rerun the "
            "same command after fixing the cause."
        )
        return 1
    except Exception as exc:
        fatal_path = output_root / "_fatal_errors.jsonl"
        append_jsonl(
            fatal_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        console.print(
            f"\n[red]Fatal experiment error:[/red] "
            f"{type(exc).__name__}: {exc}"
        )
        console.print(f"Diagnostic log: {fatal_path}")
        if args.debug:
            console.print_exception(show_locals=False)
        return 2

    result_table = Table(title="Experiment Matrix Summary")
    for column in (
        "Method",
        "Dataset",
        "Rep",
        "New",
        "Total",
        "Accuracy",
        "Errors",
        "Tokens",
        "Time",
        "Resumed",
    ):
        result_table.add_column(column)
    for row in result_rows:
        accuracy = (
            f"{row['accuracy']:.3%}" if row["accuracy"] is not None else "—"
        )
        result_table.add_row(
            row["method"],
            row["dataset"],
            str(row["repeat"]),
            str(row["new"]),
            str(row["records"]),
            accuracy,
            str(row["errors"]),
            f"{int(row['tokens']):,}",
            f"{row['seconds']:.1f}s",
            str(row["resumed"]),
        )
    console.print(result_table)
    console.print(
        f"Completed in {time.perf_counter() - matrix_started:.1f}s · "
        f"{len(result_rows)} cells · {overall_failed} sample errors"
    )
    console.print(f"Results root: [link={output_root}]{output_root}[/link]")
    return 1 if overall_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
