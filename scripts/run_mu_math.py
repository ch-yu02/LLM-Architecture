#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_core.concurrency import ordered_parallel_map  # noqa: E402
from benchmark_core.schema import Problem  # noqa: E402
from benchmark_experiments.artifacts import (  # noqa: E402
    ExperimentArtifactError,
    append_jsonl,
    atomic_write_json,
    clean_git_repository_state,
    configuration_fingerprint,
    ensure_experiment_manifest,
    git_revision,
    python_environment_state,
    summarize_api_calls,
    timestamped_experiment_directory,
    tree_fingerprint,
)
from model_backends import (  # noqa: E402
    ModelConfigurationError,
    OpenAICompatibleBackend,
    load_model_profile,
)
from scorers.model_judge import (  # noqa: E402
    ModelJudgeBackend,
    judge_protocol_metadata,
)


FORMAT_VERSION = 1
DATASET_NAME = "mu-math"
DEFAULT_DATA_ROOT = ROOT.parent / "math-benchmark-data"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "mu_math"
DEFAULT_JUDGE = "qwen35_flash"


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


def _resolve_judge_path(value: str) -> Path:
    provided = Path(value)
    if provided.is_file():
        return provided.resolve()
    candidates = ROOT / "configs" / "judges" / "candidates"
    named = candidates / f"{value}.toml"
    if named.is_file():
        return named
    for path in candidates.glob("*.toml"):
        profile = load_model_profile(path)
        if profile.profile == value or profile.model == value:
            return path
    raise ModelConfigurationError(
        f"Unknown judge candidate profile or TOML path: {value}"
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"µ-MATH data not found: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row_id = str(row["id"])
                metadata = row["metadata"]
                if (
                    row_id in seen
                    or row["answer"] not in {"true", "false"}
                    or not isinstance(metadata["label"], bool)
                    or metadata["label"] != (row["answer"] == "true")
                    or not isinstance(metadata["golden_answer"], str)
                    or not isinstance(metadata["model_output"], str)
                    or not isinstance(metadata["solution_model"], str)
                ):
                    raise ValueError("invalid µ-MATH row structure")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid µ-MATH data at {path}:{line_number}: {exc}"
                ) from exc
            seen.add(row_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"µ-MATH data is empty: {path}")
    return rows


def _read_records(
    path: Path,
    *,
    fingerprint: str,
) -> tuple[set[str], list[dict[str, Any]]]:
    if not path.exists():
        return set(), []
    completed: set[str] = set()
    samples: list[dict[str, Any]] = []
    header_seen = False
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if item.get("record_type") == "experiment":
                    if (
                        header_seen
                        or samples
                        or item.get("format_version") != FORMAT_VERSION
                        or item.get("fingerprint") != fingerprint
                    ):
                        raise ValueError("experiment header mismatch")
                    header_seen = True
                    continue
                if item.get("record_type") != "sample" or not header_seen:
                    raise ValueError("sample appears before a valid header")
                row_id = str(item["id"])
                if row_id in completed:
                    raise ValueError(f"duplicate sample ID {row_id}")
                completed.add(row_id)
                samples.append(item)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid result at {path}:{line_number}: {exc}"
                ) from exc
    if not header_seen:
        raise ValueError(f"Result file has no experiment header: {path}")
    return completed, samples


def _append_record(
    path: Path,
    *,
    fingerprint: str,
    record: dict[str, Any],
) -> None:
    if not path.exists() or path.stat().st_size == 0:
        append_jsonl(
            path,
            {
                "record_type": "experiment",
                "format_version": FORMAT_VERSION,
                "dataset": DATASET_NAME,
                "fingerprint": fingerprint,
            },
        )
    append_jsonl(path, record)


@contextmanager
def _exclusive_run(records_path: Path) -> Iterator[None]:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = records_path.with_name(f".{records_path.name}.lock")
    with lock_path.open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExperimentArtifactError(
                f"Another process is already running this experiment: "
                f"{records_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        record for record in records if record.get("status") == "scored"
    ]
    errors = sum(record.get("status") == "error" for record in records)
    tp = tn = fp = fn = 0
    positive_inconclusive = negative_inconclusive = 0
    for record in successful:
        label = bool(record["label"])
        verdict = record["judge"]["verdict"]
        if verdict == "Yes":
            if label:
                tp += 1
            else:
                fp += 1
        elif verdict == "No":
            if label:
                fn += 1
            else:
                tn += 1
        elif verdict == "Inconclusive":
            if label:
                positive_inconclusive += 1
            else:
                negative_inconclusive += 1
        else:
            raise ValueError(f"Unknown judge verdict in result: {verdict}")

    positives = tp + fn + positive_inconclusive
    negatives = tn + fp + negative_inconclusive
    tpr = _ratio(tp, positives)
    tnr = _ratio(tn, negatives)
    ppv = _ratio(tp, tp + fp)
    npv = _ratio(tn, tn + fn)
    positive_f1 = _ratio(2 * ppv * tpr, ppv + tpr)
    negative_f1 = _ratio(2 * npv * tnr, npv + tnr)
    correct = tp + tn
    metric_ratios = {
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "positive_f1": positive_f1,
        "negative_f1": negative_f1,
        "accuracy": _ratio(correct, len(successful)),
        "tpr": tpr,
        "tnr": tnr,
        "ppv": ppv,
        "npv": npv,
    }
    return {
        "records": len(records),
        "scored": len(successful),
        "errors": errors,
        "correct": correct,
        "inconclusive": (
            positive_inconclusive + negative_inconclusive
        ),
        "labels": {
            "positive": positives,
            "negative": negatives,
        },
        "confusion": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "positive_inconclusive": positive_inconclusive,
            "negative_inconclusive": negative_inconclusive,
        },
        "metrics": metric_ratios,
        "metrics_percent": {
            key: round(value * 100, 4)
            for key, value in metric_ratios.items()
        },
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_models = sorted(
        {
            str(record["candidate"]["solution_model"])
            for record in records
        }
    )
    return {
        "aggregate": _metrics(records),
        "by_solution_model": {
            source_model: _metrics(
                [
                    record
                    for record in records
                    if record["candidate"]["solution_model"] == source_model
                ]
            )
            for source_model in source_models
        },
    }


def _report_error(console, label: str, exc: Exception, *, debug: bool) -> None:
    console.print(f"[red]{label}:[/red] {type(exc).__name__}: {exc}")
    if debug:
        console.print_exception(show_locals=False)


def _evaluate_row(
    row: dict[str, Any],
    *,
    judge,
    fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    row_id = str(row["id"])
    metadata = row["metadata"]
    sample_started = time.perf_counter()
    try:
        problem = Problem(
            dataset=DATASET_NAME,
            id=row_id,
            prompt=row["problem"],
            reference_answer=metadata["golden_answer"],
            metadata={
                "source_uuid": metadata["source_uuid"],
                "source_row_idx": metadata["source_row_idx"],
            },
        )
        decision = judge.judge(problem, metadata["model_output"])
        predicted_label = {
            "Yes": True,
            "No": False,
            "Inconclusive": None,
        }[decision.verdict.value]
        label = bool(metadata["label"])
        correct = (
            predicted_label is not None and predicted_label == label
        )
        usage = decision.metadata.get("usage", {})
        latency = decision.metadata.get("latency_seconds")
        return (
            {
                "record_type": "sample",
                "status": "scored",
                "id": row_id,
                "problem": row["problem"],
                "reference_answer": metadata["golden_answer"],
                "candidate": {
                    "solution_model": metadata["solution_model"],
                    "text": metadata["model_output"],
                },
                "metadata": {
                    "source_uuid": metadata["source_uuid"],
                    "source_row_idx": metadata["source_row_idx"],
                },
                "label": label,
                "judge": {
                    "verdict": decision.verdict.value,
                    "predicted_label": predicted_label,
                    "correct": correct,
                    "rationale": decision.rationale,
                    "parse_status": decision.metadata[
                        "verdict_parse_status"
                    ],
                    "usage": usage,
                    "latency_seconds": latency,
                },
                "processing_seconds": (
                    time.perf_counter() - sample_started
                ),
            },
            None,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - sample_started
        diagnostic = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "id": row_id,
            "solution_model": metadata.get("solution_model"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "processing_seconds": elapsed,
        }
        return (
            {
                "record_type": "sample",
                "status": "error",
                "id": row_id,
                "problem": row["problem"],
                "reference_answer": metadata["golden_answer"],
                "candidate": {
                    "solution_model": metadata["solution_model"],
                    "text": metadata["model_output"],
                },
                "metadata": {
                    "source_uuid": metadata["source_uuid"],
                    "source_row_idx": metadata["source_row_idx"],
                },
                "label": bool(metadata["label"]),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "processing_seconds": elapsed,
            },
            diagnostic,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one fixed judge candidate on official µ-MATH"
    )
    parser.add_argument(
        "--judge",
        default=DEFAULT_JUDGE,
        help="Candidate profile name, model ID, or TOML path",
    )
    parser.add_argument(
        "--batch-size",
        type=_parse_batch_size,
        default="1",
        metavar="N|all",
        help="Run the next N unfinished rows, or all remaining",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=1,
        help="Concurrent judge requests (default: 1)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
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
    load_dotenv(args.env_file, override=False)

    try:
        judge_path = _resolve_judge_path(args.judge)
        profile = load_model_profile(judge_path)
        inference = profile.inference_defaults()
        data_root = args.data_root.resolve()
        data_state = clean_git_repository_state(data_root)
        data_path = data_root / "data" / "processed" / "mu_math.jsonl"
        rows = _load_rows(data_path)
        environment = python_environment_state(Path(sys.executable))
    except (
        ExperimentArtifactError,
        FileNotFoundError,
        ModelConfigurationError,
        ValueError,
    ) as exc:
        _report_error(console, "Configuration error", exc, debug=args.debug)
        return 2

    configuration = {
        "dataset": DATASET_NAME,
        "dataset_scope": "official-test",
        "judge_profile": profile.public_dict(),
        "inference_config": inference,
        "protocol": judge_protocol_metadata(),
        "environment": environment,
        "revisions": {
            "runner_git": git_revision(ROOT),
            "runner_tree": tree_fingerprint(ROOT),
            "data": data_state,
        },
    }
    fingerprint = configuration_fingerprint(configuration)
    output_root = args.output_root.resolve()
    started_at = datetime.now(timezone.utc).isoformat()
    directory = timestamped_experiment_directory(
        output_root,
        components=(profile.profile, "mu-math"),
        fingerprint=fingerprint,
        created_at=started_at,
    )
    records_path = directory / "records.jsonl"
    api_path = directory / "api_calls.jsonl"
    errors_path = directory / "errors.jsonl"
    try:
        completed, existing_records = _read_records(
            records_path,
            fingerprint=fingerprint,
        )
    except ValueError as exc:
        _report_error(console, "Resume validation failed", exc, debug=args.debug)
        return 2
    unknown_completed = completed - {str(row["id"]) for row in rows}
    if unknown_completed:
        console.print(
            "[red]Resume validation failed:[/red] result contains IDs absent "
            f"from current µ-MATH data: {sorted(unknown_completed)[:5]}"
        )
        return 2
    remaining = [row for row in rows if str(row["id"]) not in completed]
    target = (
        len(remaining)
        if args.batch_size is None
        else min(args.batch_size, len(remaining))
    )

    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column()
    table.add_row("Judge", f"{profile.profile} → {profile.model}")
    table.add_row("Endpoint", profile.base_url)
    table.add_row("Protocol", judge_protocol_metadata()["name"])
    table.add_row("Rows", f"{len(rows):,} official test rows")
    table.add_row("Completed", f"{len(completed):,}")
    table.add_row("Next batch", f"{target:,}")
    table.add_row("Concurrency", str(args.concurrency))
    table.add_row("Remaining after", f"{len(remaining) - target:,}")
    table.add_row("Inference", json.dumps(inference, ensure_ascii=False))
    table.add_row("Output", str(directory))
    console.print(
        Panel(table, title="µ-MATH Judge Evaluation", border_style="blue")
    )

    if args.dry_run:
        console.print(
            "[yellow]Dry run complete; no files or API calls were made.[/yellow]"
        )
        return 0
    if target == 0:
        aggregate = _summarize(existing_records)["aggregate"]
        if aggregate["errors"]:
            console.print(
                "[red]µ-MATH evaluation has no unfinished rows, but contains "
                f"{aggregate['errors']} recorded errors.[/red]"
            )
            return 1
        console.print("[green]µ-MATH evaluation is already complete.[/green]")
        return 0
    if not args.yes and not Confirm.ask(
        "Start this µ-MATH judge evaluation?",
        default=False,
    ):
        console.print("Cancelled.")
        return 0

    try:
        if not args.skip_preflight:
            console.print("[cyan]Running judge API preflight…[/cyan]")
            preflight_backend = OpenAICompatibleBackend(profile)
            preflight = preflight_backend.generate(
                [{"role": "user", "content": "Reply with exactly: OK"}],
                config={
                    **inference,
                    "max_output_tokens": 16,
                },
            )
            console.print(
                f"[green]Judge API ready[/green] · "
                f"{preflight.latency_seconds:.2f}s · "
                f"{preflight.usage.get('total_tokens', 0):,} tokens"
            )
    except Exception as exc:
        _report_error(console, "API preflight failed", exc, debug=args.debug)
        return 2

    directory = timestamped_experiment_directory(
        output_root,
        components=(profile.profile, "mu-math"),
        fingerprint=fingerprint,
        created_at=started_at,
        claim=True,
    )
    records_path = directory / "records.jsonl"
    api_path = directory / "api_calls.jsonl"
    errors_path = directory / "errors.jsonl"
    try:
        ensure_experiment_manifest(
            directory / "experiment.json",
            configuration=configuration,
            fingerprint=fingerprint,
            created_at=started_at,
        )
    except ExperimentArtifactError as exc:
        _report_error(
            console,
            "Experiment manifest conflict",
            exc,
            debug=args.debug,
        )
        return 2

    backend = OpenAICompatibleBackend(profile, telemetry_path=api_path)
    judge = ModelJudgeBackend(
        backend,
        model_name=profile.model,
        inference_config=inference,
    )
    live_tokens = 0
    live_latencies: list[float] = []
    new_records: list[dict[str, Any]] = []
    invocation_started = time.perf_counter()
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
    task = progress.add_task(
        "µ-MATH",
        total=target,
        latency="lat —",
        tokens="tok 0",
    )

    try:
        with _exclusive_run(records_path), progress:
            locked_completed, _ = _read_records(
                records_path,
                fingerprint=fingerprint,
            )
            selected = [
                row
                for row in rows
                if str(row["id"]) not in locked_completed
            ][:target]

            def evaluate(row):
                return _evaluate_row(
                    row,
                    judge=judge,
                    fingerprint=fingerprint,
                )

            evaluated = ordered_parallel_map(
                evaluate,
                selected,
                concurrency=args.concurrency,
                thread_name_prefix="mu-math-judge",
            )
            for record, diagnostic in evaluated:
                if diagnostic is not None:
                    row_id = record["id"]
                    error = record["error"]
                    append_jsonl(errors_path, diagnostic)
                    progress.console.print(
                        f"[red]ERROR[/red] µ-MATH · id={row_id} · "
                        f"{error['type']}: {error['message']}"
                    )
                    if args.debug:
                        progress.console.print(
                            diagnostic["traceback"],
                            style="dim red",
                            markup=False,
                        )
                else:
                    usage = record["judge"]["usage"]
                    latency = record["judge"]["latency_seconds"]
                    if isinstance(
                        usage.get("total_tokens"),
                        (int, float),
                    ):
                        live_tokens += int(usage["total_tokens"])
                    if isinstance(latency, (int, float)):
                        live_latencies.append(float(latency))
                _append_record(
                    records_path,
                    fingerprint=fingerprint,
                    record=record,
                )
                new_records.append(record)
                mean_latency = (
                    sum(live_latencies) / len(live_latencies)
                    if live_latencies
                    else 0.0
                )
                progress.update(
                    task,
                    advance=1,
                    latency=f"lat {mean_latency:.2f}s",
                    tokens=f"tok {live_tokens:,}",
                )
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted. Completed rows are durable and will resume "
            "on the next identical command.[/yellow]"
        )
        return 130
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
        _report_error(console, "Fatal evaluation error", exc, debug=args.debug)
        console.print(f"Diagnostic log: {fatal_path}")
        return 2

    _, all_records = _read_records(
        records_path,
        fingerprint=fingerprint,
    )
    metric_summary = _summarize(all_records)
    api_summary = summarize_api_calls(api_path)
    aggregate = metric_summary["aggregate"]
    summary = {
        "fingerprint": fingerprint,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "invocation": {
            "batch_size": args.batch_size,
            "concurrency": args.concurrency,
            "completed_before": len(completed),
            "new_records": len(new_records),
            "seconds": time.perf_counter() - invocation_started,
        },
        **metric_summary,
        "api": api_summary,
        "paths": {
            "directory": str(directory),
            "records": str(records_path),
            "api_calls": str(api_path),
            "errors": str(errors_path),
        },
    }
    atomic_write_json(directory / "summary.json", summary)

    result = Table(title="µ-MATH Judge Result")
    result.add_column("Scope")
    result.add_column("Rows", justify="right")
    result.add_column("Macro-F1", justify="right")
    result.add_column("TPR", justify="right")
    result.add_column("TNR", justify="right")
    result.add_column("PPV", justify="right")
    result.add_column("NPV", justify="right")
    result.add_column("Inconclusive", justify="right")

    def add_metric_row(scope: str, metrics: dict[str, Any]) -> None:
        percent = metrics["metrics_percent"]
        result.add_row(
            scope,
            str(metrics["scored"]),
            f"{percent['macro_f1']:.2f}",
            f"{percent['tpr']:.2f}",
            f"{percent['tnr']:.2f}",
            f"{percent['ppv']:.2f}",
            f"{percent['npv']:.2f}",
            str(metrics["inconclusive"]),
        )

    add_metric_row("all", aggregate)
    for source_model, metrics in metric_summary[
        "by_solution_model"
    ].items():
        add_metric_row(source_model, metrics)
    console.print(result)
    console.print(f"Artifacts: {directory}")
    return 1 if aggregate["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
