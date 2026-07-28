from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from typing import Any

from benchmark_core.fairness import FairnessPolicy
from benchmark_core.interfaces import EvaluationMethod


def _resolve(symbol: str) -> type:
    module_name, separator, attribute = symbol.rpartition(".")
    if not separator:
        raise ValueError(f"Expected dotted Python symbol, got {symbol!r}")
    module = importlib.import_module(module_name)
    resolved = getattr(module, attribute)
    if not isinstance(resolved, type):
        raise TypeError(f"Configured symbol is not a class: {symbol}")
    return resolved


def load_method_config(path: Path) -> tuple[EvaluationMethod, dict[str, Any]]:
    """Instantiate a baseline or paper adapter from its tracked TOML config."""

    path = path.resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    symbol = config.get("adapter") or config.get("implementation")
    if not isinstance(symbol, str):
        raise ValueError(f"Method config has no adapter/implementation: {path}")
    method_class = _resolve(symbol)
    if "source_path" in config:
        source_path = (path.parent / config["source_path"]).resolve()
        method = method_class(source_path)
    else:
        method = method_class()
    if method.name != config.get("name"):
        raise ValueError(
            f"Method config name {config.get('name')!r} != implementation {method.name!r}"
        )
    return method, config


def method_run_settings(
    config: dict[str, Any],
) -> tuple[dict[str, Any], FairnessPolicy]:
    """Translate tracked TOML sections into EvaluationRunner arguments."""

    name = config.get("name")
    if name == "aflow":
        method_config = {
            "workflow": config["workflow"],
            "workflow_artifact": config["workflow_artifact"],
        }
    else:
        method_config = dict(config.get("method", {}))
    fairness = dict(config.get("fairness", {}))
    if "allowed_inference_overrides" in fairness:
        fairness["allowed_inference_overrides"] = tuple(
            fairness["allowed_inference_overrides"]
        )
    policy = FairnessPolicy(**fairness)
    expected_bounds: tuple[int, int] | None = None
    if name == "pal":
        expected_bounds = (1, 1)
    elif name == "self_refine":
        expected_bounds = (2, 1 + int(method_config.get("max_refinements", 4)))
    elif name == "aflow":
        nodes = method_config["workflow"]["nodes"]
        minimum = len(nodes)
        maximum = sum(
            node.get("max_attempts", 3)
            if node["operator"] == "programmer"
            else 1
            for node in nodes
        )
        expected_bounds = (minimum, maximum)
    if expected_bounds is not None and (
        policy.min_model_calls_per_problem,
        policy.max_model_calls_per_problem,
    ) != expected_bounds:
        raise ValueError(
            f"{name} fairness call bounds do not match method config: "
            f"expected {expected_bounds}, got "
            f"{(policy.min_model_calls_per_problem, policy.max_model_calls_per_problem)}"
        )
    return method_config, policy
