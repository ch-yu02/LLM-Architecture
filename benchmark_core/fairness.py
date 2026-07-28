from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

from .interfaces import ModelBackend
from .schema import Generation


class FairnessViolation(ValueError):
    pass


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return (
        normalized in {"api_key", "apikey", "password", "secret", "token"}
        or normalized.endswith(("_api_key", "_password", "_secret", "_token"))
    )


def _sensitive_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if _is_sensitive_key(str(key)):
                found.append(path)
            found.extend(_sensitive_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_sensitive_paths(child, f"{prefix}[{index}]"))
    return found


def _safe_json_config(config: dict[str, Any]) -> dict[str, Any]:
    sensitive = _sensitive_paths(config)
    if sensitive:
        raise FairnessViolation(
            f"Inference config contains sensitive keys that must not enter results: {sensitive}"
        )
    try:
        return json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise FairnessViolation(f"Inference config must be JSON-serializable: {exc}") from exc


@dataclass(frozen=True)
class FairnessPolicy:
    """Per-problem controls shared by methods in the same comparison."""

    min_model_calls_per_problem: int = 1
    max_model_calls_per_problem: int = 1
    allowed_inference_overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.min_model_calls_per_problem < 0:
            raise ValueError("min_model_calls_per_problem must be non-negative")
        if self.max_model_calls_per_problem < self.min_model_calls_per_problem:
            raise ValueError("max_model_calls_per_problem must be >= minimum")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_inference_overrides"] = list(
            self.allowed_inference_overrides
        )
        return value


class ControlledModelBackend:
    """Enforces call budget and a fixed inference configuration."""

    def __init__(
        self,
        backend: ModelBackend,
        *,
        inference_config: dict[str, Any],
        policy: FairnessPolicy,
    ):
        self._backend = backend
        self.inference_config = _safe_json_config(inference_config)
        self.policy = policy
        self.model_calls = 0
        self._generations: list[Generation] = []
        self._resolved_call_configs: list[dict[str, Any]] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        config: dict[str, Any],
    ) -> Generation:
        disallowed = set(config) - set(self.policy.allowed_inference_overrides)
        if disallowed:
            raise FairnessViolation(
                f"Method attempted undeclared inference overrides: {sorted(disallowed)}"
            )
        if self.model_calls >= self.policy.max_model_calls_per_problem:
            raise FairnessViolation(
                "Method exceeded max_model_calls_per_problem="
                f"{self.policy.max_model_calls_per_problem}"
            )
        resolved = {**self.inference_config, **config}
        self.model_calls += 1
        self._resolved_call_configs.append(resolved)
        generation = self._backend.generate(messages, config=resolved)
        self._generations.append(generation)
        return generation

    def finalize(self, generation: Generation) -> Generation:
        if self.model_calls < self.policy.min_model_calls_per_problem:
            raise FairnessViolation(
                "Method used fewer than min_model_calls_per_problem="
                f"{self.policy.min_model_calls_per_problem}"
            )
        metadata = dict(generation.metadata)
        metadata["evaluation_control"] = {
            "model_calls": self.model_calls,
            "inference_config": self.inference_config,
            "resolved_call_configs": self._resolved_call_configs,
            "fairness_policy": self.policy.to_dict(),
            "calls": [
                {
                    "finish_reason": item.finish_reason,
                    "latency_seconds": item.latency_seconds,
                    "usage": item.usage,
                }
                for item in self._generations
            ],
        }
        aggregate_usage: dict[str, Any] = {}
        for item in self._generations:
            for key, value in item.usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    aggregate_usage[key] = aggregate_usage.get(key, 0) + value
                else:
                    aggregate_usage[key] = value
        aggregate_latency = [
            item.latency_seconds
            for item in self._generations
            if item.latency_seconds is not None
        ]
        return replace(
            generation,
            latency_seconds=sum(aggregate_latency) if aggregate_latency else None,
            usage=aggregate_usage,
            metadata=metadata,
        )


def evaluation_control_metadata(
    inference_config: dict[str, Any],
    method_config: dict[str, Any],
    policy: FairnessPolicy,
) -> dict[str, Any]:
    return {
        "inference_config": _safe_json_config(inference_config),
        "method_config": _safe_json_config(method_config),
        "fairness_policy": policy.to_dict(),
    }
