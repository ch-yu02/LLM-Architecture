from __future__ import annotations

import json
import os
import random
import threading
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from benchmark_core.schema import Generation


class ModelConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ModelProfile:
    profile: str
    provider: str
    api_type: str
    model: str
    base_url: str
    api_key_env: str
    enable_thinking: bool | None = None
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    max_output_tokens: int | None = 8192
    seed: int | None = None
    request_timeout_seconds: float = 600
    max_retries: int = 5
    min_request_interval_seconds: float = 1.0

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)

    def inference_defaults(self) -> dict[str, Any]:
        values = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "seed": self.seed,
        }
        return {key: value for key, value in values.items() if value is not None}


def load_model_profile(path: Path) -> ModelProfile:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if config.get("api_type") != "openai":
        raise ModelConfigurationError("Only OpenAI-compatible profiles are supported")
    try:
        profile = ModelProfile(**config)
    except TypeError as exc:
        raise ModelConfigurationError(f"Invalid model profile {path}: {exc}") from exc
    if not profile.base_url.startswith(("https://", "http://")):
        raise ModelConfigurationError("Model base_url must use http(s)")
    if profile.max_retries < 1:
        raise ModelConfigurationError("max_retries must be at least 1")
    if profile.request_timeout_seconds <= 0:
        raise ModelConfigurationError("request_timeout_seconds must be positive")
    if profile.min_request_interval_seconds < 0:
        raise ModelConfigurationError(
            "min_request_interval_seconds must be non-negative"
        )
    validate_inference_config(profile.inference_defaults())
    return profile


def validate_inference_config(config: dict[str, Any]) -> None:
    supported = {
        "temperature",
        "top_p",
        "max_output_tokens",
        "seed",
        "stop",
    }
    unknown = set(config) - supported
    if unknown:
        raise ModelConfigurationError(
            f"Unsupported inference parameters: {sorted(unknown)}"
        )
    temperature = config.get("temperature")
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= temperature <= 2
    ):
        raise ModelConfigurationError("temperature must be between 0 and 2")
    top_p = config.get("top_p")
    if top_p is not None and (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not 0 <= top_p <= 1
    ):
        raise ModelConfigurationError("top_p must be between 0 and 1")
    max_output_tokens = config.get("max_output_tokens")
    if max_output_tokens is not None and (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 1
    ):
        raise ModelConfigurationError(
            "max_output_tokens must be a positive integer"
        )
    seed = config.get("seed")
    if seed is not None and (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**31 - 1
    ):
        raise ModelConfigurationError(
            "seed must be an integer between 0 and 2147483647"
        )
    stop = config.get("stop")
    if stop is not None and not (
        isinstance(stop, str)
        or (
            isinstance(stop, list)
            and all(isinstance(item, str) for item in stop)
        )
    ):
        raise ModelConfigurationError("stop must be a string or list of strings")


class JsonlApiTelemetry:
    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        durable = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(durable, ensure_ascii=False) + "\n")
                stream.flush()


class OpenAICompatibleBackend:
    """Synchronous backend with retry, pacing, token and latency telemetry."""

    def __init__(
        self,
        profile: ModelProfile,
        *,
        telemetry_path: Path | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelConfigurationError(
                "openai is not installed; install the experiment extra"
            ) from exc
        api_key = os.environ.get(profile.api_key_env)
        if not api_key:
            raise ModelConfigurationError(
                f"Required environment variable {profile.api_key_env!r} is not set"
            )
        self.profile = profile
        self._api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url=profile.base_url,
            timeout=profile.request_timeout_seconds,
            max_retries=0,
        )
        self.telemetry = JsonlApiTelemetry(telemetry_path)
        self._pace_lock = threading.Lock()
        self._last_request_started = 0.0

    def _pace(self) -> float:
        with self._pace_lock:
            elapsed = time.monotonic() - self._last_request_started
            delay = max(
                0.0,
                self.profile.min_request_interval_seconds - elapsed,
            )
            if delay:
                time.sleep(delay)
            self._last_request_started = time.monotonic()
            return delay

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
        except ImportError:
            return False
        return isinstance(
            exc,
            (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            ),
        )

    def _request_options(self, config: dict[str, Any]) -> dict[str, Any]:
        validate_inference_config(config)
        options: dict[str, Any] = {}
        if config.get("temperature") is not None:
            options["temperature"] = config["temperature"]
        if config.get("top_p") is not None:
            options["top_p"] = config["top_p"]
        if config.get("max_output_tokens") is not None:
            options["max_tokens"] = config["max_output_tokens"]
        if config.get("seed") is not None:
            options["seed"] = config["seed"]
        if config.get("stop") is not None:
            options["stop"] = config["stop"]
        if self.profile.enable_thinking is not None:
            options["extra_body"] = {
                "enable_thinking": self.profile.enable_thinking
            }
        return options

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        config: dict[str, Any],
    ) -> Generation:
        options = self._request_options(config)
        total_started = time.perf_counter()
        total_retry_wait = 0.0
        last_error: Exception | None = None

        for attempt in range(1, self.profile.max_retries + 1):
            pacing_wait = self._pace()
            request_started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.profile.model,
                    messages=list(messages),
                    **options,
                )
            except Exception as exc:
                request_latency = time.perf_counter() - request_started
                retryable = self._retryable(exc)
                status_code = getattr(exc, "status_code", None)
                self.telemetry.append(
                    {
                        "profile": self.profile.profile,
                        "model": self.profile.model,
                        "status": "error",
                        "attempt": attempt,
                        "latency_seconds": request_latency,
                        "pacing_wait_seconds": pacing_wait,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc).replace(
                            self._api_key,
                            "[REDACTED]",
                        )[:4000],
                        "status_code": status_code,
                        "retryable": retryable,
                        **(
                            {"seed": options["seed"]}
                            if "seed" in options
                            else {}
                        ),
                    }
                )
                last_error = exc
                if not retryable or attempt == self.profile.max_retries:
                    raise
                retry_wait = min(60.0, 2.0 ** attempt) + random.random()
                total_retry_wait += retry_wait
                time.sleep(retry_wait)
                continue

            request_latency = time.perf_counter() - request_started
            usage = response.usage
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(
                getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
                or prompt_tokens + completion_tokens
            )
            completion_details = getattr(
                usage, "completion_tokens_details", None
            )
            reasoning_tokens = int(
                getattr(completion_details, "reasoning_tokens", 0) or 0
            )
            cached_details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = int(
                getattr(cached_details, "cached_tokens", 0) or 0
            )
            request_id = getattr(response, "id", None)
            finish_reason = response.choices[0].finish_reason or "unknown"
            text = response.choices[0].message.content or ""
            durable_usage = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "reasoning_tokens": reasoning_tokens,
                "cached_input_tokens": cached_tokens,
            }
            self.telemetry.append(
                {
                    "profile": self.profile.profile,
                    "model": self.profile.model,
                    "status": "success",
                    "attempt": attempt,
                    "latency_seconds": request_latency,
                    "pacing_wait_seconds": pacing_wait,
                    "request_id": request_id,
                    "finish_reason": finish_reason,
                    "usage": durable_usage,
                    **(
                        {"seed": options["seed"]}
                        if "seed" in options
                        else {}
                    ),
                }
            )
            return Generation(
                text=text,
                finish_reason=finish_reason,
                latency_seconds=time.perf_counter() - total_started,
                usage=durable_usage,
                metadata={
                    "provider": self.profile.provider,
                    "profile": self.profile.profile,
                    "model": self.profile.model,
                    "request_id": request_id,
                    "attempt_count": attempt,
                    "retry_wait_seconds": total_retry_wait,
                    "api_latency_seconds": request_latency,
                },
            )

        assert last_error is not None
        raise last_error
