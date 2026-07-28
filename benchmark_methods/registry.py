from __future__ import annotations

from benchmark_core.interfaces import EvaluationMethod


_methods: dict[str, EvaluationMethod] = {}
_builtins_loaded = False


def register_method(method: EvaluationMethod) -> EvaluationMethod:
    key = method.name.lower()
    if key in _methods:
        raise ValueError(f"Method already registered: {method.name}")
    _methods[key] = method
    return method


def _load_builtins() -> None:
    global _builtins_loaded
    if not _builtins_loaded:
        from . import baselines  # noqa: F401

        _builtins_loaded = True


def get_method(name: str) -> EvaluationMethod:
    _load_builtins()
    try:
        return _methods[name.lower()]
    except KeyError as exc:
        raise KeyError(
            f"Unknown method {name!r}; available: {', '.join(list_methods())}"
        ) from exc


def list_methods() -> list[str]:
    _load_builtins()
    return sorted(_methods)
