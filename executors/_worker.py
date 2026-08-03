#!/usr/bin/env python3
from __future__ import annotations

import ast
import io
import json
import math
import resource
import sys
from contextlib import redirect_stdout
from decimal import Decimal
from fractions import Fraction
from typing import Any


_DENIED_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)
_ALLOWED_MODULES = {
    "cmath",
    "collections",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "math",
    "numpy",
    "scipy",
    "statistics",
    "sympy",
    "sys",
}


def _module_is_allowed(name: str) -> bool:
    return name.split(".", 1)[0] in _ALLOWED_MODULES


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, _DENIED_NODES):
            raise ValueError(f"syntax is not allowed: {type(node).__name__}")
        if (
            isinstance(node, ast.Name)
            and node.id.startswith("_")
            and node.id not in {"_", "__name__"}
        ):
            raise ValueError(f"private name is not allowed: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"private attribute is not allowed: {node.attr}")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
            and node.attr != "setrecursionlimit"
        ):
            raise ValueError(f"sys attribute is not allowed: {node.attr}")
        if isinstance(node, ast.Import):
            if any(not _module_is_allowed(alias.name) for alias in node.names):
                raise ValueError("module is not allowed")
        if isinstance(node, ast.ImportFrom):
            if node.level or not node.module or not _module_is_allowed(node.module):
                raise ValueError("module is not allowed")
            if node.module == "sys" and any(
                alias.name != "setrecursionlimit" for alias in node.names
            ):
                raise ValueError("only sys.setrecursionlimit is allowed")
            if any(
                alias.name == "*" or alias.name.startswith("_")
                for alias in node.names
            ):
                raise ValueError("wildcard/private imports are not allowed")


def _set_limits(request: dict[str, Any]) -> None:
    memory_bytes = int(request["memory_mb"]) * 1024 * 1024
    cpu_seconds = max(1, int(request["cpu_seconds"]))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (8, 8))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def _safe_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    if level or not _module_is_allowed(name):
        raise ImportError(f"module is not allowed: {name}")
    return __import__(name, globals_, locals_, fromlist, level)


def _safe_hasattr(value: Any, name: str) -> bool:
    if not isinstance(name, str) or name.startswith("_"):
        return False
    return hasattr(value, name)


def _safe_builtins() -> dict[str, Any]:
    return {
        "__import__": _safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "chr": chr,
        "complex": complex,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "hasattr": _safe_hasattr,
        "isinstance": isinstance,
        "int": int,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "Exception": Exception,
        "ImportError": ImportError,
        "RuntimeError": RuntimeError,
        "TypeError": TypeError,
        "ValueError": ValueError,
    }


def main() -> None:
    request = json.load(sys.stdin)
    _set_limits(request)
    tree = ast.parse(request["code"], mode="exec")
    _validate(tree)
    globals_dict = {
        "__builtins__": _safe_builtins(),
        "__name__": "__main__",
        "Decimal": Decimal,
        "Fraction": Fraction,
        "math": math,
    }
    with redirect_stdout(io.StringIO()):
        exec(compile(tree, "<generated-program>", "exec"), globals_dict)
        solution = globals_dict.get("solution")
        if not callable(solution):
            raise ValueError("generated program must define callable solution()")
        result = solution()
    json.dump(
        {"ok": True, "value": result, "value_type": type(result).__name__},
        sys.stdout,
        ensure_ascii=False,
        default=str,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        json.dump(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            sys.stdout,
            ensure_ascii=False,
        )
        raise SystemExit(1)
