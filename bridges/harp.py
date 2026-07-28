#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import queue
import sys
import traceback
import types
from enum import Enum
from pathlib import Path
from typing import Any, Callable


def _timeout_worker(
    output_queue: mp.Queue,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    try:
        output_queue.put(function(*args, **kwargs))
    except Exception:
        output_queue.put(None)


def _run_with_timeout(
    function: Callable[..., Any],
    timeout: int,
    default: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    output_queue = mp.Queue()
    process = mp.Process(
        target=_timeout_worker,
        args=(output_queue, function, args, kwargs),
    )
    process.start()
    try:
        result = output_queue.get(timeout=timeout)
        process.join()
        return result
    except (mp.TimeoutError, queue.Empty):
        process.terminate()
        process.join()
        return default
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def main() -> None:
    source_root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(source_root))

    # HARP's eval.utils imports cloud SDKs unrelated to answer checking. Supply only
    # the utility required by latex_answer_check.
    utility_module = types.ModuleType("eval.utils")
    utility_module.run_with_timeout = _run_with_timeout
    sys.modules["eval.utils"] = utility_module
    enum_module = types.ModuleType("eval.enums")

    class ModelAPI(str, Enum):
        OPENAI = "openai"
        ANTHROPIC = "anthropic"
        GOOGLE = "google"
        TOGETHER = "together"

    enum_module.ModelAPI = ModelAPI
    sys.modules["eval.enums"] = enum_module

    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        from eval.latex_answer_check import latex_answer_check

        result = latex_answer_check(
            [
                {
                    "generated_text": request["generated_text"],
                    "finish_reason": request.get("finish_reason", "stop"),
                    "answer": request["reference_answer"],
                }
            ],
            use_tqdm=False,
        )[0]
    json.dump(
        {
            "correct": bool(result["is_correct"]),
            "extracted_answer": result.get("predict"),
            "details": {
                "literal_correct": result.get("is_literal_correct"),
                "official_checker": "HARP/eval/latex_answer_check.py",
            },
        },
        sys.stdout,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
