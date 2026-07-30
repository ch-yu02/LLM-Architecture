#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import queue
import re
import sys
import traceback
import types
from enum import Enum
from pathlib import Path
from typing import Any, Callable


_BOXED_COMMAND = re.compile(r"\\boxed\s*\{")
_ANSWER_LINE = re.compile(
    r"(?im)^[ \t]*Answer:[ \t]*(.*?)[ \t]*$"
)


def _balanced_box_contents(text: str) -> list[str]:
    r"""Return every complete ``\boxed{...}``, including nested braces."""

    contents: list[str] = []
    for match in _BOXED_COMMAND.finditer(text):
        opening_brace = match.end() - 1
        depth = 0
        escaped = False
        for index in range(opening_brace, len(text)):
            character = text[index]
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[opening_brace + 1 : index].strip())
                    break
    return contents


def extract_final_answer(text: object) -> tuple[str | None, str]:
    """Select the last complete box, then the last standalone Answer line."""

    if text is None:
        return None, "missing"
    raw = str(text)
    boxed = _balanced_box_contents(raw)
    if boxed:
        return boxed[-1], "last_complete_boxed"

    answer_lines = [
        match.group(1).strip()
        for match in _ANSWER_LINE.finditer(raw)
        if match.group(1).strip()
    ]
    if answer_lines:
        return answer_lines[-1], "last_answer_line"
    return None, "missing"


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
    candidate, extraction_source = extract_final_answer(
        request.get("generated_text")
    )
    if request.get("finish_reason", "stop") != "stop":
        candidate = None
        extraction_source = "non_stop_finish"
    with contextlib.redirect_stdout(sys.stderr):
        from eval.latex_answer_check import latex_answer_check

        result = latex_answer_check(
            [
                {
                    "generated_text": candidate,
                    "finish_reason": request.get("finish_reason", "stop"),
                    "answer": request["reference_answer"],
                }
            ],
            extract_policy="none",
            eval_policy="aggressive",
            use_tqdm=False,
        )[0]
    json.dump(
        {
            "correct": bool(result["is_correct"]),
            "extracted_answer": result.get("predict"),
            "details": {
                "literal_correct": result.get("is_literal_correct"),
                "official_checker": "HARP/eval/latex_answer_check.py",
                "official_extract_policy": "none",
                "official_eval_policy": "aggressive",
                "answer_extraction": extraction_source,
                "finish_reason": request.get("finish_reason", "stop"),
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
