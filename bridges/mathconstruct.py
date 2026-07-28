#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> str:
    return str(value)


def main() -> None:
    source_root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(source_root / "src"))
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        from math_construct.problems import get_problem_class

        operation = request["operation"]
        if operation != "score":
            raise ValueError(f"Unknown operation: {operation}")
        metadata = request["metadata"]
        problem_class = get_problem_class(metadata["instance"]["config"]["name"])
        if problem_class is None:
            raise KeyError(metadata["instance"]["config"]["name"])
        instance = problem_class.from_json(metadata["instance"])
        # Current upstream annotation permits str, but warn_small_length
        # expects the chat-message form used by the official evaluator.
        answer, correct, details = instance.parse_and_check(
            [{"role": "assistant", "content": request["generated_text"]}]
        )
        response = {
            "correct": bool(correct),
            "extracted_answer": answer,
            "details": {
                "checker_details": details,
                "official_checker": "MathConstruct/parse_and_check",
            },
        }
    json.dump(response, sys.stdout, ensure_ascii=False, default=_json_default)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
