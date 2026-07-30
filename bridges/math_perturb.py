#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import sys
import traceback
from pathlib import Path


def main() -> None:
    source_root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(source_root))
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        from evaluate import answer_check, extract_predicted_answer

        predicted = extract_predicted_answer(
            request["problem"], request["generated_text"]
        )
        correct = answer_check(
            request["problem"],
            request["generated_text"],
            request["reference_answer"],
            request.get("dataset_type", "perturb"),
        )
    json.dump(
        {
            "correct": bool(correct),
            "extracted_answer": predicted,
            "details": {
                "official_checker": "MATH-Perturb/evaluate.py",
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
