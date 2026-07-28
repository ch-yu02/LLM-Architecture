#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_smoke import SmokeMatrixRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 5x5 plumbing smoke matrix")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "configs" / "smoke" / "matrix.toml",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT.parent / "math-benchmark-data",
    )
    parser.add_argument(
        "--checker-python",
        default=os.environ.get(
            "MATH_CHECKER_PYTHON",
            str(ROOT / ".venv-checkers" / "bin" / "python"),
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "smoke" / "plumbing_matrix.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every cell instead of only the summary and failures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checker = Path(args.checker_python)
    if not checker.is_file():
        print(f"Checker Python not found: {checker}", file=sys.stderr)
        return 2
    runner = SmokeMatrixRunner(
        runner_root=ROOT,
        data_root=args.data_root,
        checker_python=str(checker.absolute()),
    )
    report = runner.run(args.matrix)
    runner.write_report(report, args.output)
    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"{report['matrix']}: {report['passed']}/{report['expected_cells']} "
            f"passed, {report['failed']} failed, "
            f"{report['duration_seconds']:.3f}s"
        )
        for cell in report["cells"]:
            if cell["status"] != "passed":
                print(
                    f"FAIL {cell['dataset']} × {cell['method']}: "
                    f"{cell['error'] or cell['score_status']}"
                )
    print(f"Report: {args.output}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
