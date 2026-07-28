from __future__ import annotations

import os
import unittest
from pathlib import Path

from benchmark_smoke import SmokeMatrixRunner


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get(
        "MATH_BENCHMARK_DATA_ROOT",
        ROOT.parent / "math-benchmark-data",
    )
)
CHECKER_PYTHON = os.environ.get("MATH_CHECKER_PYTHON")
MATRIX_PATH = ROOT / "configs" / "smoke" / "matrix.toml"


class SmokeMatrixTests(unittest.TestCase):
    def test_matrix_contract_is_five_by_five(self):
        matrix = SmokeMatrixRunner.load_matrix(MATRIX_PATH)
        self.assertEqual(matrix["samples_per_cell"], 1)
        self.assertEqual(
            matrix["methods"],
            ["direct", "zero_shot_cot", "pal", "self_refine", "aflow"],
        )
        self.assertEqual(
            [dataset["name"] for dataset in matrix["datasets"]],
            [
                "gsm1k",
                "math-perturb",
                "harp",
                "u-math-text-only",
                "mathconstruct",
            ],
        )
        self.assertEqual(
            len(matrix["methods"]) * len(matrix["datasets"]), 25
        )

    @unittest.skipUnless(
        CHECKER_PYTHON,
        "set MATH_CHECKER_PYTHON for the full plumbing smoke matrix",
    )
    def test_full_matrix(self):
        report = SmokeMatrixRunner(
            runner_root=ROOT,
            data_root=DATA_ROOT,
            checker_python=CHECKER_PYTHON,
        ).run(MATRIX_PATH)
        self.assertEqual(report["expected_cells"], 25)
        self.assertEqual(report["passed"], 25)
        self.assertEqual(report["failed"], 0)
        pairs = {
            (cell["dataset"], cell["method"]) for cell in report["cells"]
        }
        self.assertEqual(len(pairs), 25)
        for cell in report["cells"]:
            expected_calls = 2 if cell["method"] == "self_refine" else 1
            self.assertEqual(cell["model_calls"], expected_calls)
