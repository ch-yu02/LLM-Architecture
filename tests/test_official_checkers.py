from __future__ import annotations

import os
import unittest
from pathlib import Path

from benchmark_core.schema import Generation
from benchmark_datasets import get_dataset
from benchmark_datasets.base import DatasetContext


DATA_ROOT = Path(
    os.environ.get(
        "MATH_BENCHMARK_DATA_ROOT",
        Path(__file__).resolve().parents[2] / "math-benchmark-data",
    )
)
CHECKER_PYTHON = os.environ.get("MATH_CHECKER_PYTHON")


@unittest.skipUnless(CHECKER_PYTHON, "set MATH_CHECKER_PYTHON for official checker tests")
class OfficialCheckerTests(unittest.TestCase):
    def context(self):
        return DatasetContext(
            DATA_ROOT,
            bridge_python={
                "math-perturb": CHECKER_PYTHON,
                "harp": CHECKER_PYTHON,
                "mathconstruct": CHECKER_PYTHON,
            },
        )

    def test_math_perturb(self):
        plugin = get_dataset("math-perturb")
        problem = next(iter(plugin.iter_problems(self.context())))
        scorer = plugin.create_scorer(self.context())
        self.assertTrue(scorer.score(problem, Generation(r"\boxed{10}")).correct)
        self.assertFalse(scorer.score(problem, Generation(r"\boxed{11}")).correct)

    def test_harp(self):
        plugin = get_dataset("harp")
        problem = next(iter(plugin.iter_problems(self.context())))
        scorer = plugin.create_scorer(self.context())
        self.assertTrue(
            scorer.score(problem, Generation(r"\boxed{10\frac{2}{3}}")).correct
        )
        self.assertFalse(scorer.score(problem, Generation(r"\boxed{11}")).correct)

    def test_mathconstruct_frozen_test_split_and_scoring(self):
        plugin = get_dataset("mathconstruct")
        problems = list(plugin.iter_problems(self.context()))
        self.assertEqual(len(problems), 439)
        self.assertEqual(
            len({problem.metadata["family"] for problem in problems}),
            97,
        )
        scorer = plugin.create_scorer(self.context())
        correct_problem = next(
            problem
            for problem in problems
            if (
                problem.metadata["family"]
                == "bulgarian-mo-r2-2021-8-4"
                and problem.metadata["instance_kind"] == "original"
            )
        )
        correct_score = scorer.score(
            correct_problem, Generation(r"\boxed{4194304}")
        )
        self.assertEqual(correct_score.status.value, "scored")
        self.assertTrue(correct_score.correct)
        wrong_score = scorer.score(
            problems[0], Generation("This is intentionally not a formatted answer.")
        )
        self.assertEqual(wrong_score.status.value, "scored")
        self.assertFalse(wrong_score.correct)
