from __future__ import annotations

import os
import unittest
from pathlib import Path

from benchmark_core.schema import Generation, Problem
from benchmark_datasets import get_dataset, list_datasets
from benchmark_datasets.base import DatasetContext
from benchmark_datasets.registry import register_dataset
from scorers.judge import JudgeDecision, JudgeRequiredError, JudgeVerdict
from scorers.numeric import NumericAnswerScorer


DATA_ROOT = Path(
    os.environ.get(
        "MATH_BENCHMARK_DATA_ROOT",
        Path(__file__).resolve().parents[2] / "math-benchmark-data",
    )
)


class DummyPlugin:
    name = "external-dummy"
    aliases = ("dummy-alias",)

    def iter_problems(self, context):
        return iter(())

    def create_scorer(self, context):
        return NumericAnswerScorer()


class AlwaysCorrectJudge:
    def judge(self, problem, candidate):
        return JudgeDecision(
            JudgeVerdict.YES,
            "test judge",
            {"backend": "fake"},
        )


class DatasetTests(unittest.TestCase):
    def test_builtin_registry_and_aliases(self):
        self.assertEqual(
            list_datasets(),
            ["gsm1k", "harp", "math-perturb", "mathconstruct", "u-math-text-only"],
        )
        self.assertIs(get_dataset("MATH-Perturb"), get_dataset("math_perturb"))

    def test_external_plugin_needs_no_runner_change(self):
        plugin = DummyPlugin()
        register_dataset(plugin)
        self.assertIs(get_dataset("dummy-alias"), plugin)

    def test_static_dataset_counts(self):
        context = DatasetContext(DATA_ROOT)
        expected = {
            "gsm1k": 1205,
            "math-perturb": 230,
            "harp": 4302,
            "mathconstruct": 439,
            "u-math-text-only": 720,
        }
        for name, count in expected.items():
            with self.subTest(dataset=name):
                problems = list(get_dataset(name).iter_problems(context))
                self.assertEqual(len(problems), count)
                self.assertEqual(problems[0].dataset, name)

    def test_gsm1k_numeric_scorer(self):
        problem = Problem("gsm1k", "x", "question", "1,024")
        scorer = NumericAnswerScorer()
        self.assertTrue(scorer.score(problem, Generation(r"\boxed{1024}")).correct)
        self.assertFalse(scorer.score(problem, Generation("1025")).correct)

    def test_u_math_requires_explicit_judge(self):
        plugin = get_dataset("u-math-text-only")
        problem = next(iter(plugin.iter_problems(DatasetContext(DATA_ROOT))))
        with self.assertRaises(JudgeRequiredError):
            plugin.create_scorer(DatasetContext(DATA_ROOT)).score(
                problem, Generation("candidate")
            )
        score = plugin.create_scorer(
            DatasetContext(DATA_ROOT, judge=AlwaysCorrectJudge())
        ).score(problem, Generation("candidate"))
        self.assertTrue(score.correct)
