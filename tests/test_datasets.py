from __future__ import annotations

import json
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
            [
                "gsm1k",
                "harp",
                "harp-small",
                "math-perturb",
                "mathconstruct",
                "u-math-text-only",
            ],
        )
        self.assertIs(get_dataset("MATH-Perturb"), get_dataset("math_perturb"))
        self.assertIs(get_dataset("harp-small"), get_dataset("harp_small"))

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
            "harp-small": 1434,
            "mathconstruct": 439,
            "u-math-text-only": 720,
        }
        for name, count in expected.items():
            with self.subTest(dataset=name):
                problems = list(get_dataset(name).iter_problems(context))
                self.assertEqual(len(problems), count)
                self.assertEqual(problems[0].dataset, name)

    def test_harp_splits_are_unambiguous(self):
        context = DatasetContext(DATA_ROOT)
        test = list(get_dataset("harp").iter_problems(context))
        small = list(get_dataset("harp-small").iter_problems(context))
        validation_ids = {
            json.loads(line)["id"]
            for line in (
                DATA_ROOT / "data" / "processed" / "harp_validation.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        }
        test_ids = {problem.id for problem in test}
        small_ids = {problem.id for problem in small}
        self.assertEqual(len(small_ids), 1434)
        self.assertLess(small_ids, test_ids)
        self.assertFalse(small_ids & validation_ids)
        self.assertEqual(
            {problem.metadata["split"] for problem in small},
            {"small_test"},
        )
        self.assertEqual(
            {problem.metadata["source_split"] for problem in small},
            {"test"},
        )
        self.assertEqual(
            {problem.metadata["subset_id"] for problem in small},
            {"harp_small_test_v1"},
        )

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
