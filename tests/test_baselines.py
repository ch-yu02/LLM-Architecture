from __future__ import annotations

import tempfile
import threading
import time
import tomllib
import unittest
from pathlib import Path

from benchmark_core.fairness import FairnessPolicy, FairnessViolation
from benchmark_core.runner import EvaluationRunner, SampleEvaluationError
from benchmark_core.schema import Experiment, Generation, Problem
from benchmark_core.store import ExperimentConflictError, JsonlResultStore
from benchmark_datasets.base import DatasetContext
from benchmark_methods import get_method, list_methods
from scorers.numeric import NumericAnswerScorer


ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "configs" / "evaluation" / "fair_baselines.toml").open("rb") as stream:
    FAIR_CONFIG = tomllib.load(stream)


class RecordingBackend:
    def __init__(self, answer=r"\boxed{2}"):
        self.answer = answer
        self.calls = []

    def generate(self, messages, *, config):
        self.calls.append({"messages": messages, "config": config})
        return Generation(self.answer, usage={"output_tokens": 5})


class ConcurrentBackend:
    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def generate(self, messages, *, config):
        prompt = messages[-1]["content"]
        index = int(prompt.split("What is ", 1)[1].split("+", 1)[0])
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.03 * (3 - index))
            return Generation(rf"\boxed{{{index}}}")
        finally:
            with self._lock:
                self.active -= 1


class ThreeProblemPlugin:
    name = "tiny"
    aliases = ()

    def iter_problems(self, context):
        for index in range(3):
            yield Problem("tiny", str(index), f"What is {index}+0?", str(index))

    def create_scorer(self, context):
        return NumericAnswerScorer()


class TwoCallMethod:
    name = "two_call"

    def run(self, problem, backend, *, config):
        backend.generate([{"role": "user", "content": problem.prompt}], config={})
        return backend.generate(
            [{"role": "user", "content": problem.prompt}], config={}
        )


class OverrideMethod:
    name = "override"

    def run(self, problem, backend, *, config):
        return backend.generate(
            [{"role": "user", "content": problem.prompt}],
            config={"temperature": 1.0},
        )


class BaselineTests(unittest.TestCase):
    def run_one(self, method_name):
        backend = RecordingBackend()
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlResultStore(Path(directory) / "result.jsonl")
            summary = EvaluationRunner().run(
                experiment=Experiment(
                    f"exp-{method_name}", "tiny", method_name, "fixed-model"
                ),
                plugin=ThreeProblemPlugin(),
                context=DatasetContext(Path(directory)),
                method=get_method(method_name),
                backend=backend,
                store=store,
                inference_config=FAIR_CONFIG["inference"],
                batch_size=1,
            )
            record = list(store.read())[0]
        return backend, summary, record

    def test_baselines_have_equal_resources(self):
        self.assertEqual(list_methods(), ["direct", "zero_shot_cot"])
        self.assertEqual(FAIR_CONFIG["methods"], list_methods())
        direct_backend, direct_summary, direct_record = self.run_one("direct")
        cot_backend, cot_summary, cot_record = self.run_one("zero_shot_cot")
        self.assertEqual(len(direct_backend.calls), len(cot_backend.calls))
        self.assertEqual(
            direct_backend.calls[0]["config"], cot_backend.calls[0]["config"]
        )
        self.assertEqual((direct_summary.attempted, cot_summary.attempted), (1, 1))
        for record in (direct_record, cot_record):
            self.assertEqual(record["generation"]["model_calls"], 1)
            self.assertEqual(record["generation"]["usage"]["output_tokens"], 5)

    def test_prompt_difference_is_only_strategy_instruction(self):
        direct_backend, _, _ = self.run_one("direct")
        cot_backend, _, _ = self.run_one("zero_shot_cot")
        direct_messages = direct_backend.calls[0]["messages"]
        cot_messages = cot_backend.calls[0]["messages"]
        self.assertEqual(direct_messages[0], cot_messages[0])
        self.assertIn("reasoning step by step", cot_messages[1]["content"])
        self.assertNotIn("reasoning step by step", direct_messages[1]["content"])
        harp_backend = RecordingBackend()
        get_method("direct").run(
            Problem(
                "harp",
                "one",
                "What is 1+1?",
                "2",
                answer_instruction=(
                    "End the response with exactly one final line in the form "
                    "`Answer: \\boxed{...}`. Do not write anything after that line."
                ),
            ),
            harp_backend,
            config={},
        )
        harp_prompt = harp_backend.calls[0]["messages"][1]["content"]
        self.assertTrue(
            harp_prompt.endswith("Do not write anything after that line.")
        )
        self.assertIn(r"Answer: \boxed{...}", harp_prompt)

    def test_call_budget_and_overrides_are_enforced(self):
        cases = (TwoCallMethod(), OverrideMethod())
        for method in cases:
            with self.subTest(method=method.name), tempfile.TemporaryDirectory() as directory:
                store = JsonlResultStore(Path(directory) / "result.jsonl")
                with self.assertRaises(SampleEvaluationError) as caught:
                    EvaluationRunner().run(
                        experiment=Experiment(
                            f"exp-{method.name}", "tiny", method.name, "model"
                        ),
                        plugin=ThreeProblemPlugin(),
                        context=DatasetContext(Path(directory)),
                        method=method,
                        backend=RecordingBackend(),
                        store=store,
                        batch_size=1,
                    )
                self.assertIn(
                    "FairnessViolation", caught.exception.record.score.error
                )
                self.assertFalse(store.path.exists())

    def test_batch_size_advances_through_unfinished_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlResultStore(Path(directory) / "result.jsonl")
            common = dict(
                experiment=Experiment("stable", "tiny", "direct", "model"),
                plugin=ThreeProblemPlugin(),
                context=DatasetContext(Path(directory)),
                method=get_method("direct"),
                backend=RecordingBackend("0"),
                store=store,
            )
            first = EvaluationRunner().run(**common, batch_size=1)
            second = EvaluationRunner().run(**common, batch_size=2)
            third = EvaluationRunner().run(**common, batch_size=2)
            self.assertEqual((first.attempted, first.resumed), (1, 0))
            self.assertEqual((second.attempted, second.resumed), (2, 1))
            self.assertEqual((third.attempted, third.resumed), (0, 3))
            self.assertEqual(
                [item["problem"]["id"] for item in store.read()],
                ["0", "1", "2"],
            )

    def test_concurrent_samples_are_persisted_in_dataset_order(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlResultStore(Path(directory) / "result.jsonl")
            backend = ConcurrentBackend()
            callback_ids = []
            summary = EvaluationRunner().run(
                experiment=Experiment(
                    "concurrent", "tiny", "direct", "model"
                ),
                plugin=ThreeProblemPlugin(),
                context=DatasetContext(Path(directory)),
                method=get_method("direct"),
                backend=backend,
                store=store,
                batch_size=3,
                concurrency=3,
                record_callback=lambda record: callback_ids.append(
                    record.problem.id
                ),
            )
            self.assertEqual(summary.attempted, 3)
            self.assertEqual(summary.errors, 0)
            self.assertGreaterEqual(backend.max_active, 2)
            self.assertEqual(
                [item["problem"]["id"] for item in store.read()],
                ["0", "1", "2"],
            )
            self.assertEqual(callback_ids, ["0", "1", "2"])

    def test_reused_experiment_id_cannot_change_model(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlResultStore(Path(directory) / "result.jsonl")
            common = dict(
                plugin=ThreeProblemPlugin(),
                context=DatasetContext(Path(directory)),
                method=get_method("direct"),
                backend=RecordingBackend(),
                store=store,
                batch_size=1,
            )
            EvaluationRunner().run(
                experiment=Experiment("collision", "tiny", "direct", "model-a"),
                **common,
            )
            with self.assertRaises(ExperimentConflictError):
                EvaluationRunner().run(
                    experiment=Experiment(
                        "collision", "tiny", "direct", "model-b"
                    ),
                    **common,
                )

    def test_sensitive_inference_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FairnessViolation):
                EvaluationRunner().run(
                    experiment=Experiment("secret", "tiny", "direct", "model"),
                    plugin=ThreeProblemPlugin(),
                    context=DatasetContext(Path(directory)),
                    method=get_method("direct"),
                    backend=RecordingBackend(),
                    store=JsonlResultStore(Path(directory) / "result.jsonl"),
                    inference_config={"provider": {"api_key": "do-not-store"}},
                    fairness_policy=FairnessPolicy(),
                    batch_size=1,
                )

    def test_non_positive_batch_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                EvaluationRunner().run(
                    experiment=Experiment("negative", "tiny", "direct", "model"),
                    plugin=ThreeProblemPlugin(),
                    context=DatasetContext(Path(directory)),
                    method=get_method("direct"),
                    backend=RecordingBackend(),
                    store=JsonlResultStore(Path(directory) / "result.jsonl"),
                    batch_size=0,
                )
            with self.assertRaises(ValueError):
                EvaluationRunner().run(
                    experiment=Experiment(
                        "concurrency", "tiny", "direct", "model"
                    ),
                    plugin=ThreeProblemPlugin(),
                    context=DatasetContext(Path(directory)),
                    method=get_method("direct"),
                    backend=RecordingBackend(),
                    store=JsonlResultStore(
                        Path(directory) / "concurrency.jsonl"
                    ),
                    concurrency=0,
                )
