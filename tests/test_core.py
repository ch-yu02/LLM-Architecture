from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_core.runner import EvaluationRunner
from benchmark_core.schema import (
    EvaluationRecord,
    Experiment,
    Generation,
    Problem,
    Score,
)
from benchmark_core.store import ExperimentConflictError, JsonlResultStore
from benchmark_datasets.base import DatasetContext
from benchmark_methods import get_method


class StaticBackend:
    def __init__(self, text: str):
        self.text = text

    def generate(self, messages, *, config):
        return Generation(self.text, metadata={"message_count": len(messages)})


class FailingBackend:
    def generate(self, messages, *, config):
        raise RuntimeError("local backend failure")


class TinyPlugin:
    name = "tiny"
    aliases = ()

    def iter_problems(self, context):
        yield Problem("tiny", "one", "What is 1+1?", "2")

    def create_scorer(self, context):
        from scorers.numeric import NumericAnswerScorer

        return NumericAnswerScorer()


class CoreTests(unittest.TestCase):
    def test_store_rejects_concurrent_runs_for_the_same_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlResultStore(Path(directory) / "results.jsonl")
            competing_store = JsonlResultStore(store.path)
            with store.exclusive_run():
                with self.assertRaisesRegex(
                    ExperimentConflictError,
                    "already running",
                ):
                    with competing_store.exclusive_run():
                        self.fail("concurrent run lock should not be acquired")

    def test_record_is_json_serializable(self):
        record = EvaluationRecord(
            Experiment("exp", "tiny", "direct", "static"),
            Problem("tiny", "one", "p", "2"),
            Generation("2"),
            Score.scored(True),
        )
        self.assertEqual(json.loads(json.dumps(record.to_dict()))["score"]["status"], "scored")

    def test_runner_scores_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlResultStore(Path(directory) / "results.jsonl")
            callback_records = []
            kwargs = dict(
                experiment=Experiment("exp", "tiny", "direct", "static"),
                plugin=TinyPlugin(),
                context=DatasetContext(Path(directory)),
                method=get_method("direct"),
                backend=StaticBackend(r"\boxed{2}"),
                store=store,
                record_callback=callback_records.append,
            )
            first = EvaluationRunner().run(**kwargs)
            second = EvaluationRunner().run(**kwargs)
            self.assertEqual((first.scored, first.correct), (1, 1))
            self.assertEqual((second.attempted, second.resumed), (0, 1))
            self.assertEqual(len(list(store.read())), 1)
            self.assertEqual(len(callback_records), 1)
            record = list(store.read())[0]
            self.assertGreaterEqual(record["timing"]["processing_seconds"], 0)

    def test_runner_records_error_stage_type_and_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlResultStore(Path(directory) / "results.jsonl")
            summary = EvaluationRunner().run(
                experiment=Experiment("error", "tiny", "direct", "failing"),
                plugin=TinyPlugin(),
                context=DatasetContext(Path(directory)),
                method=get_method("direct"),
                backend=FailingBackend(),
                store=store,
            )
            self.assertEqual(summary.errors, 1)
            details = list(store.read())[0]["score"]["details"]
            self.assertEqual(details["failure_stage"], "method")
            self.assertEqual(details["exception_type"], "RuntimeError")
            self.assertNotIn("traceback", details)
