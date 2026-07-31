from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from adapters import AFlowAdapter, PALAdapter, SelfRefineAdapter
from adapters.pal import format_final_answer
from benchmark_core.fairness import FairnessPolicy
from benchmark_core.runner import EvaluationRunner, SampleEvaluationError
from benchmark_core.schema import Experiment, Generation, Problem
from benchmark_core.store import JsonlResultStore
from benchmark_datasets.base import DatasetContext
from benchmark_methods import load_method_config, method_run_settings
from executors import ProgramExecutionError, RestrictedPythonExecutor
from scorers.numeric import NumericAnswerScorer


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


class QueueBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages, *, config):
        self.calls.append({"messages": messages, "config": config})
        if not self.responses:
            raise AssertionError("No fake response remaining")
        response = self.responses.pop(0)
        if isinstance(response, Generation):
            return response
        return Generation(
            response,
            usage={"input_tokens": 10, "output_tokens": 5},
        )


class OneProblemPlugin:
    name = "tiny"
    aliases = ()

    def iter_problems(self, context):
        yield Problem("tiny", "one", "What is 1+1?", "2")

    def create_scorer(self, context):
        return NumericAnswerScorer()


class MethodAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pal_source = WORKSPACE / "methods" / "PAL"
        cls.self_refine_source = WORKSPACE / "methods" / "Self-Refine"
        cls.aflow_source = WORKSPACE / "methods" / "AFlow"

    def run_method(
        self,
        method,
        backend,
        *,
        method_config=None,
        policy=None,
    ):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlResultStore(Path(directory) / "results.jsonl")
            summary = EvaluationRunner().run(
                experiment=Experiment(
                    f"exp-{method.name}", "tiny", method.name, "fake-model"
                ),
                plugin=OneProblemPlugin(),
                context=DatasetContext(Path(directory)),
                method=method,
                backend=backend,
                store=store,
                method_config=method_config or {},
                fairness_policy=policy,
                batch_size=1,
            )
            record = list(store.read())[0]
        return summary, record

    def test_restricted_executor_runs_arithmetic(self):
        result = RestrictedPythonExecutor().execute(
            "from fractions import Fraction\n"
            "def solution():\n"
            "    return Fraction(1, 2) + Fraction(3, 2)"
        )
        self.assertEqual(result.value, "2")

    def test_final_answer_envelope_does_not_double_wrap(self):
        self.assertEqual(
            format_final_answer(r"Answer: \boxed{\frac{1}{2}}"),
            r"Answer: \boxed{\frac{1}{2}}",
        )
        self.assertEqual(
            format_final_answer(r"\boxed{2} trailing"),
            r"Answer: \boxed{\boxed{2} trailing}",
        )

    def test_restricted_executor_rejects_imports_and_private_access(self):
        executor = RestrictedPythonExecutor()
        for code in (
            "import os\ndef solution():\n    return 1",
            "def solution():\n    return (1).__class__",
        ):
            with self.subTest(code=code), self.assertRaises(ProgramExecutionError):
                executor.execute(code)

    def test_restricted_executor_times_out(self):
        executor = RestrictedPythonExecutor(
            timeout_seconds=0.2, cpu_seconds=1
        )
        with self.assertRaises(ProgramExecutionError):
            executor.execute("def solution():\n    while True:\n        pass")

    def test_pal_end_to_end(self):
        backend = QueueBackend(["```python\ndef solution():\n    return 2\n```"])
        summary, record = self.run_method(PALAdapter(self.pal_source), backend)
        self.assertEqual((summary.scored, summary.correct), (1, 1))
        self.assertEqual(record["generation"]["text"], r"Answer: \boxed{2}")
        self.assertEqual(record["generation"]["finish_reason"], "stop")
        self.assertIn("three examples", backend.calls[0]["messages"][1]["content"])
        self.assertEqual(record["generation"]["model_calls"], 1)

    def test_self_refine_end_to_end_and_usage_aggregation(self):
        backend = QueueBackend(
            [
                "def solution():\n    return 1",
                (
                    "The arithmetic is wrong. Here is the rewrite:\n\n"
                    "def solution():\n    return 2\n### END ###"
                ),
            ]
        )
        summary, record = self.run_method(
            SelfRefineAdapter(self.self_refine_source),
            backend,
            method_config={
                "max_refinements": 1,
                "feedback_inference_overrides": {"temperature": 0.7},
            },
            policy=FairnessPolicy(
                min_model_calls_per_problem=2,
                max_model_calls_per_problem=2,
                allowed_inference_overrides=("temperature",),
            ),
        )
        self.assertEqual((summary.scored, summary.correct), (1, 1))
        self.assertEqual(record["generation"]["usage"]["input_tokens"], 20)
        self.assertEqual(record["generation"]["model_calls"], 2)
        self.assertEqual(
            record["generation"]["method_details"]["refinement_iterations"],
            1,
        )
        self.assertEqual(backend.calls[1]["config"]["temperature"], 0.7)

    def test_aflow_official_round_one_and_frozen_graph(self):
        default_backend = QueueBackend([r"\boxed{2}"])
        summary, record = self.run_method(
            AFlowAdapter(self.aflow_source), default_backend
        )
        self.assertEqual((summary.scored, summary.correct), (1, 1))
        self.assertEqual(
            record["generation"]["method_details"]["workflow_steps"],
            [{"id": "answer", "operator": "custom"}],
        )

        workflow = {
            "id": "test-ensemble",
            "nodes": [
                {"id": "a", "operator": "custom", "inputs": []},
                {"id": "b", "operator": "answer_generate", "inputs": []},
                {"id": "pick", "operator": "ensemble", "inputs": ["a", "b"]},
            ],
            "output": "pick",
        }
        backend = QueueBackend(
            [
                "wrong",
                "<answer>2</answer>",
                "<solution_letter>B</solution_letter>",
            ]
        )
        summary, record = self.run_method(
            AFlowAdapter(self.aflow_source),
            backend,
            method_config={
                "workflow": workflow,
                "workflow_artifact": {
                    "source": "unit-test optimizer",
                    "optimization_split": "validation",
                    "optimization_cost": 1.25,
                    "evaluation_data_used": False,
                    "frozen": True,
                },
            },
            policy=FairnessPolicy(
                min_model_calls_per_problem=3,
                max_model_calls_per_problem=3,
            ),
        )
        self.assertEqual((summary.scored, summary.correct), (1, 1))
        self.assertEqual(record["generation"]["text"], r"Answer: \boxed{2}")
        self.assertEqual(record["generation"]["model_calls"], 3)

    def test_successful_program_output_is_complete_even_if_code_call_hit_length(self):
        backend = QueueBackend(
            [
                Generation(
                    "```python\ndef solution():\n    return 2\n```",
                    finish_reason="length",
                )
            ]
        )
        summary, record = self.run_method(PALAdapter(self.pal_source), backend)
        self.assertEqual((summary.scored, summary.correct), (1, 1))
        self.assertEqual(record["generation"]["finish_reason"], "stop")

    def test_aflow_rejects_unfrozen_workflow(self):
        backend = QueueBackend(["unused"])
        with self.assertRaises(SampleEvaluationError) as caught:
            self.run_method(
                AFlowAdapter(self.aflow_source),
                backend,
                method_config={
                    "workflow_artifact": {
                        "source": "online",
                        "optimization_split": "test",
                        "optimization_cost": 0,
                        "evaluation_data_used": True,
                        "frozen": False,
                    }
                },
            )
        self.assertIn(
            "frozen workflow_artifact",
            caught.exception.record.score.error,
        )

    def test_aflow_programmer_retries_with_execution_feedback(self):
        workflow = {
            "id": "programmer-retry",
            "nodes": [
                {
                    "id": "program",
                    "operator": "programmer",
                    "inputs": [],
                    "max_attempts": 2,
                }
            ],
            "output": "program",
        }
        backend = QueueBackend(
            [
                "```python\ndef solution():\n    return missing_name\n```",
                "```python\ndef solution():\n    return 2\n```",
            ]
        )
        summary, record = self.run_method(
            AFlowAdapter(self.aflow_source),
            backend,
            method_config={
                "workflow": workflow,
                "workflow_artifact": {
                    "source": "unit-test optimizer",
                    "optimization_split": "validation",
                    "optimization_cost": 0.5,
                    "evaluation_data_used": False,
                    "frozen": True,
                },
            },
            policy=FairnessPolicy(
                min_model_calls_per_problem=1,
                max_model_calls_per_problem=2,
            ),
        )
        self.assertEqual((summary.scored, summary.correct), (1, 1))
        self.assertEqual(record["generation"]["model_calls"], 2)
        attempts = record["generation"]["method_details"]["workflow_steps"][0][
            "attempts"
        ]
        self.assertEqual(
            [attempt["status"] for attempt in attempts], ["error", "success"]
        )
        self.assertIn(
            "previous program failed",
            backend.calls[1]["messages"][0]["content"],
        )

    def test_tracked_configs_construct_all_adapters(self):
        for name in ("pal", "self_refine", "aflow"):
            with self.subTest(method=name):
                method, config = load_method_config(
                    ROOT / "configs" / "methods" / f"{name}.toml"
                )
                self.assertEqual(method.name, name)
                self.assertEqual(config["status"], "runtime_ready")
                method_config, policy = method_run_settings(config)
                self.assertLessEqual(
                    policy.min_model_calls_per_problem,
                    policy.max_model_calls_per_problem,
                )
                if name == "aflow":
                    self.assertTrue(
                        method_config["workflow_artifact"]["frozen"]
                    )
