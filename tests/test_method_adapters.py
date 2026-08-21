from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from adapters import AFlowAdapter, PALAdapter, SelfRefineAdapter
from adapters.aflow import workflow_call_bounds
from adapters.pal import format_final_answer
from benchmark_core.fairness import FairnessPolicy
from benchmark_core.interfaces import MethodOutcomeError
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


class TwoProblemPlugin(OneProblemPlugin):
    def iter_problems(self, context):
        yield Problem("tiny", "one", "What is 1+1?", "2")
        yield Problem("tiny", "two", "What is 1+2?", "3")


class TwoTextProblemPlugin:
    name = "math-perturb"
    aliases = ()

    def iter_problems(self, context):
        instruction = "Put the final answer in `\\boxed{...}`."
        yield Problem(
            self.name,
            "one",
            "What is 1+1?",
            "2",
            answer_instruction=instruction,
        )
        yield Problem(
            self.name,
            "two",
            "What is 1+2?",
            "3",
            answer_instruction=instruction,
        )

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

    def test_restricted_executor_renders_exact_math_values_as_latex(self):
        fraction = RestrictedPythonExecutor().execute(
            "from fractions import Fraction\n"
            "def solution():\n"
            "    return Fraction(1, 2)"
        )
        symbolic = RestrictedPythonExecutor().execute(
            "from sympy import symbols\n"
            "def solution():\n"
            "    r = symbols('r')\n"
            "    return r**2"
        )

        self.assertEqual(fraction.latex_value, r"\frac{1}{2}")
        self.assertEqual(symbolic.latex_value, r"r^{2}")

    def test_restricted_executor_supports_pal_math_dependencies(self):
        result = RestrictedPythonExecutor().execute(
            "import numpy as np\n"
            "from scipy.optimize import minimize_scalar\n"
            "from sympy import symbols, solve\n"
            "def solution():\n"
            "    _, expected = (0, 4)\n"
            "    assert hasattr(np, 'array')\n"
            "    x = symbols('x')\n"
            "    symbolic = int(solve(x - expected, x)[0])\n"
            "    optimized = round(minimize_scalar(\n"
            "        lambda value: (value - expected) ** 2\n"
            "    ).x)\n"
            "    return int(np.array([symbolic, optimized]).mean())"
        )
        self.assertEqual(result.value, 4)

    def test_restricted_executor_environment_preflight(self):
        RestrictedPythonExecutor().validate_environment()

    def test_restricted_executor_isolates_generated_stdout(self):
        result = RestrictedPythonExecutor().execute(
            "print('top level output')\n"
            "def solution():\n"
            "    print('solution output')\n"
            "    return 2"
        )
        self.assertEqual(result.value, 2)

    def test_final_answer_envelope_does_not_double_wrap(self):
        self.assertEqual(
            format_final_answer(r"Answer: \boxed{\frac{1}{2}}"),
            r"Answer: \boxed{\frac{1}{2}}",
        )
        self.assertEqual(
            format_final_answer(r"\boxed{2} trailing"),
            r"Answer: \boxed{\boxed{2} trailing}",
        )
        self.assertEqual(
            format_final_answer("r**2", latex_value=r"r^{2}"),
            r"Answer: \boxed{r^{2}}",
        )

    def test_restricted_executor_rejects_imports_and_private_access(self):
        executor = RestrictedPythonExecutor()
        for code in (
            "import os\ndef solution():\n    return 1",
            "def solution():\n    return (1).__class__",
            "import sys\ndef solution():\n    return sys.modules",
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

    def test_pal_routes_non_gsm_datasets_to_program_return_contracts(self):
        method = PALAdapter(self.pal_source)
        cases = {
            "math-perturb": "final mathematical answer",
            "harp": "single final mathematical answer",
            "u-math-text-only": "multipart, or textual answers",
        }
        for dataset, expected_contract in cases.items():
            with self.subTest(dataset=dataset):
                backend = QueueBackend(
                    ["```python\ndef solution():\n    return 'x = 2'\n```"]
                )
                result = method.run(
                    Problem(dataset, "one", "Find x.", "x = 2"),
                    backend,
                    config={},
                )
                prompt = backend.calls[0]["messages"][1]["content"]
                self.assertIn("Return-value contract:", prompt)
                self.assertIn(expected_contract, prompt)
                self.assertIn("valid LaTeX", prompt)
                self.assertIn("rather than a Python list or tuple", prompt)
                if dataset == "harp":
                    self.assertIn("Preserve units", prompt)
                self.assertIn("Find x.", prompt)
                self.assertNotIn("Olivia has $23", prompt)
                self.assertNotIn(r"Answer: \\boxed", prompt)
                self.assertEqual(result.text, r"Answer: \boxed{x = 2}")
                self.assertEqual(
                    result.metadata["prompt_profile"],
                    f"{dataset}-program-return",
                )

    def test_pal_renders_symbolic_execution_result_as_latex(self):
        backend = QueueBackend(
            [
                "```python\n"
                "from sympy import symbols\n"
                "def solution():\n"
                "    r = symbols('r')\n"
                "    return r**2\n"
                "```"
            ]
        )
        result = PALAdapter(self.pal_source).run(
            Problem("harp-small", "one", "Find the area.", r"$r^{2}$"),
            backend,
            config={},
        )

        self.assertEqual(result.text, r"Answer: \boxed{r^{2}}")
        self.assertEqual(result.metadata["execution"]["answer_rendering"], "latex")
        self.assertEqual(result.metadata["execution"]["protocol"], "pal-python-v5")

    def test_pal_invalid_generated_program_is_scored_incorrect(self):
        backend = QueueBackend(
            ["```python\ndef solution():\n    return missing_name\n```"]
        )
        summary, record = self.run_method(PALAdapter(self.pal_source), backend)

        self.assertEqual((summary.scored, summary.correct), (1, 0))
        self.assertEqual(record["generation"]["text"], "")
        self.assertEqual(record["generation"]["model_calls"], 1)
        self.assertEqual(record["generation"]["usage"]["input_tokens"], 10)
        self.assertEqual(record["score"]["status"], "scored")
        self.assertFalse(record["score"]["correct"])
        details = record["generation"]["method_details"]
        self.assertIn("return missing_name", details["generated_program"])
        self.assertEqual(details["execution"]["status"], "error")
        self.assertIn("missing_name", details["execution"]["error"])

    def test_pal_resume_uses_current_executor_for_new_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlResultStore(Path(directory) / "results.jsonl")
            experiment = Experiment(
                "exp-pal-resume", "tiny", "pal", "fake-model"
            )
            context = DatasetContext(Path(directory))
            legacy_executor = RestrictedPythonExecutor()
            legacy_executor.protocol = "legacy-pal-python"
            first = EvaluationRunner().run(
                experiment=experiment,
                plugin=TwoProblemPlugin(),
                context=context,
                method=PALAdapter(
                    self.pal_source, executor=legacy_executor
                ),
                backend=QueueBackend(
                    ["```python\ndef solution():\n    return 2\n```"]
                ),
                store=store,
                batch_size=1,
            )
            second = EvaluationRunner().run(
                experiment=experiment,
                plugin=TwoProblemPlugin(),
                context=context,
                method=PALAdapter(self.pal_source),
                backend=QueueBackend(
                    ["```python\ndef solution():\n    return 3\n```"]
                ),
                store=store,
                batch_size=1,
            )
            records = list(store.read())

        self.assertEqual((first.scored, second.resumed, second.scored), (1, 1, 1))
        protocols = [
            record["generation"]["method_details"]["execution"]["protocol"]
            for record in records
        ]
        self.assertEqual(protocols, ["legacy-pal-python", "pal-python-v5"])

    def test_self_refine_end_to_end_and_usage_aggregation(self):
        backend = QueueBackend(
            [
                Generation(
                    "def solution():\n    return 1",
                    finish_reason="length",
                    usage={"input_tokens": 10, "output_tokens": 5},
                ),
                Generation(
                    (
                        "The arithmetic is wrong.\n"
                        "# VERDICT: INCORRECT\n"
                        "def solution():\n    return 2\n### END ###"
                    ),
                    finish_reason="stop",
                    usage={"input_tokens": 10, "output_tokens": 5},
                ),
            ]
        )
        summary, record = self.run_method(
            SelfRefineAdapter(self.self_refine_source),
            backend,
            method_config={
                "max_refinements": 1,
                "gsm_initial_inference_overrides": {
                    "max_output_tokens": 1024,
                    "stop": ["\n\n"],
                },
                "gsm_feedback_inference_overrides": {
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                    "stop": ["### END"],
                },
            },
            policy=FairnessPolicy(
                min_model_calls_per_problem=2,
                max_model_calls_per_problem=2,
                allowed_inference_overrides=(
                    "temperature",
                    "max_output_tokens",
                    "stop",
                ),
            ),
        )
        self.assertEqual((summary.scored, summary.correct), (1, 1))
        self.assertEqual(record["generation"]["usage"]["input_tokens"], 20)
        self.assertEqual(record["generation"]["model_calls"], 2)
        self.assertEqual(
            record["generation"]["method_details"]["refinement_iterations"],
            1,
        )
        self.assertEqual(
            backend.calls[0]["config"],
            {"max_output_tokens": 1024, "stop": ["\n\n"]},
        )
        self.assertEqual(backend.calls[1]["config"]["temperature"], 0.7)
        self.assertEqual(backend.calls[1]["config"]["max_output_tokens"], 2048)
        self.assertEqual(backend.calls[1]["config"]["stop"], ["### END"])
        feedback_prompt = backend.calls[1]["messages"][0]["content"]
        self.assertEqual(feedback_prompt.count("# VERDICT: CORRECT"), 2)
        self.assertEqual(feedback_prompt.count("# VERDICT: INCORRECT"), 4)
        self.assertIn(
            "If no error is found, output # VERDICT: CORRECT and do not "
            "rewrite the program.",
            feedback_prompt,
        )

    def test_self_refine_uses_text_refinement_outside_gsm(self):
        backend = QueueBackend(
            [
                "A mistaken solution. Final answer: 1",
                (
                    "<status>incorrect</status>\n"
                    "<feedback>The arithmetic is wrong.</feedback>\n"
                    "<revised_solution>Reasoning. Finally \\boxed{2}."
                    "</revised_solution>"
                ),
            ]
        )
        problem = Problem(
            "math-perturb",
            "one",
            "What is 1+1?",
            "2",
            answer_instruction="Put the final answer in `\\boxed{...}`.",
        )
        result = SelfRefineAdapter(self.self_refine_source).run(
            problem,
            backend,
            config={
                "max_refinements": 1,
                "text_feedback_inference_overrides": {"temperature": 0.7},
            },
        )

        self.assertEqual(result.text, r"Reasoning. Finally \boxed{2}.")
        self.assertEqual(
            result.metadata["prompt_profile"],
            "math-perturb-text-refinement",
        )
        for call in backend.calls:
            self.assertIn(r"Put the final answer in `\boxed{...}`.", call["messages"][0]["content"])
            self.assertNotIn("solution using Python", call["messages"][0]["content"])

    def test_self_refine_accepts_complete_text_with_imperfect_closing_tags(self):
        status, feedback, revised = SelfRefineAdapter._split_text_refinement(
            "<status>incorrect</status>\n"
            "<feedback>Fix the calculation.</wrong_tag>\n"
            "<revised_solution>Reasoning. \\boxed{2}"
        )

        self.assertEqual(status, "incorrect")
        self.assertEqual(feedback, "Fix the calculation.")
        self.assertEqual(revised, r"Reasoning. \boxed{2}")

    def test_self_refine_method_failures_are_scored_incorrect_and_continue(self):
        backend = QueueBackend(
            [
                "First candidate",
                "malformed refinement",
                "Second candidate",
                (
                    "<status>correct</status>\n"
                    "<feedback>The answer is correct.</feedback>\n"
                    "<revised_solution>Reasoning. \\boxed{3}</revised_solution>"
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlResultStore(Path(directory) / "results.jsonl")
            summary = EvaluationRunner().run(
                experiment=Experiment(
                    "exp-self-refine-text",
                    "math-perturb",
                    "self_refine",
                    "fake-model",
                ),
                plugin=TwoTextProblemPlugin(),
                context=DatasetContext(Path(directory)),
                method=SelfRefineAdapter(self.self_refine_source),
                backend=backend,
                store=store,
                method_config={
                    "max_refinements": 1,
                    "text_feedback_inference_overrides": {
                        "temperature": 0.7
                    },
                },
                fairness_policy=FairnessPolicy(
                    min_model_calls_per_problem=2,
                    max_model_calls_per_problem=2,
                    allowed_inference_overrides=("temperature",),
                ),
            )
            records = list(store.read())

        self.assertEqual(
            (summary.scored, summary.method_failed, summary.correct),
            (1, 1, 1),
        )
        self.assertEqual(
            records[0]["generation"]["text"], "malformed refinement"
        )
        self.assertEqual(records[0]["score"]["status"], "method_failed")
        self.assertFalse(records[0]["score"]["correct"])
        failure = records[0]["generation"]["method_details"]["method_failure"]
        self.assertEqual(failure["stage"], "feedback_parse")
        self.assertEqual(failure["exception_type"], "ValueError")
        self.assertTrue(records[1]["score"]["correct"])

    def test_self_refine_truncated_text_is_method_failure(self):
        backend = QueueBackend(
            [
                "First candidate",
                Generation(
                    "<status>incorrect</status>\n"
                    "<feedback>Incomplete",
                    finish_reason="length",
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlResultStore(Path(directory) / "results.jsonl")
            summary = EvaluationRunner().run(
                experiment=Experiment(
                    "exp-self-refine-truncated",
                    "math-perturb",
                    "self_refine",
                    "fake-model",
                ),
                plugin=TwoTextProblemPlugin(),
                context=DatasetContext(Path(directory)),
                method=SelfRefineAdapter(self.self_refine_source),
                backend=backend,
                store=store,
                method_config={
                    "max_refinements": 1,
                    "text_feedback_inference_overrides": {
                        "temperature": 0.7
                    },
                },
                fairness_policy=FairnessPolicy(
                    min_model_calls_per_problem=2,
                    max_model_calls_per_problem=2,
                    allowed_inference_overrides=("temperature",),
                ),
                batch_size=1,
            )
            record = next(store.read())

        self.assertEqual((summary.scored, summary.method_failed), (0, 1))
        self.assertEqual(record["score"]["status"], "method_failed")
        self.assertEqual(
            record["score"]["details"]["failure_stage"],
            "feedback_generation",
        )

    def test_self_refine_truncated_initial_text_is_revised(self):
        backend = QueueBackend(
            [
                Generation(
                    "An unfinished candidate solution...",
                    finish_reason="length",
                ),
                (
                    "<status>incorrect</status>\n"
                    "<feedback>The candidate was truncated.</feedback>\n"
                    "<revised_solution>Complete reasoning. "
                    "\\boxed{2}</revised_solution>"
                ),
            ]
        )
        problem = Problem(
            "math-perturb",
            "one",
            "What is 1+1?",
            "2",
            answer_instruction="Put the final answer in `\\boxed{...}`.",
        )

        result = SelfRefineAdapter(self.self_refine_source).run(
            problem,
            backend,
            config={
                "max_refinements": 1,
                "text_feedback_inference_overrides": {"temperature": 0.7},
            },
        )

        self.assertEqual(result.text, r"Complete reasoning. \boxed{2}")
        self.assertEqual(len(backend.calls), 2)
        self.assertIn(
            "Treat it as an incomplete draft and reconstruct a complete",
            backend.calls[1]["messages"][0]["content"],
        )

    def test_self_refine_filtered_initial_text_is_method_failure(self):
        backend = QueueBackend(
            [
                Generation(
                    "Blocked candidate",
                    finish_reason="content_filter",
                )
            ]
        )
        problem = Problem(
            "math-perturb",
            "one",
            "What is 1+1?",
            "2",
            answer_instruction="Put the final answer in `\\boxed{...}`.",
        )

        with self.assertRaises(MethodOutcomeError) as caught:
            SelfRefineAdapter(self.self_refine_source).run(
                problem,
                backend,
                config={
                    "max_refinements": 1,
                    "text_feedback_inference_overrides": {
                        "temperature": 0.7
                    },
                },
            )

        self.assertEqual(caught.exception.stage, "initial_generation")
        self.assertEqual(len(backend.calls), 1)

    def test_self_refine_gsm_token_limit_without_rewrite_keeps_input(self):
        backend = QueueBackend(
            [
                "def solution():\n    return 2",
                Generation(
                    "The calculations look good; I cannot identify an error.",
                    finish_reason="length",
                ),
            ]
        )
        summary, record = self.run_method(
            SelfRefineAdapter(self.self_refine_source),
            backend,
            method_config={
                "max_refinements": 1,
                "gsm_feedback_inference_overrides": {"temperature": 0.7},
            },
            policy=FairnessPolicy(
                min_model_calls_per_problem=2,
                max_model_calls_per_problem=2,
                allowed_inference_overrides=("temperature",),
            ),
        )

        self.assertEqual((summary.scored, summary.correct), (1, 1))
        self.assertEqual(record["generation"]["text"], r"Answer: \boxed{2}")
        self.assertEqual(
            record["generation"]["method_details"][
                "refinement_termination"
            ],
            "assumed_correct_after_feedback_token_limit",
        )

    def test_self_refine_gsm_correct_verdict_keeps_input(self):
        backend = QueueBackend(
            [
                "def solution():\n    return 2",
                "There is no error in the code.\n# VERDICT: CORRECT",
            ]
        )
        summary, record = self.run_method(
            SelfRefineAdapter(self.self_refine_source),
            backend,
            method_config={
                "max_refinements": 1,
                "gsm_feedback_inference_overrides": {"temperature": 0.7},
            },
            policy=FairnessPolicy(
                min_model_calls_per_problem=2,
                max_model_calls_per_problem=2,
                allowed_inference_overrides=("temperature",),
            ),
        )

        self.assertEqual((summary.scored, summary.correct), (1, 1))
        self.assertEqual(
            record["generation"]["method_details"][
                "refinement_termination"
            ],
            "feedback_declared_correct",
        )

    def test_self_refine_gsm_token_limit_with_correct_verdict_keeps_input(self):
        backend = QueueBackend(
            [
                "def solution():\n    return 2",
                Generation(
                    "There is no error.\n# VERDICT: CORRECT",
                    finish_reason="length",
                ),
            ]
        )
        summary, record = self.run_method(
            SelfRefineAdapter(self.self_refine_source),
            backend,
            method_config={
                "max_refinements": 1,
                "gsm_feedback_inference_overrides": {"temperature": 0.7},
            },
            policy=FairnessPolicy(
                min_model_calls_per_problem=2,
                max_model_calls_per_problem=2,
                allowed_inference_overrides=("temperature",),
            ),
        )

        self.assertEqual((summary.scored, summary.correct), (1, 1))
        self.assertEqual(
            record["generation"]["method_details"][
                "refinement_termination"
            ],
            "feedback_declared_correct",
        )

    def test_self_refine_gsm_other_feedback_finish_reason_fails(self):
        backend = QueueBackend(
            [
                "def solution():\n    return 2",
                Generation(
                    "# VERDICT: CORRECT",
                    finish_reason="content_filter",
                ),
            ]
        )
        summary, record = self.run_method(
            SelfRefineAdapter(self.self_refine_source),
            backend,
            method_config={
                "max_refinements": 1,
                "gsm_feedback_inference_overrides": {"temperature": 0.7},
            },
            policy=FairnessPolicy(
                min_model_calls_per_problem=2,
                max_model_calls_per_problem=2,
                allowed_inference_overrides=("temperature",),
            ),
        )

        self.assertEqual((summary.scored, summary.method_failed), (0, 1))
        self.assertEqual(
            record["score"]["details"]["failure_stage"],
            "feedback_generation",
        )

    def test_self_refine_gsm_stop_without_judgment_or_rewrite_fails(self):
        backend = QueueBackend(
            [
                "def solution():\n    return 2",
                "I am still reviewing the calculations.",
            ]
        )
        summary, record = self.run_method(
            SelfRefineAdapter(self.self_refine_source),
            backend,
            method_config={
                "max_refinements": 1,
                "gsm_feedback_inference_overrides": {"temperature": 0.7},
            },
            policy=FairnessPolicy(
                min_model_calls_per_problem=2,
                max_model_calls_per_problem=2,
                allowed_inference_overrides=("temperature",),
            ),
        )

        self.assertEqual((summary.scored, summary.method_failed), (0, 1))
        self.assertEqual(record["score"]["status"], "method_failed")
        self.assertEqual(
            record["score"]["details"]["failure_stage"],
            "feedback_parse",
        )

    def test_self_refine_gsm_token_limit_after_incorrect_verdict_fails(self):
        backend = QueueBackend(
            [
                "def solution():\n    return 2",
                Generation(
                    "The arithmetic is wrong.\n# VERDICT: INCORRECT",
                    finish_reason="length",
                ),
            ]
        )
        summary, record = self.run_method(
            SelfRefineAdapter(self.self_refine_source),
            backend,
            method_config={
                "max_refinements": 1,
                "gsm_feedback_inference_overrides": {"temperature": 0.7},
            },
            policy=FairnessPolicy(
                min_model_calls_per_problem=2,
                max_model_calls_per_problem=2,
                allowed_inference_overrides=("temperature",),
            ),
        )

        self.assertEqual((summary.scored, summary.method_failed), (0, 1))
        self.assertEqual(record["score"]["status"], "method_failed")
        self.assertEqual(
            record["score"]["details"]["failure_stage"],
            "feedback_generation",
        )

    def test_self_refine_gsm_stop_incorrect_verdict_requires_rewrite(self):
        backend = QueueBackend(
            [
                "def solution():\n    return 2",
                "The arithmetic is wrong.\n# VERDICT: INCORRECT",
            ]
        )
        summary, record = self.run_method(
            SelfRefineAdapter(self.self_refine_source),
            backend,
            method_config={
                "max_refinements": 1,
                "gsm_feedback_inference_overrides": {"temperature": 0.7},
            },
            policy=FairnessPolicy(
                min_model_calls_per_problem=2,
                max_model_calls_per_problem=2,
                allowed_inference_overrides=("temperature",),
            ),
        )

        self.assertEqual((summary.scored, summary.method_failed), (0, 1))
        self.assertEqual(
            record["score"]["details"]["failure_stage"],
            "feedback_parse",
        )

    def test_self_refine_invalid_final_program_is_scored_incorrect(self):
        backend = QueueBackend(
            [
                "def solution():\n    return missing_name",
                (
                    "The implementation has an unresolved name.\n"
                    "# VERDICT: INCORRECT\n"
                    "def solution():\n    return missing_name\n### END ###"
                ),
            ]
        )
        summary, record = self.run_method(
            SelfRefineAdapter(self.self_refine_source),
            backend,
            method_config={
                "max_refinements": 1,
                "gsm_feedback_inference_overrides": {"temperature": 0.7},
            },
            policy=FairnessPolicy(
                min_model_calls_per_problem=2,
                max_model_calls_per_problem=2,
                allowed_inference_overrides=("temperature",),
            ),
        )
        self.assertEqual(
            (summary.scored, summary.method_failed, summary.correct),
            (0, 1, 0),
        )
        self.assertEqual(record["score"]["status"], "method_failed")
        self.assertIn("missing_name", record["generation"]["text"])
        details = record["generation"]["method_details"]
        self.assertEqual(
            details["method_failure"]["stage"], "program_execution"
        )
        self.assertEqual(details["execution"]["status"], "error")

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

    def test_aflow_exhausted_programmer_is_a_scored_method_failure(self):
        workflow = {
            "id": "programmer-failure",
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
        self.assertEqual(workflow_call_bounds(workflow), (1, 2))
        backend = QueueBackend(
            [
                "def solution():\n    return missing_one",
                "def solution():\n    return missing_two",
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
                    "optimization_cost": 0,
                    "evaluation_data_used": False,
                    "frozen": True,
                },
            },
            policy=FairnessPolicy(
                min_model_calls_per_problem=1,
                max_model_calls_per_problem=2,
            ),
        )
        self.assertEqual((summary.scored, summary.correct), (1, 0))
        self.assertEqual(record["generation"]["text"], "")
        details = record["generation"]["method_details"]
        self.assertEqual(
            details["method_failure"]["stage"], "program_execution"
        )
        self.assertEqual(record["generation"]["model_calls"], 2)

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
