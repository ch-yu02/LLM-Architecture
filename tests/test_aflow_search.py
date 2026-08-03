from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmark_aflow import (
    build_optimizer_prompt,
    candidate_rank,
    parse_workflow_proposal,
)
from scripts.run_experiments import ALL_METHODS


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DATA_ROOT = WORKSPACE / "math-benchmark-data"
PYTHON = ROOT / ".venv-checkers" / "bin" / "python"


class AFlowSearchTests(unittest.TestCase):
    def test_optimizer_proposal_parser_and_rank(self):
        workflow = {
            "id": "candidate",
            "nodes": [
                {
                    "id": "answer",
                    "operator": "custom",
                    "instruction": "Solve carefully.",
                    "inputs": [],
                }
            ],
            "output": "answer",
        }
        parsed, rationale = parse_workflow_proposal(
            "```json\n"
            + json.dumps({"rationale": "smaller", "workflow": workflow})
            + "\n```"
        )
        self.assertEqual(parsed, workflow)
        self.assertEqual(rationale, "smaller")
        self.assertGreater(
            candidate_rank({"accuracy": 0.8, "workflow": workflow}),
            candidate_rank(
                {
                    "accuracy": 0.8,
                    "workflow": {
                        "id": "larger",
                        "nodes": [
                            {
                                "id": "a",
                                "operator": "custom",
                                "instruction": "",
                                "inputs": [],
                            },
                            {
                                "id": "b",
                                "operator": "custom",
                                "instruction": "",
                                "inputs": [],
                            },
                            {
                                "id": "pick",
                                "operator": "ensemble",
                                "inputs": ["a", "b"],
                            },
                        ],
                        "output": "pick",
                    },
                }
            ),
        )

    def test_optimizer_prompt_contains_validation_boundary(self):
        workflow = {
            "id": "initial",
            "nodes": [
                {
                    "id": "answer",
                    "operator": "custom",
                    "instruction": "",
                    "inputs": [],
                }
            ],
            "output": "answer",
        }
        candidate = {"round": 0, "accuracy": 0.5, "workflow": workflow}
        prompt = build_optimizer_prompt(
            dataset="gsm1k",
            answer_instruction="Answer: <number>",
            best_candidate=candidate,
            history=[candidate],
            failure_examples=[],
            round_number=1,
        )
        self.assertIn("fixed validation split", prompt)
        self.assertIn("never receive or use test examples", prompt)
        self.assertIn("Answer: <number>", prompt)

    def test_general_matrix_excludes_aflow(self):
        self.assertEqual(
            ALL_METHODS,
            ("direct", "zero_shot_cot", "pal", "self_refine"),
        )
        result = subprocess.run(
            [
                str(PYTHON),
                str(ROOT / "scripts" / "run_experiments.py"),
                "--methods",
                "aflow",
                "--dry-run",
                "--yes",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown methods", result.stdout)

    def test_search_artifact_runs_bound_test_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            output_root = temporary / "aflow-results"
            model_path = temporary / "model.toml"
            fake_package = temporary / "fake-packages" / "openai"
            fake_package.mkdir(parents=True)
            (fake_package / "__init__.py").write_text(
                """
from types import SimpleNamespace


class OpenAI:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    @staticmethod
    def _create(**kwargs):
        content = "\\n".join(
            message["content"] for message in kwargs.get("messages", [])
        )
        if "You are the graph optimizer in AFlow" in content:
            response_text = '''{
              "rationale": "add a careful task instruction",
              "workflow": {
                "id": "optimized-careful",
                "nodes": [
                  {
                    "id": "answer",
                    "operator": "custom",
                    "instruction": "Solve carefully.",
                    "inputs": []
                  }
                ],
                "output": "answer"
              }
            }'''
        elif "Solve carefully." in content:
            response_text = r"\\boxed{100}"
        else:
            response_text = r"\\boxed{0}"
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=3,
            total_tokens=13,
            completion_tokens_details=None,
            prompt_tokens_details=None,
        )
        return SimpleNamespace(
            id="chatcmpl-aflow-test",
            usage=usage,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response_text),
                    finish_reason="stop",
                )
            ],
        )
""".lstrip(),
                encoding="utf-8",
            )
            model_path.write_text(
                "\n".join(
                    (
                        'profile = "local-aflow"',
                        'provider = "local"',
                        'api_type = "openai"',
                        'model = "local-aflow-model"',
                        'base_url = "http://local.invalid/v1"',
                        'api_key_env = "LOCAL_AFLOW_API_KEY"',
                        "enable_thinking = false",
                        "temperature = 0.0",
                        "top_p = 1.0",
                        "max_output_tokens = 256",
                        "seed = 7",
                        "request_timeout_seconds = 10",
                        "max_retries = 1",
                        "min_request_interval_seconds = 0",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            environment = {
                **os.environ,
                "LOCAL_AFLOW_API_KEY": "local-test-key",
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(fake_package.parent),
                        os.environ.get("PYTHONPATH", ""),
                    )
                ),
            }
            base = [
                str(PYTHON),
                str(ROOT / "scripts" / "run_aflow.py"),
                "--dataset",
                "gsm1k",
                "--model",
                str(model_path),
                "--data-root",
                str(DATA_ROOT),
                "--output-root",
                str(output_root),
                "--checker-python",
                str(PYTHON),
                "--skip-preflight",
                "--yes",
            ]
            search = subprocess.run(
                [
                    *base,
                    "--mode",
                    "search",
                    "--optimizer-model",
                    str(model_path),
                    "--search-rounds",
                    "1",
                    "--validation-size",
                    "1",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(search.returncode, 0, search.stderr or search.stdout)
            workflow_paths = list(output_root.rglob("workflow.json"))
            self.assertEqual(len(workflow_paths), 1)
            workflow_path = workflow_paths[0]
            self.assertEqual(workflow_path.parent.parent.name, "aflow")
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            self.assertEqual(workflow["dataset"], "gsm1k")
            self.assertEqual(workflow["search"]["selected_round"], 1)
            self.assertEqual(workflow["search"]["validation_accuracy"], 1.0)
            self.assertTrue(workflow["workflow_artifact"]["frozen"])
            self.assertFalse(
                workflow["workflow_artifact"]["evaluation_data_used"]
            )
            classified_search = workflow_path.parent.with_name(
                "manually-classified-search"
            )
            workflow_path.parent.rename(classified_search)
            workflow_path = classified_search / "workflow.json"
            optimizer_calls_before = (
                workflow_path.parent / "optimizer_api_calls.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            resumed_search = subprocess.run(
                [
                    *base,
                    "--mode",
                    "search",
                    "--optimizer-model",
                    str(model_path),
                    "--search-rounds",
                    "1",
                    "--validation-size",
                    "1",
                    "--concurrency",
                    "2",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                resumed_search.returncode,
                0,
                resumed_search.stderr or resumed_search.stdout,
            )
            self.assertEqual(
                len(list(output_root.rglob("workflow.json"))),
                1,
            )
            self.assertEqual(
                (
                    workflow_path.parent / "optimizer_api_calls.jsonl"
                ).read_text(encoding="utf-8").splitlines(),
                optimizer_calls_before,
            )

            test = subprocess.run(
                [
                    *base,
                    "--mode",
                    "test",
                    "--workflow",
                    str(workflow_path),
                    "--batch-size",
                    "1",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(test.returncode, 0, test.stderr or test.stdout)
            test_directories = [
                path.parent for path in output_root.rglob("experiment.json")
            ]
            self.assertEqual(len(test_directories), 1)
            manifest = json.loads(
                (test_directories[0] / "experiment.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["configuration"]["dataset"], "gsm1k")
            self.assertEqual(manifest["configuration"]["method"], "aflow")
            records = (
                test_directories[0] / "records.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 2)
            classified_test = test_directories[0].with_name(
                "manually-classified-test"
            )
            test_directories[0].rename(classified_test)
            test_directories[0] = classified_test

            continued_test = subprocess.run(
                [
                    *base,
                    "--mode",
                    "test",
                    "--workflow",
                    str(workflow_path),
                    "--batch-size",
                    "2",
                    "--concurrency",
                    "2",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                continued_test.returncode,
                0,
                continued_test.stderr or continued_test.stdout,
            )
            self.assertEqual(
                len(list(output_root.rglob("experiment.json"))),
                1,
            )
            continued_records = (
                test_directories[0] / "records.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(continued_records), 4)

            wrong_dataset = subprocess.run(
                [
                    str(PYTHON),
                    str(ROOT / "scripts" / "run_aflow.py"),
                    "--mode",
                    "test",
                    "--dataset",
                    "harp",
                    "--model",
                    str(model_path),
                    "--workflow",
                    str(workflow_path),
                    "--data-root",
                    str(DATA_ROOT),
                    "--output-root",
                    str(output_root),
                    "--checker-python",
                    str(PYTHON),
                    "--dry-run",
                    "--yes",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(wrong_dataset.returncode, 1)
            self.assertIn("does not match", wrong_dataset.stdout)


if __name__ == "__main__":
    unittest.main()
