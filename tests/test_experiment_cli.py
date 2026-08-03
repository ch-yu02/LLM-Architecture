from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.run_experiments import (
    _dataset_selection,
    _experiment_identity,
    _find_resume_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get(
        "MATH_BENCHMARK_DATA_ROOT",
        ROOT.parent / "math-benchmark-data",
    )
)
PYTHON = ROOT / ".venv-checkers" / "bin" / "python"


class ExperimentCliTests(unittest.TestCase):
    def test_resume_identity_ignores_runner_revision_and_prefers_superset(self):
        old = {
            "model": {"model": "qwen", "enable_thinking": False},
            "method": "direct",
            "dataset": "harp-small",
            "dataset_artifacts": {"data.jsonl": "same-blob"},
            "revisions": {
                "runner_git": "old",
                "runner_tree": "old-tree",
                "data": {"revision": "old-data-commit"},
                "method_source": None,
            },
        }
        current = {
            **old,
            "model": {
                **old["model"],
                "thinking_type": None,
                "reasoning_effort": None,
                "omit_sampling_parameters": False,
            },
            "revisions": {
                **old["revisions"],
                "runner_git": "new",
                "runner_tree": "new-tree",
            },
        }
        artifacts = {"data.jsonl": "same-blob"}
        identity = _experiment_identity(
            current,
            dataset_artifacts=artifacts,
        )
        self.assertEqual(
            _experiment_identity(old, dataset_artifacts=artifacts),
            identity,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)

            def create_result(name, fingerprint, problem_ids):
                result = output_root / name
                result.mkdir()
                (result / "experiment.json").write_text(
                    json.dumps(
                        {
                            "fingerprint": fingerprint,
                            "configuration": old,
                        }
                    ),
                    encoding="utf-8",
                )
                records = [
                    {
                        "record_type": "experiment",
                        "format_version": 2,
                    },
                    *(
                        {
                            "record_type": "sample",
                            "problem": {"id": problem_id},
                        }
                        for problem_id in problem_ids
                    ),
                ]
                (result / "records.jsonl").write_text(
                    "\n".join(json.dumps(item) for item in records) + "\n",
                    encoding="utf-8",
                )
                return result

            classified = output_root / "local" / "direct"
            classified.mkdir(parents=True)
            complete = create_result(
                "local/direct/harp-small__r001__old",
                "a" * 64,
                ("p1", "p2", "p3"),
            )
            create_result(
                "local__direct__harp-small__r001__new",
                "b" * 64,
                ("p1",),
            )
            self.assertEqual(
                _find_resume_experiment(
                    output_root,
                    identity=identity,
                    data_root=DATA_ROOT,
                    dataset_paths=(),
                ),
                (complete, "a" * 64, old),
            )

        changed_model = json.loads(json.dumps(current))
        changed_model["model"]["model"] = "another-model"
        self.assertNotEqual(
            identity,
            _experiment_identity(
                changed_model,
                dataset_artifacts=artifacts,
            ),
        )
        self.assertNotEqual(
            identity,
            _experiment_identity(
                current,
                dataset_artifacts={"data.jsonl": "changed-blob"},
            ),
        )

    def test_dataset_selection_keeps_harp_small_explicit(self):
        self.assertEqual(
            _dataset_selection("all"),
            [
                "gsm1k",
                "math-perturb",
                "harp",
                "u-math-text-only",
                "mathconstruct",
            ],
        )
        self.assertEqual(_dataset_selection("harp-small"), ["harp-small"])

    def test_local_api_run_resumes_and_separates_changed_config(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model_path = temporary / "model.toml"
            output_root = temporary / "results"
            fake_package = temporary / "fake-packages" / "openai"
            fake_package.mkdir(parents=True)
            (fake_package / "__init__.py").write_text(
                """
import os
from types import SimpleNamespace


class OpenAI:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    @staticmethod
    def _create(**kwargs):
        with open(os.environ["LOCAL_CALL_LOG"], "a", encoding="utf-8") as stream:
            stream.write(f"{kwargs.get('seed')}\\n")
        content = "\\n".join(
            message["content"] for message in kwargs.get("messages", [])
        )
        if "SOLUTION TO EVALUATE:" in content:
            response_text = "The candidate does not match.\\nNo"
        elif "What is the error?" in content:
            response_text = (
                "It is correct.\\n"
                "def solution():\\n"
                "    return 0\\n"
                "### END"
            )
        elif (
            "# solution using Python:" in content
            or "# solution in Python:" in content
            or "only write code blocks" in content
        ):
            response_text = "def solution():\\n    return 0"
        else:
            response_text = r"\\boxed{0}"
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=3,
            total_tokens=13,
            completion_tokens_details=None,
            prompt_tokens_details=None,
        )
        message = SimpleNamespace(content=response_text)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(
            id="chatcmpl-local",
            usage=usage,
            choices=[choice],
        )
""".lstrip(),
                encoding="utf-8",
            )
            call_log = temporary / "calls.txt"
            model_path.write_text(
                "\n".join(
                    (
                        'profile = "local-test"',
                        'provider = "local"',
                        'api_type = "openai"',
                        'model = "local-test-model"',
                        'base_url = "http://local.invalid/v1"',
                        'api_key_env = "LOCAL_TEST_API_KEY"',
                        "enable_thinking = false",
                        "temperature = 0.0",
                        "top_p = 1.0",
                        "max_output_tokens = 64",
                        "request_timeout_seconds = 10",
                        "max_retries = 1",
                        "min_request_interval_seconds = 0",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            command = [
                str(PYTHON),
                str(ROOT / "scripts" / "run_experiments.py"),
                "--model",
                str(model_path),
                "--methods",
                "direct",
                "--datasets",
                "gsm1k",
                "--batch-size",
                "1",
                "--concurrency",
                "2",
                "--seed",
                "1234",
                "--data-root",
                str(DATA_ROOT),
                "--output-root",
                str(output_root),
                "--checker-python",
                str(PYTHON),
                "--skip-preflight",
                "--yes",
            ]
            environment = {
                **os.environ,
                "LOCAL_TEST_API_KEY": "local-test-key",
                "LOCAL_CALL_LOG": str(call_log),
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(fake_package.parent),
                        os.environ.get("PYTHONPATH", ""),
                    )
                ),
            }

            first = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr or first.stdout)
            self.assertEqual(len(call_log.read_text().splitlines()), 1)
            self.assertEqual(call_log.read_text().splitlines()[0], "1234")
            directories = [
                path.parent for path in output_root.rglob("experiment.json")
            ]
            self.assertEqual(len(directories), 1)
            self.assertEqual(
                directories[0].parent,
                output_root / "local-test" / "direct",
            )
            self.assertRegex(
                directories[0].name,
                r"__\d{8}T\d{6}Z__[0-9a-f]{16}$",
            )
            records_path = directories[0] / "records.jsonl"
            record_lines = [
                json.loads(line) for line in records_path.read_text().splitlines()
            ]
            self.assertEqual(len(record_lines), 2)
            self.assertEqual(record_lines[0]["record_type"], "experiment")
            self.assertNotIn("experiment", record_lines[0])
            self.assertEqual(record_lines[1]["record_type"], "sample")
            self.assertNotIn("experiment", record_lines[1])
            self.assertEqual(
                record_lines[1]["problem"]["id"],
                "gsm1k-0000",
            )
            self.assertTrue(record_lines[1]["problem"]["prompt"])
            self.assertEqual(
                record_lines[1]["problem"]["reference_answer"],
                "133",
            )
            self.assertEqual(
                record_lines[1]["problem"]["metadata"]["split"],
                "test",
            )
            self.assertNotIn("value", record_lines[1]["score"])
            self.assertNotIn("error", record_lines[1]["score"])
            api_calls = (directories[0] / "api_calls.jsonl").read_text().splitlines()
            self.assertEqual(len(api_calls), 1)
            manifest = json.loads(
                (directories[0] / "experiment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["configuration"]["environment"]["packages"]["openai"],
                "2.49.0",
            )
            self.assertEqual(
                manifest["configuration"]["inference_config"]["seed"],
                1234,
            )
            self.assertEqual(
                manifest["configuration"]["dataset_scope"],
                "test",
            )
            self.assertIn(
                "Answer: <number>",
                manifest["configuration"]["dataset_answer_instruction"],
            )

            continued = subprocess.run(
                [
                    *command,
                    "--batch-size",
                    "2",
                    "--concurrency",
                    "3",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                continued.returncode,
                0,
                continued.stderr or continued.stdout,
            )
            self.assertEqual(len(call_log.read_text().splitlines()), 3)
            self.assertEqual(
                len(list(output_root.rglob("experiment.json"))),
                1,
            )
            continued_lines = [
                json.loads(line) for line in records_path.read_text().splitlines()
            ]
            self.assertEqual(len(continued_lines), 4)
            self.assertEqual(
                [
                    item["problem"]["id"]
                    for item in continued_lines[1:]
                ],
                ["gsm1k-0000", "gsm1k-0001", "gsm1k-0002"],
            )
            summary = json.loads(
                (directories[0] / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["run"]["resumed"], 1)
            self.assertEqual(summary["run"]["attempted"], 2)
            self.assertEqual(summary["invocation"]["batch_size"], 2)
            self.assertEqual(summary["invocation"]["concurrency"], 3)
            self.assertEqual(summary["aggregate"]["records"], 3)

            changed = subprocess.run(
                [*command, "--temperature", "0.2"],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                changed.returncode,
                0,
                changed.stderr or changed.stdout,
            )
            self.assertEqual(len(call_log.read_text().splitlines()), 4)
            self.assertEqual(
                len(list(output_root.rglob("experiment.json"))),
                2,
            )

            repeated_output = temporary / "repeat-results"
            repeated = subprocess.run(
                [
                    *command,
                    "--output-root",
                    str(repeated_output),
                    "--repeats",
                    "2",
                    "--seed-mode",
                    "increment",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                repeated.returncode,
                0,
                repeated.stderr or repeated.stdout,
            )
            self.assertEqual(
                call_log.read_text().splitlines()[-2:],
                ["1234", "1235"],
            )
            repeated_manifests = [
                json.loads(
                    (path / "experiment.json").read_text(encoding="utf-8")
                )
                for manifest_path in repeated_output.rglob("experiment.json")
                for path in (manifest_path.parent,)
            ]
            self.assertEqual(len(repeated_manifests), 2)
            self.assertEqual(
                sorted(
                    manifest["configuration"]["inference_config"]["seed"]
                    for manifest in repeated_manifests
                ),
                [1234, 1235],
            )
            self.assertTrue(
                all(
                    manifest["configuration"]["repeat_seed"]["mode"]
                    == "increment"
                    for manifest in repeated_manifests
                )
            )
            repeated_continued = subprocess.run(
                [
                    *command,
                    "--output-root",
                    str(repeated_output),
                    "--repeats",
                    "2",
                    "--seed-mode",
                    "increment",
                    "--batch-size",
                    "2",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                repeated_continued.returncode,
                0,
                repeated_continued.stderr or repeated_continued.stdout,
            )
            self.assertEqual(
                call_log.read_text().splitlines()[-4:],
                ["1234", "1234", "1235", "1235"],
            )
            self.assertEqual(
                len(
                    [
                        path
                        for path in repeated_output.rglob("experiment.json")
                    ]
                ),
                2,
            )
            for manifest_path in repeated_output.rglob("experiment.json"):
                path = manifest_path.parent
                records = [
                    json.loads(line)
                    for line in (path / "records.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                self.assertEqual(len(records), 4)

            matrix_output = temporary / "matrix-results"
            matrix_command = [
                str(PYTHON),
                str(ROOT / "scripts" / "run_experiments.py"),
                "--model",
                str(model_path),
                "--methods",
                "all",
                "--datasets",
                "gsm1k,math-perturb,harp,mathconstruct",
                "--batch-size",
                "1",
                "--data-root",
                str(DATA_ROOT),
                "--output-root",
                str(matrix_output),
                "--checker-python",
                str(PYTHON),
                "--skip-preflight",
                "--yes",
            ]
            matrix = subprocess.run(
                matrix_command,
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(matrix.returncode, 0, matrix.stderr or matrix.stdout)
            matrix_directories = [
                path.parent for path in matrix_output.rglob("experiment.json")
            ]
            self.assertEqual(len(matrix_directories), 16)
            summaries = [
                json.loads(
                    (path / "summary.json").read_text(encoding="utf-8")
                )
                for path in matrix_directories
            ]
            self.assertTrue(
                all(summary["aggregate"]["errors"] == 0 for summary in summaries)
            )

            harp_small_output = temporary / "harp-small-results"
            harp_small = subprocess.run(
                [
                    *command,
                    "--datasets",
                    "harp-small",
                    "--output-root",
                    str(harp_small_output),
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                harp_small.returncode,
                0,
                harp_small.stderr or harp_small.stdout,
            )
            harp_small_directory = next(
                harp_small_output.rglob("experiment.json")
            ).parent
            harp_small_manifest = json.loads(
                (harp_small_directory / "experiment.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                harp_small_manifest["configuration"]["dataset"],
                "harp-small",
            )
            self.assertEqual(
                harp_small_manifest["configuration"]["dataset_scope"],
                "small-test:harp_small_test_v1",
            )
            harp_small_records = [
                json.loads(line)
                for line in (harp_small_directory / "records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                harp_small_records[1]["problem"]["metadata"]["split"],
                "small_test",
            )
            self.assertEqual(
                harp_small_records[1]["problem"]["metadata"]["source_split"],
                "test",
            )
            self.assertEqual(
                harp_small_records[1]["problem"]["metadata"]["subset_id"],
                "harp_small_test_v1",
            )

            locked_u_math_output = temporary / "locked-u-math"
            locked_u_math = subprocess.run(
                [
                    *command,
                    "--datasets",
                    "u-math-text-only",
                    "--output-root",
                    str(locked_u_math_output),
                    "--dry-run",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                locked_u_math.returncode,
                0,
                locked_u_math.stderr or locked_u_math.stdout,
            )
            self.assertIn("qwen3.7-flash-2026-07-15", locked_u_math.stdout)
            self.assertIn("fixed", locked_u_math.stdout)
            self.assertFalse(locked_u_math_output.exists())

            error_output = temporary / "error-results"
            error_run = subprocess.run(
                [
                    *command,
                    "--output-root",
                    str(error_output),
                    "--method-param",
                    "direct.unused=true",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(error_run.returncode, 1)
            self.assertIn("ERROR direct × gsm1k", error_run.stdout)
            self.assertIn("stopped at the first failed sample", error_run.stdout)
            error_directory = next(error_output.rglob("experiment.json")).parent
            self.assertFalse((error_directory / "records.jsonl").exists())
            diagnostics = [
                json.loads(line)
                for line in (error_directory / "errors.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(diagnostics), 1)
            self.assertEqual(diagnostics[0]["problem_id"], "gsm1k-0000")
            self.assertEqual(
                diagnostics[0]["details"]["exception_type"],
                "ValueError",
            )
            self.assertEqual(
                diagnostics[0]["details"]["failure_stage"],
                "method",
            )
            self.assertIn("Traceback", diagnostics[0]["details"]["traceback"])

    def test_sensitive_method_parameter_is_rejected_before_artifact_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "results"
            result = subprocess.run(
                [
                    str(PYTHON),
                    str(ROOT / "scripts" / "run_experiments.py"),
                    "--methods",
                    "self_refine",
                    "--datasets",
                    "gsm1k",
                    "--method-param",
                    'self_refine.api_key="must-not-write"',
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
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("sensitive keys", result.stdout)
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
