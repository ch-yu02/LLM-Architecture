from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmark_experiments.artifacts import configuration_fingerprint
from scripts.run_mu_math import (
    _experiment_identity,
    _find_resume_experiment,
    _metrics,
    _resolve_judge_path,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get(
        "MATH_BENCHMARK_DATA_ROOT",
        ROOT.parent / "math-benchmark-data",
    )
)
PYTHON = ROOT / ".venv-checkers" / "bin" / "python"


class MuMathCliTests(unittest.TestCase):
    def test_resume_identity_ignores_runner_revision_and_neutral_new_fields(self):
        old = {
            "dataset": "mu-math",
            "dataset_blob": "fixed-data-blob",
            "judge_profile": {"model": "qwen", "enable_thinking": False},
            "revisions": {
                "runner_git": "old",
                "runner_tree": "old-tree",
                "data": {"revision": "fixed-data"},
            },
        }
        current = {
            "dataset": "mu-math",
            "dataset_blob": "fixed-data-blob",
            "judge_profile": {
                "model": "qwen",
                "enable_thinking": False,
                "thinking_type": None,
                "reasoning_effort": None,
                "omit_sampling_parameters": False,
            },
            "revisions": {
                "runner_git": "new",
                "runner_tree": "new-tree",
                "data": {"revision": "fixed-data"},
            },
        }
        identity = _experiment_identity(current)
        self.assertEqual(_experiment_identity(old), identity)

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            legacy = output_root / "candidate" / "judge" / "official-test__legacy"
            legacy.mkdir(parents=True)
            legacy_fingerprint = configuration_fingerprint(old)
            (legacy / "experiment.json").write_text(
                json.dumps(
                    {
                        "fingerprint": legacy_fingerprint,
                        "configuration": old,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                _find_resume_experiment(
                    output_root,
                    identity=identity,
                    data_root=DATA_ROOT,
                    dataset_relative_path=Path(
                        "data/processed/mu_math.jsonl"
                    ),
                ),
                (legacy, legacy_fingerprint, old),
            )

        changed_data = json.loads(json.dumps(current))
        changed_data["dataset_blob"] = "other-data-blob"
        self.assertNotEqual(
            _experiment_identity(old),
            _experiment_identity(changed_data),
        )

    def test_bundled_candidate_profiles_resolve_by_short_name_and_model(self):
        for value, filename in (
            ("gemini36_flash", "gemini36_flash.toml"),
            ("gemini-3.6-flash", "gemini36_flash.toml"),
            ("qwen37_flash", "qwen37_flash.toml"),
            ("qwen3.7-flash-2026-07-15", "qwen37_flash.toml"),
            ("deepseek_v4_pro", "deepseek_v4_pro.toml"),
            ("deepseek-v4-pro", "deepseek_v4_pro.toml"),
            ("deepseek_v4_flash", "deepseek_v4_flash.toml"),
            ("deepseek-v4-flash", "deepseek_v4_flash.toml"),
        ):
            with self.subTest(value=value):
                self.assertEqual(_resolve_judge_path(value).name, filename)

    def test_metrics_treat_inconclusive_as_an_error_without_binary_mapping(self):
        records = [
            {
                "status": "scored",
                "label": label,
                "judge": {"verdict": verdict},
            }
            for label, verdict in (
                (True, "Yes"),
                (True, "No"),
                (True, "Inconclusive"),
                (False, "No"),
                (False, "Yes"),
                (False, "Inconclusive"),
            )
        ]
        metrics = _metrics(records)
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["inconclusive"], 2)
        self.assertEqual(
            metrics["confusion"],
            {
                "tp": 1,
                "tn": 1,
                "fp": 1,
                "fn": 1,
                "positive_inconclusive": 1,
                "negative_inconclusive": 1,
            },
        )
        self.assertAlmostEqual(metrics["metrics"]["macro_f1"], 0.4)
        self.assertAlmostEqual(metrics["metrics"]["tpr"], 1 / 3)
        self.assertAlmostEqual(metrics["metrics"]["tnr"], 1 / 3)
        self.assertAlmostEqual(metrics["metrics"]["ppv"], 0.5)
        self.assertAlmostEqual(metrics["metrics"]["npv"], 0.5)

    def test_local_judge_run_resumes_and_reports_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            profile_path = temporary / "judge.toml"
            output_root = temporary / "results"
            fake_package = temporary / "fake-packages" / "openai"
            fake_package.mkdir(parents=True)
            (fake_package / "__init__.py").write_text(
                """
import os
import threading
import time
from types import SimpleNamespace


_lock = threading.Lock()
_active = 0
_max_active = 0


class OpenAI:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    @staticmethod
    def _create(**kwargs):
        global _active, _max_active
        if os.environ.get("LOCAL_FORCE_ERROR"):
            raise ValueError("forced judge failure")
        with _lock:
            _active += 1
            _max_active = max(_max_active, _active)
            with open(
                os.environ["LOCAL_CALL_LOG"], "a", encoding="utf-8"
            ) as stream:
                stream.write(f"{kwargs.get('seed')}\\n")
            with open(
                os.environ["LOCAL_MAX_ACTIVE"], "w", encoding="utf-8"
            ) as stream:
                stream.write(str(_max_active))
        try:
            time.sleep(0.05)
            usage = SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=3,
                total_tokens=13,
                completion_tokens_details=None,
                prompt_tokens_details=None,
            )
            message = SimpleNamespace(content="The candidate is wrong.\\nNo")
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(
                id="chatcmpl-mu-math-local",
                usage=usage,
                choices=[choice],
            )
        finally:
            with _lock:
                _active -= 1
""".lstrip(),
                encoding="utf-8",
            )
            profile_path.write_text(
                "\n".join(
                    (
                        'profile = "local-mu-math-judge"',
                        'provider = "local"',
                        'api_type = "openai"',
                        'model = "local-judge"',
                        'base_url = "http://local.invalid/v1"',
                        'api_key_env = "LOCAL_TEST_API_KEY"',
                        "enable_thinking = false",
                        "temperature = 0.0",
                        "top_p = 1.0",
                        "max_output_tokens = 64",
                        "seed = 42",
                        "request_timeout_seconds = 10",
                        "max_retries = 1",
                        "min_request_interval_seconds = 0",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            call_log = temporary / "calls.txt"
            max_active_log = temporary / "max-active.txt"
            command = [
                str(PYTHON),
                str(ROOT / "scripts" / "run_mu_math.py"),
                "--judge",
                str(profile_path),
                "--batch-size",
                "1",
                "--concurrency",
                "2",
                "--data-root",
                str(DATA_ROOT),
                "--output-root",
                str(output_root),
                "--skip-preflight",
                "--yes",
            ]
            environment = {
                **os.environ,
                "LOCAL_TEST_API_KEY": "local-test-key",
                "LOCAL_CALL_LOG": str(call_log),
                "LOCAL_MAX_ACTIVE": str(max_active_log),
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
            self.assertEqual(call_log.read_text().splitlines(), ["42"])
            experiment_dirs = [
                path.parent for path in output_root.rglob("experiment.json")
            ]
            self.assertEqual(len(experiment_dirs), 1)
            experiment_dir = experiment_dirs[0]
            self.assertEqual(
                experiment_dir.parent,
                output_root / "local-mu-math-judge" / "judge",
            )
            self.assertRegex(
                experiment_dir.name,
                r"__\d{8}T\d{6}Z__[0-9a-f]{16}$",
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
            self.assertEqual(
                call_log.read_text().splitlines(),
                ["42", "42", "42"],
            )
            self.assertEqual(max_active_log.read_text(), "2")
            self.assertEqual(
                len(list(output_root.rglob("experiment.json"))),
                1,
            )

            failed_output = temporary / "failed-results"
            failed_command = [
                *command,
                "--output-root",
                str(failed_output),
            ]
            failed = subprocess.run(
                failed_command,
                cwd=ROOT,
                env={**environment, "LOCAL_FORCE_ERROR": "1"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(failed.returncode, 1, failed.stderr or failed.stdout)
            self.assertIn("stopped at the first failed sample", failed.stdout)
            failed_directory = next(failed_output.rglob("experiment.json")).parent
            self.assertFalse((failed_directory / "records.jsonl").exists())
            self.assertEqual(
                len((failed_directory / "errors.jsonl").read_text().splitlines()),
                1,
            )

            recovered = subprocess.run(
                failed_command,
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                recovered.returncode,
                0,
                recovered.stderr or recovered.stdout,
            )
            recovered_lines = (
                failed_directory / "records.jsonl"
            ).read_text().splitlines()
            self.assertEqual(len(recovered_lines), 2)

            records = [
                json.loads(line)
                for line in (experiment_dir / "records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(records), 4)
            self.assertEqual(records[0]["record_type"], "experiment")
            self.assertEqual(
                [record["id"] for record in records[1:]],
                ["mu-math-0000", "mu-math-0001", "mu-math-0002"],
            )
            for record in records[1:]:
                self.assertTrue(record["problem"])
                self.assertTrue(record["reference_answer"])
                self.assertTrue(record["candidate"]["text"])
                self.assertEqual(record["judge"]["verdict"], "No")
                self.assertEqual(record["judge"]["usage"]["total_tokens"], 13)

            summary = json.loads(
                (experiment_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["invocation"]["completed_before"], 1)
            self.assertEqual(summary["invocation"]["new_records"], 2)
            self.assertEqual(summary["invocation"]["concurrency"], 3)
            self.assertEqual(summary["aggregate"]["records"], 3)
            self.assertEqual(summary["aggregate"]["scored"], 3)
            self.assertEqual(summary["aggregate"]["errors"], 0)
            self.assertEqual(summary["aggregate"]["correct"], 3)
            self.assertEqual(
                summary["aggregate"]["metrics_percent"]["macro_f1"],
                50.0,
            )
            self.assertEqual(
                set(summary["by_solution_model"]),
                {
                    "GPT-4o",
                    "Gemini-1.5-Pro",
                    "Llama-3.1-70B-Instruct",
                },
            )
            self.assertEqual(summary["api"]["attempts"], 3)
            self.assertEqual(summary["api"]["successes"], 3)

            manifest = json.loads(
                (experiment_dir / "experiment.json").read_text(
                    encoding="utf-8"
                )
            )
            configuration = manifest["configuration"]
            self.assertEqual(configuration["dataset"], "mu-math")
            self.assertEqual(
                configuration["protocol"]["name"],
                "u-math-manual-cot-self-verdict",
            )
            self.assertEqual(
                configuration["inference_config"]["seed"],
                42,
            )

    def test_dry_run_creates_no_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "results"
            result = subprocess.run(
                [
                    str(PYTHON),
                    str(ROOT / "scripts" / "run_mu_math.py"),
                    "--data-root",
                    str(DATA_ROOT),
                    "--output-root",
                    str(output_root),
                    "--batch-size",
                    "5",
                    "--dry-run",
                    "--yes",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            self.assertIn("1,084 official test rows", result.stdout)
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
