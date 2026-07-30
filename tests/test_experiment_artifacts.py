from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_experiments.artifacts import (
    ExperimentArtifactError,
    clean_git_repository_state,
    configuration_fingerprint,
    ensure_experiment_manifest,
    experiment_directory,
    summarize_api_calls,
    summarize_records,
)
from model_backends import (
    ModelConfigurationError,
    OpenAICompatibleBackend,
    load_model_profile,
    validate_inference_config,
)
from scripts.run_experiments import (
    _load_locked_u_math_judge,
    _repeat_inference_config,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get(
        "MATH_BENCHMARK_DATA_ROOT",
        ROOT.parent / "math-benchmark-data",
    )
)


class ExperimentArtifactTests(unittest.TestCase):
    def test_repeat_seed_modes(self):
        inference = {"temperature": 0.0, "seed": 100}
        self.assertEqual(
            _repeat_inference_config(
                inference,
                repeat_index=3,
                seed_mode="fixed",
            )["seed"],
            100,
        )
        self.assertEqual(
            _repeat_inference_config(
                inference,
                repeat_index=3,
                seed_mode="increment",
            )["seed"],
            102,
        )
        self.assertEqual(inference["seed"], 100)
        with self.assertRaises(ModelConfigurationError):
            _repeat_inference_config(
                {"temperature": 0.0},
                repeat_index=2,
                seed_mode="increment",
            )
        with self.assertRaises(ModelConfigurationError):
            _repeat_inference_config(
                {"seed": 2**31 - 1},
                repeat_index=2,
                seed_mode="increment",
            )
        with self.assertRaises(ModelConfigurationError):
            _repeat_inference_config(
                inference,
                repeat_index=0,
                seed_mode="increment",
            )
        with self.assertRaises(ModelConfigurationError):
            _repeat_inference_config(
                inference,
                repeat_index=1,
                seed_mode="unknown",
            )

    def test_configuration_fingerprint_is_order_independent(self):
        first = configuration_fingerprint({"model": "qwen", "config": {"a": 1}})
        second = configuration_fingerprint({"config": {"a": 1}, "model": "qwen"})
        changed = configuration_fingerprint({"model": "qwen", "config": {"a": 2}})
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_experiment_directory_is_flat_and_readable(self):
        root = Path("/results/experiments")
        path = experiment_directory(
            root,
            model="qwen 3.5/flash",
            method="self_refine",
            dataset="u-math-text-only",
            repeat=2,
            run_tag="paper baseline",
            fingerprint="0123456789abcdef" * 4,
        )
        self.assertEqual(path.parent, root)
        self.assertEqual(
            path.name,
            "qwen-3.5-flash__self_refine__u-math-text-only__r002"
            "__paper-baseline__0123456789abcdef",
        )

    def test_manifest_rejects_another_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            first = {"method": "pal", "temperature": 0}
            fingerprint = configuration_fingerprint(first)
            ensure_experiment_manifest(
                path,
                configuration=first,
                fingerprint=fingerprint,
                created_at="now",
            )
            ensure_experiment_manifest(
                path,
                configuration=first,
                fingerprint=fingerprint,
                created_at="later",
            )
            with self.assertRaises(ExperimentArtifactError):
                ensure_experiment_manifest(
                    path,
                    configuration={"method": "pal", "temperature": 1},
                    fingerprint=fingerprint,
                    created_at="later",
                )

    def test_record_and_api_summaries_include_usage_and_latency(self):
        records = [
            {
                "score": {"status": "scored", "correct": True},
                "generation": {"usage": {"total_tokens": 7}},
                "timing": {
                    "processing_seconds": 2.0,
                    "generation_seconds": 1.5,
                },
            },
            {
                "score": {"status": "error", "correct": None},
                "generation": None,
                "timing": {"processing_seconds": 1.0},
            },
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["usage"]["total_tokens"], 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api_calls.jsonl"
            path.write_text(
                "\n".join(
                    (
                        json.dumps(
                            {
                                "status": "success",
                                "latency_seconds": 1.25,
                                "usage": {"total_tokens": 7},
                            }
                        ),
                        json.dumps(
                            {
                                "status": "error",
                                "latency_seconds": 0.5,
                                "retryable": True,
                            }
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            api = summarize_api_calls(path)
        self.assertEqual((api["attempts"], api["successes"], api["errors"]), (2, 1, 1))
        self.assertEqual(api["retryable_errors"], 1)
        self.assertEqual(api["usage"]["total_tokens"], 7)

    def test_qwen_profile_contains_no_secret(self):
        profile = load_model_profile(
            ROOT / "configs" / "models" / "qwen35_flash.toml"
        )
        public = profile.public_dict()
        self.assertEqual(profile.model, "qwen3.5-flash-2026-02-23")
        self.assertEqual(profile.api_key_env, "DASHSCOPE_API_KEY_BEIJING")
        self.assertNotIn("api_key", public)

    def test_u_math_candidate_profile_has_reproducible_settings(self):
        profile = load_model_profile(
            ROOT
            / "configs"
            / "judges"
            / "candidates"
            / "qwen35_flash.toml"
        )
        self.assertEqual(profile.model, "qwen3.5-flash-2026-02-23")
        self.assertEqual(
            profile.inference_defaults(),
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": 4096,
                "seed": 20260729,
            },
        )

    def test_u_math_formal_runs_are_blocked_until_judge_is_locked(self):
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "judge is not locked yet",
        ):
            _load_locked_u_math_judge(
                ROOT / "configs" / "judges" / "u_math.lock.toml"
            )

    def test_u_math_lock_pins_the_scoring_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            judge_root = Path(directory)
            candidate_dir = judge_root / "candidates"
            candidate_dir.mkdir()
            shutil.copyfile(
                ROOT
                / "configs"
                / "judges"
                / "candidates"
                / "qwen35_flash.toml",
                candidate_dir / "qwen35_flash.toml",
            )
            lock_path = judge_root / "u_math.lock.toml"
            lock_path.write_text(
                "\n".join(
                    (
                        'status = "locked"',
                        'protocol = "u-math-manual-cot-self-verdict"',
                        "protocol_version = 1",
                        'verdict_extractor = "none"',
                        'inconclusive_treatment = "incorrect"',
                        'profile = "candidates/qwen35_flash.toml"',
                    )
                ),
                encoding="utf-8",
            )
            profile = _load_locked_u_math_judge(lock_path)
            self.assertEqual(profile.model, "qwen3.5-flash-2026-02-23")

            lock_path.write_text(
                lock_path.read_text(encoding="utf-8").replace(
                    "protocol_version = 1",
                    "protocol_version = 2",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ModelConfigurationError,
                "protocol does not match",
            ):
                _load_locked_u_math_judge(lock_path)

    def test_external_data_repository_is_clean_and_isolated(self):
        state = clean_git_repository_state(DATA_ROOT)
        self.assertEqual(len(state["revision"]), 40)
        self.assertEqual(state["branch"], "main")

    def test_invalid_inference_ranges_are_rejected_before_api_use(self):
        invalid = (
            {"temperature": -0.1},
            {"temperature": 2.1},
            {"top_p": 1.1},
            {"max_output_tokens": 0},
            {"max_output_tokens": 1.5},
            {"seed": -1},
            {"seed": True},
            {"seed": 2**31},
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ModelConfigurationError):
                    validate_inference_config(config)
        validate_inference_config({"seed": 0})
        validate_inference_config({"seed": 2**31 - 1})

    def test_backend_forwards_seed_to_openai_compatible_api(self):
        requests = []

        class SuccessfulCompletions:
            @staticmethod
            def create(**kwargs):
                requests.append(kwargs)
                message = type("Message", (), {"content": "ok"})()
                choice = type(
                    "Choice",
                    (),
                    {"message": message, "finish_reason": "stop"},
                )()
                usage = type(
                    "Usage",
                    (),
                    {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                        "completion_tokens_details": None,
                        "prompt_tokens_details": None,
                    },
                )()
                return type(
                    "Response",
                    (),
                    {
                        "id": "seed-test",
                        "choices": [choice],
                        "usage": usage,
                    },
                )()

        class SuccessfulClient:
            chat = type(
                "Chat",
                (),
                {"completions": SuccessfulCompletions()},
            )()

        profile = load_model_profile(
            ROOT / "configs" / "models" / "qwen35_flash.toml"
        )
        with patch.dict(
            "os.environ",
            {profile.api_key_env: "seed-test-key"},
        ), tempfile.TemporaryDirectory() as directory:
            telemetry_path = Path(directory) / "api_calls.jsonl"
            backend = OpenAICompatibleBackend(
                profile,
                telemetry_path=telemetry_path,
            )
            backend.client = SuccessfulClient()
            backend.generate(
                [{"role": "user", "content": "test"}],
                config={**profile.inference_defaults(), "seed": 1234},
            )
            telemetry = json.loads(
                telemetry_path.read_text(encoding="utf-8")
            )
        self.assertEqual(requests[0]["seed"], 1234)
        self.assertEqual(telemetry["seed"], 1234)

    def test_api_error_telemetry_records_message_and_redacts_key(self):
        class FailingCompletions:
            @staticmethod
            def create(**kwargs):
                raise ValueError("request rejected for secret-test-key")

        class FailingClient:
            chat = type(
                "Chat",
                (),
                {"completions": FailingCompletions()},
            )()

        profile = load_model_profile(
            ROOT / "configs" / "models" / "qwen35_flash.toml"
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {profile.api_key_env: "secret-test-key"},
        ):
            path = Path(directory) / "api_calls.jsonl"
            backend = OpenAICompatibleBackend(profile, telemetry_path=path)
            backend.client = FailingClient()
            with self.assertRaises(ValueError):
                backend.generate(
                    [{"role": "user", "content": "test"}],
                    config={**profile.inference_defaults(), "seed": 5678},
                )
            telemetry = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("request rejected", telemetry["error_message"])
        self.assertIn("[REDACTED]", telemetry["error_message"])
        self.assertNotIn("secret-test-key", telemetry["error_message"])
        self.assertEqual(telemetry["seed"], 5678)


if __name__ == "__main__":
    unittest.main()
