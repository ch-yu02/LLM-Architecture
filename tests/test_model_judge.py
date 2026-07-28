from __future__ import annotations

import unittest

from benchmark_core.schema import Generation, Problem
from scorers.judge import JudgeVerdict, UMathJudgeScorer
from scorers.model_judge import ModelJudgeBackend, judge_protocol_metadata


class StaticJudgeBackend:
    def __init__(self, output: str):
        self.output = output
        self.messages = None
        self.config = None

    def generate(self, messages, *, config):
        self.messages = messages
        self.config = config
        return Generation(
            self.output,
            latency_seconds=0.25,
            usage={"total_tokens": 12},
        )


class ModelJudgeTests(unittest.TestCase):
    def _judge(self, output: str):
        backend = StaticJudgeBackend(output)
        judge = ModelJudgeBackend(
            backend,
            model_name="candidate",
            inference_config={"temperature": 0.0},
        )
        decision = judge.judge(
            Problem("u-math-text-only", "one", "Compute 1+1.", "2"),
            "The answer is 2.",
        )
        return backend, decision

    def test_uses_official_manual_cot_steps_with_tristate_final_verdict(self):
        backend, decision = self._judge("The answers agree.\nYes")
        prompt = backend.messages[0]["content"]
        self.assertIn("PROBLEM STATEMENT:", prompt)
        self.assertIn("CORRECT ANSWER:", prompt)
        self.assertIn("SOLUTION TO EVALUATE:", prompt)
        for step in range(1, 5):
            self.assertIn(f"{step}.", prompt)
        self.assertIn('"Yes", "No", or "Inconclusive"', prompt)
        self.assertEqual(decision.verdict, JudgeVerdict.YES)
        self.assertTrue(decision.correct)
        self.assertEqual(
            decision.metadata["verdict_parse_status"],
            "standalone-final-line",
        )

    def test_last_standalone_verdict_wins(self):
        _, decision = self._judge(
            "I first considered Yes.\nNo\nAfter checking again:\n"
            "Final verdict: Inconclusive"
        )
        self.assertEqual(decision.verdict, JudgeVerdict.INCONCLUSIVE)
        self.assertFalse(decision.correct)

    def test_no_and_inconclusive_are_incorrect(self):
        for output, expected in (
            ("Reasoning.\nNo", JudgeVerdict.NO),
            ("Reasoning.\nInconclusive", JudgeVerdict.INCONCLUSIVE),
        ):
            with self.subTest(output=output):
                _, decision = self._judge(output)
                self.assertEqual(decision.verdict, expected)
                self.assertFalse(decision.correct)

    def test_malformed_output_falls_back_to_inconclusive(self):
        _, decision = self._judge("I cannot provide a final answer.")
        self.assertEqual(decision.verdict, JudgeVerdict.INCONCLUSIVE)
        self.assertEqual(
            decision.metadata["verdict_parse_status"],
            "malformed-output-fallback",
        )
        self.assertFalse(decision.correct)

    def test_score_preserves_tristate_verdict_and_parse_status(self):
        backend = StaticJudgeBackend("Unable to decide.\nInconclusive")
        judge = ModelJudgeBackend(
            backend,
            model_name="candidate",
            inference_config={"temperature": 0.0},
        )
        score = UMathJudgeScorer(judge).score(
            Problem("u-math-text-only", "one", "Question", "Reference"),
            Generation("Candidate"),
        )
        self.assertFalse(score.correct)
        self.assertEqual(score.details["judge_verdict"], "Inconclusive")
        self.assertEqual(
            score.details["judge_metadata"]["verdict_parse_status"],
            "standalone-final-line",
        )

    def test_protocol_metadata_records_the_local_adaptation(self):
        self.assertEqual(
            judge_protocol_metadata(),
            {
                "name": "u-math-manual-cot-self-verdict",
                "version": 1,
                "prompt": "u-math-official-manual-cot-adapted-tristate",
                "verdicts": ["Yes", "No", "Inconclusive"],
                "inconclusive_treatment": "incorrect",
                "verdict_extractor": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
