from __future__ import annotations

import traceback

from benchmark_core.schema import Generation, Problem, Score

from .bridge import BridgeError, JsonSubprocessBridge


class OfficialBridgeScorer:
    def __init__(self, bridge: JsonSubprocessBridge, dataset_type: str | None = None):
        self.bridge = bridge
        self.dataset_type = dataset_type

    def score(self, problem: Problem, generation: Generation) -> Score:
        request = {
            "operation": "score",
            "problem": problem.prompt,
            "generated_text": generation.text,
            "finish_reason": generation.finish_reason,
            "reference_answer": problem.reference_answer,
            "metadata": problem.metadata,
        }
        if self.dataset_type:
            request["dataset_type"] = self.dataset_type
        try:
            result = self.bridge.call(request)
        except BridgeError as exc:
            return Score.failed(
                str(exc),
                details={
                    "checker": "official",
                    "failure_stage": "scoring",
                    "exception_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
        return Score.scored(
            bool(result["correct"]),
            extracted_answer=result.get("extracted_answer"),
            details=result.get("details", {}),
        )
