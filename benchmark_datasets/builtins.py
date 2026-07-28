from __future__ import annotations

from pathlib import Path
from typing import Iterable

from benchmark_core.schema import Problem
from scorers.bridge import JsonSubprocessBridge
from scorers.judge import UMathJudgeScorer
from scorers.numeric import NumericAnswerScorer
from scorers.official import OfficialBridgeScorer

from .base import DatasetContext
from .jsonl import iter_unified_jsonl
from .registry import register_dataset


_BRIDGES = Path(__file__).resolve().parents[1] / "bridges"


class JsonlPlugin:
    aliases: tuple[str, ...] = ()
    filename: str

    def iter_problems(self, context: DatasetContext) -> Iterable[Problem]:
        return iter_unified_jsonl(
            context.data_root / "data" / "processed" / self.filename, self.name
        )

    def problem_count(self, context: DatasetContext) -> int:
        path = context.data_root / "data" / "processed" / self.filename
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with path.open(encoding="utf-8") as stream:
            return sum(bool(line.strip()) for line in stream)


class GSM1KPlugin(JsonlPlugin):
    name = "gsm1k"
    aliases = ("GSM1K",)
    filename = "gsm1k.jsonl"

    def create_scorer(self, context: DatasetContext) -> NumericAnswerScorer:
        return NumericAnswerScorer()


class MathPerturbPlugin(JsonlPlugin):
    name = "math-perturb"
    aliases = ("math_perturb", "MATH-Perturb")
    filename = "math_perturb_test.jsonl"

    def create_scorer(self, context: DatasetContext) -> OfficialBridgeScorer:
        bridge = JsonSubprocessBridge(
            _BRIDGES / "math_perturb.py",
            context.data_root / "data" / "math_perturb" / "source",
            python=context.python_for(self.name),
        )
        return OfficialBridgeScorer(bridge, dataset_type="perturb")


class HARPPlugin(JsonlPlugin):
    name = "harp"
    aliases = ("HARP",)
    filename = "harp_test.jsonl"

    def create_scorer(self, context: DatasetContext) -> OfficialBridgeScorer:
        bridge = JsonSubprocessBridge(
            _BRIDGES / "harp.py",
            context.data_root / "data" / "harp",
            python=context.python_for(self.name),
        )
        return OfficialBridgeScorer(bridge)


class UMathPlugin(JsonlPlugin):
    name = "u-math-text-only"
    aliases = ("u_math_text_only", "U-MATH text-only")
    filename = "u_math_text_only_test.jsonl"

    def create_scorer(self, context: DatasetContext) -> UMathJudgeScorer:
        return UMathJudgeScorer(context.judge)


class MathConstructPlugin(JsonlPlugin):
    name = "mathconstruct"
    aliases = ("MathConstruct",)
    filename = "mathconstruct_test.jsonl"

    def _bridge(self, context: DatasetContext) -> JsonSubprocessBridge:
        return JsonSubprocessBridge(
            _BRIDGES / "mathconstruct.py",
            context.data_root / "data" / "mathconstruct" / "source",
            python=context.python_for(self.name),
            timeout_seconds=60,
        )

    def create_scorer(self, context: DatasetContext) -> OfficialBridgeScorer:
        return OfficialBridgeScorer(self._bridge(context))


for _plugin in (
    GSM1KPlugin(),
    MathPerturbPlugin(),
    HARPPlugin(),
    UMathPlugin(),
    MathConstructPlugin(),
):
    register_dataset(_plugin)
