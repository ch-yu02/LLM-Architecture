from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from benchmark_core.schema import Generation, Problem, Score


_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def extract_last_number(text: str) -> str | None:
    boxed = _BOXED.findall(text)
    if boxed:
        text = boxed[-1]
    matches = _NUMBER.findall(text)
    return matches[-1].replace(",", "") if matches else None


class NumericAnswerScorer:
    def score(self, problem: Problem, generation: Generation) -> Score:
        predicted = extract_last_number(generation.text)
        expected = extract_last_number(str(problem.reference_answer))
        if predicted is None or expected is None:
            return Score.scored(
                False,
                extracted_answer=predicted,
                details={"expected": expected, "reason": "number_not_found"},
            )
        try:
            correct = Decimal(predicted) == Decimal(expected)
        except InvalidOperation:
            correct = predicted == expected
        return Score.scored(
            correct, extracted_answer=predicted, details={"expected": expected}
        )
