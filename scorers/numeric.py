from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from benchmark_core.schema import Generation, Problem, Score


_BOXED_COMMAND = re.compile(r"\\boxed\s*\{")
_ANSWER_LINE = re.compile(r"^\s*Answer\s*:\s*(.*?)\s*$", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _last_complete_box(text: str) -> str | None:
    answer: str | None = None
    for match in _BOXED_COMMAND.finditer(text):
        opening_brace = match.end() - 1
        depth = 0
        escaped = False
        for index in range(opening_brace, len(text)):
            character = text[index]
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    answer = text[opening_brace + 1 : index].strip()
                    break
    return answer


def _number_from(text: str) -> str | None:
    matches = _NUMBER.findall(text)
    return matches[-1].replace(",", "") if matches else None


def extract_numeric_submission(text: str) -> tuple[str | None, str]:
    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        answer_match = _ANSWER_LINE.fullmatch(nonempty_lines[-1])
        if answer_match:
            return _number_from(answer_match.group(1)), "final_answer_line"

    boxed = _last_complete_box(text)
    if boxed is not None:
        return _number_from(boxed), "last_complete_boxed"

    return _number_from(text), "last_number_fallback"


def extract_last_number(text: str) -> str | None:
    """Compatibility helper for reference answers and external callers."""

    return extract_numeric_submission(text)[0]


class NumericAnswerScorer:
    def score(self, problem: Problem, generation: Generation) -> Score:
        predicted, extraction_source = extract_numeric_submission(generation.text)
        expected = extract_last_number(str(problem.reference_answer))
        if predicted is None or expected is None:
            return Score.scored(
                False,
                extracted_answer=predicted,
                details={
                    "expected": expected,
                    "reason": "number_not_found",
                    "answer_extraction": extraction_source,
                    "finish_reason": generation.finish_reason,
                },
            )
        try:
            correct = Decimal(predicted) == Decimal(expected)
        except InvalidOperation:
            correct = predicted == expected
        return Score.scored(
            correct,
            extracted_answer=predicted,
            details={
                "expected": expected,
                "answer_extraction": extraction_source,
                "finish_reason": generation.finish_reason,
            },
        )
