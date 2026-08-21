from __future__ import annotations

from pathlib import Path
from typing import Iterable

from benchmark_core.schema import Generation, Problem
from benchmark_datasets import get_dataset
from benchmark_datasets.base import DatasetContext


class CheckerCapabilityError(RuntimeError):
    pass


def _unbox_reference(value: object) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text.startswith("$") and text.endswith("$"):
        return text[1:-1].strip()
    return text


def validate_checker_capabilities(
    checker_python: Path | str,
    data_root: Path,
    datasets: Iterable[str],
) -> None:
    """Exercise selected official checkers, including symbolic equivalence."""

    selected = set(datasets)
    context = DatasetContext(
        data_root,
        bridge_python={
            "math-perturb": str(checker_python),
            "harp": str(checker_python),
            "harp-small": str(checker_python),
            "mathconstruct": str(checker_python),
        },
    )

    if "math-perturb" in selected:
        scorer = get_dataset("math-perturb").create_scorer(context)
        cases = (
            Problem(
                "math-perturb",
                "checker-preflight-fraction",
                "Compute one quarter.",
                "0.25",
            ),
            Problem(
                "math-perturb",
                "checker-preflight-symbolic",
                "Simplify the expression.",
                r"12+12\sqrt{2}",
            ),
        )
        generations = (
            Generation(r"\boxed{\frac{1}{4}}"),
            Generation(r"\boxed{12 + 12\sqrt{2}}"),
        )
        for problem, generation in zip(cases, generations, strict=True):
            score = scorer.score(problem, generation)
            if score.status.value != "scored" or not score.correct:
                raise CheckerCapabilityError(
                    "MATH-Perturb symbolic equivalence is unavailable; "
                    "verify the checker environment includes lark"
                )

    harp_names = selected.intersection({"harp", "harp-small"})
    if harp_names:
        dataset_name = "harp-small" if "harp-small" in harp_names else "harp"
        plugin = get_dataset(dataset_name)
        problem = next(iter(plugin.iter_problems(context)))
        scorer = plugin.create_scorer(context)
        answer = _unbox_reference(problem.reference_answer)
        score = scorer.score(
            problem,
            Generation(rf"Answer: \boxed{{{answer}}}"),
        )
        if score.status.value != "scored" or not score.correct:
            raise CheckerCapabilityError(
                f"{dataset_name} official checker failed its reference-answer preflight"
            )
