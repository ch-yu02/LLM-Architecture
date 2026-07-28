from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark_core.interfaces import ModelBackend
from benchmark_core.schema import Generation, Problem
from executors import ProgramExecutionError, RestrictedPythonExecutor

from .base import MethodAdapter
from .pal import extract_python_code


_ANSWER_PROMPT = """Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Your task: {problem}
"""

_ENSEMBLE_PROMPT = """Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution.
"""

_PROGRAMMER_PROMPT = """You are a professional Python programmer. Write complete,
self-contained Python code that solves the mathematical problem below. Define a
zero-argument function named `solution` which returns the final result. The safe
runtime provides `math`, `Fraction`, and `Decimal`; do not import modules.

Problem description: {problem}
Other analysis: {analysis}
"""

_DEFAULT_WORKFLOW = {
    "id": "official-round-1-custom",
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

_DEFAULT_ARTIFACT = {
    "source": "AFlow workspace/MATH and workspace/GSM8K round_1",
    "optimization_split": None,
    "optimization_cost": 0,
    "evaluation_data_used": False,
    "frozen": True,
}


def _xml_field(text: str, field: str) -> str | None:
    match = re.search(
        rf"<{re.escape(field)}>\s*(.*?)\s*</{re.escape(field)}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else None


def _solution_letter(text: str, count: int) -> str:
    candidate = _xml_field(text, "solution_letter")
    if candidate is None:
        match = re.search(r"\b([A-Z])\b", text.upper())
        candidate = match.group(1) if match else ""
    letter = candidate.strip().upper()[:1]
    if not letter or not ("A" <= letter <= chr(64 + count)):
        raise ValueError(f"AFlow ensemble returned invalid solution letter: {candidate!r}")
    return letter


class AFlowAdapter(MethodAdapter):
    name = "aflow"
    required_paths = (
        "run.py",
        "run_baseline.py",
        "config/config2.example.yaml",
        "scripts/operators.py",
        "workspace/MATH/workflows/round_1/graph.py",
        "workspace/GSM8K/workflows/round_1/graph.py",
    )

    def __init__(
        self,
        source_path: Path | str,
        *,
        executor: RestrictedPythonExecutor | None = None,
    ) -> None:
        super().__init__(source_path)
        self.executor = executor or RestrictedPythonExecutor()
        self.validate_source()

    @staticmethod
    def _validate_workflow(workflow: dict[str, Any]) -> None:
        if not isinstance(workflow.get("id"), str) or not workflow["id"]:
            raise ValueError("AFlow workflow requires a non-empty id")
        nodes = workflow.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("AFlow workflow requires a non-empty nodes list")
        seen: set[str] = set()
        allowed = {"custom", "answer_generate", "ensemble", "programmer"}
        for node in nodes:
            node_id = node.get("id")
            if not isinstance(node_id, str) or not node_id or node_id in seen:
                raise ValueError(f"Invalid or duplicate AFlow node id: {node_id!r}")
            operator = node.get("operator")
            if operator not in allowed:
                raise ValueError(f"Unsupported AFlow operator: {operator!r}")
            inputs = node.get("inputs", [])
            if not isinstance(inputs, list) or any(item not in seen for item in inputs):
                raise ValueError(
                    f"AFlow node {node_id!r} must reference earlier node ids"
                )
            if operator == "ensemble" and len(inputs) < 2:
                raise ValueError("AFlow ensemble requires at least two inputs")
            if operator in {"custom", "answer_generate"} and inputs:
                raise ValueError(f"AFlow {operator} does not accept inputs")
            if operator == "programmer" and len(inputs) > 1:
                raise ValueError("AFlow programmer accepts at most one analysis input")
            if operator == "custom" and not isinstance(
                node.get("instruction", ""), str
            ):
                raise ValueError("AFlow custom instruction must be a string")
            if operator == "programmer":
                max_attempts = node.get("max_attempts", 3)
                if (
                    not isinstance(max_attempts, int)
                    or isinstance(max_attempts, bool)
                    or not 1 <= max_attempts <= 3
                ):
                    raise ValueError(
                        "AFlow programmer max_attempts must be an integer from 1 to 3"
                    )
            seen.add(node_id)
        if workflow.get("output") not in seen:
            raise ValueError("AFlow workflow output must reference a node id")

    def run(
        self,
        problem: Problem,
        backend: ModelBackend,
        *,
        config: dict[str, Any],
    ) -> Generation:
        unknown = set(config) - {"workflow", "workflow_artifact"}
        if unknown:
            raise ValueError(f"Unknown AFlow config: {sorted(unknown)}")
        workflow = config.get("workflow", _DEFAULT_WORKFLOW)
        artifact = config.get("workflow_artifact", _DEFAULT_ARTIFACT)
        if not isinstance(artifact, dict) or not artifact.get("frozen"):
            raise ValueError(
                "AFlow evaluation requires a frozen workflow_artifact; "
                "online optimization on evaluation data is forbidden"
            )
        required_artifact_fields = {
            "source",
            "optimization_split",
            "optimization_cost",
            "evaluation_data_used",
        }
        missing_artifact_fields = required_artifact_fields - set(artifact)
        if missing_artifact_fields:
            raise ValueError(
                "AFlow workflow_artifact is missing audit fields: "
                f"{sorted(missing_artifact_fields)}"
            )
        if artifact["evaluation_data_used"] is not False:
            raise ValueError(
                "AFlow workflow optimization must not use evaluation data"
            )
        if not isinstance(artifact["source"], str) or not artifact["source"]:
            raise ValueError("AFlow workflow_artifact source must be non-empty")
        if artifact["optimization_split"] is not None and not isinstance(
            artifact["optimization_split"], str
        ):
            raise ValueError("AFlow optimization_split must be a string or null")
        optimization_cost = artifact["optimization_cost"]
        if (
            not isinstance(optimization_cost, (int, float))
            or isinstance(optimization_cost, bool)
            or optimization_cost < 0
        ):
            raise ValueError("AFlow optimization_cost must be non-negative")
        self._validate_workflow(workflow)

        values: dict[str, str] = {}
        trace: list[dict[str, Any]] = []
        last_generation: Generation | None = None
        for node in workflow["nodes"]:
            node_id = node["id"]
            operator = node["operator"]
            inputs = [values[item] for item in node.get("inputs", [])]
            if operator == "custom":
                prompt = f"{node.get('instruction', '')}{problem.prompt}"
                last_generation = backend.generate(
                    [{"role": "user", "content": prompt}], config={}
                )
                value = last_generation.text
            elif operator == "answer_generate":
                last_generation = backend.generate(
                    [
                        {
                            "role": "user",
                            "content": _ANSWER_PROMPT.format(problem=problem.prompt),
                        }
                    ],
                    config={},
                )
                value = (
                    _xml_field(last_generation.text, "answer")
                    or last_generation.text
                )
            elif operator == "ensemble":
                choices = "\n\n".join(
                    f"{chr(65 + index)}:\n{solution}"
                    for index, solution in enumerate(inputs)
                )
                last_generation = backend.generate(
                    [
                        {
                            "role": "user",
                            "content": _ENSEMBLE_PROMPT.format(
                                problem=problem.prompt, solutions=choices
                            ),
                        }
                    ],
                    config={},
                )
                letter = _solution_letter(last_generation.text, len(inputs))
                value = inputs[ord(letter) - ord("A")]
            else:
                analysis = inputs[0] if inputs else "None"
                max_attempts = node.get("max_attempts", 3)
                feedback = ""
                attempts: list[dict[str, Any]] = []
                executed = None
                for attempt in range(1, max_attempts + 1):
                    last_generation = backend.generate(
                        [
                            {
                                "role": "user",
                                "content": (
                                    _PROGRAMMER_PROMPT.format(
                                        problem=problem.prompt, analysis=analysis
                                    )
                                    + feedback
                                ),
                            }
                        ],
                        config={},
                    )
                    code = extract_python_code(last_generation.text)
                    if "def solve(" in code and "def solution(" not in code:
                        code = code.replace("def solve(", "def solution(", 1)
                    try:
                        executed = self.executor.execute(code)
                        attempts.append(
                            {"attempt": attempt, "code": code, "status": "success"}
                        )
                        break
                    except ProgramExecutionError as exc:
                        attempts.append(
                            {
                                "attempt": attempt,
                                "code": code,
                                "status": "error",
                                "error": str(exc),
                            }
                        )
                        feedback = (
                            "\nThe previous program failed in the safe runtime. "
                            f"Error: {exc}\nRewrite the complete program."
                        )
                if executed is None:
                    raise ProgramExecutionError(
                        f"AFlow Programmer failed after {max_attempts} attempts"
                    )
                value = rf"\boxed{{{executed.value}}}"
            values[node_id] = value
            trace.append(
                {
                    "id": node_id,
                    "operator": operator,
                    "inputs": node.get("inputs", []),
                    "output": value,
                    **({"attempts": attempts} if operator == "programmer" else {}),
                }
            )

        assert last_generation is not None
        return replace(
            last_generation,
            text=values[workflow["output"]],
            metadata={
                **last_generation.metadata,
                "method": "aflow",
                "workflow": workflow,
                "workflow_artifact": artifact,
                "workflow_trace": trace,
            },
        )
