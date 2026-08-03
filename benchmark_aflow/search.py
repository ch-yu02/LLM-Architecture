from __future__ import annotations

import hashlib
import json
from typing import Any

from adapters.aflow import AFlowAdapter, workflow_call_bounds


SEARCH_PROTOCOL = "aflow-validation-graph-search-v1"


def workflow_digest(workflow: dict[str, Any]) -> str:
    encoded = json.dumps(
        workflow,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_workflow_proposal(text: str) -> tuple[dict[str, Any], str]:
    content = text.strip()
    if content.startswith("```"):
        first_newline = content.find("\n")
        if first_newline >= 0:
            content = content[first_newline + 1 :]
        if content.endswith("```"):
            content = content[:-3].rstrip()
    opening = content.find("{")
    closing = content.rfind("}")
    if opening < 0 or closing < opening:
        raise ValueError("AFlow optimizer did not return a JSON object")
    try:
        payload = json.loads(content[opening : closing + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"AFlow optimizer returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("AFlow optimizer proposal must be a JSON object")
    workflow = payload.get("workflow", payload)
    if not isinstance(workflow, dict):
        raise ValueError("AFlow optimizer proposal has no workflow object")
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        raise ValueError("AFlow optimizer rationale must be a string")
    AFlowAdapter.validate_workflow(workflow)
    return workflow, rationale.strip()


def candidate_rank(candidate: dict[str, Any]) -> tuple[float, int, int]:
    accuracy = candidate.get("accuracy")
    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
        return (-1.0, -10**9, -10**9)
    workflow = candidate["workflow"]
    _, maximum_calls = workflow_call_bounds(workflow)
    return float(accuracy), -maximum_calls, -len(workflow["nodes"])


def build_optimizer_prompt(
    *,
    dataset: str,
    answer_instruction: str,
    best_candidate: dict[str, Any],
    history: list[dict[str, Any]],
    failure_examples: list[dict[str, Any]],
    round_number: int,
) -> str:
    compact_history = [
        {
            "round": item["round"],
            "accuracy": item.get("accuracy"),
            "maximum_model_calls": workflow_call_bounds(item["workflow"])[1],
            "workflow": item["workflow"],
        }
        for item in history
        if item.get("accuracy") is not None
    ]
    return f"""You are the graph optimizer in AFlow. Propose one improved frozen
workflow for mathematical problem solving. The workflow will be evaluated only
on a fixed validation split; you never receive or use test examples.

Dataset: {dataset}
Required final-answer format: {answer_instruction or "none"}
Search round: {round_number}

Allowed node operators and schemas:
- custom: {{"id": str, "operator": "custom", "instruction": str, "inputs": []}}
- answer_generate: {{"id": str, "operator": "answer_generate", "inputs": []}}
- programmer: {{"id": str, "operator": "programmer", "inputs": [] or [one earlier id], "max_attempts": 1..3}}
- ensemble: {{"id": str, "operator": "ensemble", "inputs": [two or more earlier ids]}}

Nodes execute in list order. The workflow output must name one node. Prefer the
smallest graph that can improve validation accuracy; extra model calls are used
as the tie-break penalty. A custom instruction is prepended to the problem, so
it must be self-contained and must preserve the required final-answer format.

Current best:
{json.dumps(best_candidate, ensure_ascii=False, indent=2)}

Evaluated history:
{json.dumps(compact_history, ensure_ascii=False, indent=2)}

Representative validation failures from the current best:
{json.dumps(failure_examples, ensure_ascii=False, indent=2)}

Return only JSON in this exact outer shape:
{{
  "rationale": "short explanation of the modification",
  "workflow": {{
    "id": "descriptive-id",
    "nodes": [ ... ],
    "output": "node-id"
  }}
}}
""".strip()
