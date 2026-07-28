from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from benchmark_core.schema import Problem


def iter_unified_jsonl(path: Path, dataset: str) -> Iterable[Problem]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                yield Problem(
                    dataset=dataset,
                    id=str(item["id"]),
                    prompt=str(item["problem"]),
                    reference_answer=item.get("answer"),
                    metadata=item.get("metadata") or {},
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid sample at {path}:{line_number}: {exc}") from exc
