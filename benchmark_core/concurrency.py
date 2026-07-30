from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, TypeVar


Input = TypeVar("Input")
Output = TypeVar("Output")


def ordered_parallel_map(
    function: Callable[[Input], Output],
    items: Iterable[Input],
    *,
    concurrency: int,
    thread_name_prefix: str,
) -> Iterator[Output]:
    """Run a bounded number of calls concurrently and yield in input order."""

    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency < 1
    ):
        raise ValueError("concurrency must be a positive integer")
    if concurrency == 1:
        for item in items:
            yield function(item)
        return

    iterator = iter(items)
    executor = ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix=thread_name_prefix,
    )
    pending: deque[Future[Output]] = deque()
    try:
        for _ in range(concurrency):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(executor.submit(function, item))

        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(function, item))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
