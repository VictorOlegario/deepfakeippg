"""Parallel processing utilities for video pipeline."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Optional, Sequence

from src.utils.logger import get_logger

logger = get_logger(__name__)


def parallel_process(
    func: Callable[..., Any],
    items: Sequence[Any],
    max_workers: Optional[int] = None,
    desc: str = "Processing",
) -> list[Any]:
    """Process items in parallel using a process pool.

    Args:
        func: Function to apply to each item.
        items: Sequence of items to process.
        max_workers: Maximum number of worker processes. Defaults to CPU count.
        desc: Description for progress logging.

    Returns:
        List of results in the same order as input items.
    """
    total = len(items)
    results: list[Any] = [None] * total

    if total == 0:
        return results

    if max_workers == 1:
        for i, item in enumerate(items):
            try:
                results[i] = func(item)
            except Exception as e:
                logger.error("%s failed for item %d: %s", desc, i, e)
                results[i] = None
            if (i + 1) % max(1, total // 10) == 0:
                logger.info("%s: %d/%d completed", desc, i + 1, total)
        return results

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(func, item): i for i, item in enumerate(items)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error("%s failed for item %d: %s", desc, idx, e)
                results[idx] = None

            completed += 1
            if completed % max(1, total // 10) == 0:
                logger.info("%s: %d/%d completed", desc, completed, total)

    logger.info("%s: all %d items completed", desc, total)
    return results
