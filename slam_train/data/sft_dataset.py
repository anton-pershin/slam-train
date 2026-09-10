"""Conversion of an EvalCaseCollection into a trl prompt-completion dataset."""

from __future__ import annotations

import json
import math
import random
from typing import Any, Optional

from slam_core.collections.base import EvalCase, EvalCaseCollection
from slam_core.collections.text_generation import TextGenerationInput


def materialize_collection(collection: EvalCaseCollection) -> list[EvalCase]:
    """Drain a one-shot EvalCaseCollection iterator into a list."""
    collection.load()
    return list(collection)


def serialize_completion(y_true: Any) -> str:
    """Serialize ground truth to the completion string.

    Strings are used verbatim; non-strings (e.g. MergeQuality dicts) are dumped
    as JSON exactly as slam_core.scorers.merge_quality_scorer.safe_parse_prediction
    parses them back at eval time.
    """
    if isinstance(y_true, str):
        return y_true
    return json.dumps(y_true, ensure_ascii=False, indent=2)


def build_records(cases: list[EvalCase]) -> list[dict[str, Any]]:
    """Build trl prompt-completion records from eval cases.

    Both prompt and completion are lists of chat messages (trl conversational
    format): a system message in the prompt only when the collection emits one,
    the user message, and the completion as a single assistant message.
    """
    records: list[dict[str, Any]] = []
    for case in cases:
        x: TextGenerationInput = case["x"]
        prompt: list[dict[str, str]] = []
        if x["system_prompt"] is not None:
            prompt.append({"role": "system", "content": x["system_prompt"]})
        prompt.append({"role": "user", "content": x["user_prompt"]})
        records.append(
            {
                "prompt": prompt,
                "completion": [
                    {
                        "role": "assistant",
                        "content": serialize_completion(case["y_true"]),
                    }
                ],
            }
        )
    return records


def split_indices(
    n_items: int,
    train_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Shuffle and split index positions into disjoint train and eval subsets.

    Splitting raw source lines and their derived eval cases through this
    single function guarantees the two splits stay consistent: the same
    (ratio, seed) pair always yields the same permutation.
    """
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"train_ratio must lie in (0, 1], got {train_ratio}")
    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)
    split_index = math.floor(n_items * train_ratio)
    if split_index == 0:
        raise ValueError(
            f"train_ratio {train_ratio} yields an empty training subset "
            f"over {n_items} items"
        )
    return indices[:split_index], indices[split_index:]


def split_cases(
    cases: list[EvalCase],
    train_ratio: float,
    seed: int,
) -> tuple[list[EvalCase], list[EvalCase]]:
    """Shuffle and split cases into disjoint train and eval subsets.

    The ratio must lie in (0, 1]. A training subset of size zero is an error.
    """
    train_idx, eval_idx = split_indices(len(cases), train_ratio, seed)
    return [cases[i] for i in train_idx], [cases[i] for i in eval_idx]
