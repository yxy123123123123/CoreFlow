from __future__ import annotations

import math
import random
from collections import Counter


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_difference(
    reference: list[bool],
    candidate: list[bool],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("Paired outcomes must have the same non-zero length")
    differences = [int(candidate[index]) - int(reference[index]) for index in range(len(reference))]
    rng = random.Random(seed)
    n = len(differences)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": sum(differences) / n,
        "lower": quantile(estimates, alpha),
        "upper": quantile(estimates, 1.0 - alpha),
        "confidence": confidence,
        "samples": samples,
        "seed": seed,
    }


def paired_transitions(reference: dict[str, bool], candidate: dict[str, bool]) -> dict:
    if set(reference) != set(candidate):
        raise ValueError("Paired task IDs differ")
    gained = sorted(task_id for task_id in reference if not reference[task_id] and candidate[task_id])
    lost = sorted(task_id for task_id in reference if reference[task_id] and not candidate[task_id])
    return {
        "gained": gained,
        "lost": lost,
        "gained_count": len(gained),
        "lost_count": len(lost),
        "net_correct_change": len(gained) - len(lost),
    }


def summarize_numbers(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": sum(numeric) / len(numeric),
        "median": quantile(numeric, 0.5),
        "p95": quantile(numeric, 0.95),
        "min": min(numeric),
        "max": max(numeric),
    }


def dominant_fraction(values: list[str]) -> float:
    if not values:
        return 0.0
    return Counter(values).most_common(1)[0][1] / len(values)
