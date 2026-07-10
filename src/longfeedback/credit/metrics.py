"""Credit-recovery metrics against oracle interventional targets."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from longfeedback.evaluation.metrics import (
    pearson_correlation,
    sign_accuracy,
    spearman_correlation,
)


def _paired(first: ArrayLike, second: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"metric inputs must have equal shapes: {left.shape} != {right.shape}")
    if left.size == 0:
        raise ValueError("metric inputs must be non-empty")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("metric inputs must contain only finite values")
    return left, right


def kendall_tau(expected: ArrayLike, predicted: ArrayLike) -> float:
    """Return tau-a via pairwise concordance, 0.0 for constant inputs."""

    expected_array, predicted_array = _paired(expected, predicted)
    size = expected_array.size
    if size < 2:
        return 0.0
    concordant_minus_discordant = 0.0
    chunk = 512
    for start in range(0, size, chunk):
        stop = min(start + chunk, size)
        left = np.sign(expected_array[start:stop, None] - expected_array[None, :])
        right = np.sign(predicted_array[start:stop, None] - predicted_array[None, :])
        block = left * right
        # Keep each unordered pair once: strictly upper-triangular columns.
        columns = np.arange(size)[None, :]
        rows = np.arange(start, stop)[:, None]
        concordant_minus_discordant += float(np.sum(block * (columns > rows)))
    total_pairs = size * (size - 1) / 2
    return float(np.clip(concordant_minus_discordant / total_pairs, -1.0, 1.0))


def normalized_rmse(expected: ArrayLike, predicted: ArrayLike) -> float:
    """Return RMSE scaled by the oracle-target standard deviation."""

    expected_array, predicted_array = _paired(expected, predicted)
    error = float(np.sqrt(np.mean(np.square(predicted_array - expected_array))))
    scale = float(np.std(expected_array))
    if scale == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return error / scale


def credit_recovery_summary(expected: ArrayLike, predicted: ArrayLike) -> dict[str, float]:
    """Return the design-doc credit metric set for one model/target pair."""

    expected_array, predicted_array = _paired(expected, predicted)
    return {
        "pearson": pearson_correlation(expected_array, predicted_array),
        "spearman": spearman_correlation(expected_array, predicted_array),
        "kendall_tau": kendall_tau(expected_array, predicted_array),
        "sign_accuracy": sign_accuracy(expected_array, predicted_array),
        "normalized_rmse": normalized_rmse(expected_array, predicted_array),
        "examples": float(expected_array.size),
    }


def spearman_by_temporal_distance(
    expected: ArrayLike,
    predicted: ArrayLike,
    steps_to_outcome: ArrayLike,
    *,
    min_group_size: int = 8,
) -> dict[str, float]:
    """Return Spearman correlation grouped by distance to the outcome."""

    expected_array, predicted_array = _paired(expected, predicted)
    distances = np.asarray(steps_to_outcome, dtype=np.int64).reshape(-1)
    if distances.shape != expected_array.shape:
        raise ValueError("steps_to_outcome must align with the metric inputs")
    if np.any(distances < 0):
        raise ValueError("steps_to_outcome must be non-negative")
    result: dict[str, float] = {}
    for distance in np.unique(distances):
        selected = distances == distance
        if int(np.sum(selected)) < min_group_size:
            continue
        result[str(int(distance))] = spearman_correlation(
            expected_array[selected], predicted_array[selected]
        )
    return result
