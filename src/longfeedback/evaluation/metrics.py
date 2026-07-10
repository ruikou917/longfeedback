"""Dependency-light deterministic E0 metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


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


def rmse(expected: ArrayLike, predicted: ArrayLike) -> float:
    expected_array, predicted_array = _paired(expected, predicted)
    return float(np.sqrt(np.mean(np.square(predicted_array - expected_array))))


def pearson_correlation(expected: ArrayLike, predicted: ArrayLike) -> float:
    """Return Pearson correlation, using zero for a constant-input diagnostic.

    The zero convention keeps baseline plots finite; it represents an undefined
    correlation and must be reported with that qualification.
    """

    expected_array, predicted_array = _paired(expected, predicted)
    if expected_array.size < 2:
        return 0.0
    centered_expected = expected_array - expected_array.mean()
    centered_predicted = predicted_array - predicted_array.mean()
    denominator = np.linalg.norm(centered_expected) * np.linalg.norm(centered_predicted)
    if denominator == 0:
        return 0.0
    correlation = np.dot(centered_expected, centered_predicted) / denominator
    return float(np.clip(correlation, -1.0, 1.0))


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop + 1)
        start = stop
    return ranks


def spearman_correlation(expected: ArrayLike, predicted: ArrayLike) -> float:
    """Return tie-aware Spearman correlation with the same zero convention."""

    expected_array, predicted_array = _paired(expected, predicted)
    return pearson_correlation(
        _average_ranks(expected_array),
        _average_ranks(predicted_array),
    )


def sign_accuracy(expected: ArrayLike, predicted: ArrayLike) -> float:
    expected_array, predicted_array = _paired(expected, predicted)
    return float(np.mean(np.sign(expected_array) == np.sign(predicted_array)))


def _binary_labels(labels: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(labels, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("labels must be non-empty")
    if not np.all(np.isin(array, (0.0, 1.0))):
        raise ValueError("labels must be binary (0 or 1)")
    return array


def auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Return tie-aware AUROC; 0.5 when only one class is present."""

    label_array = _binary_labels(labels)
    _, score_array = _paired(label_array, scores)
    positives = float(np.sum(label_array))
    negatives = float(label_array.size - positives)
    if positives == 0.0 or negatives == 0.0:
        return 0.5
    ranks = _average_ranks(score_array)
    positive_rank_sum = float(np.sum(ranks[label_array == 1.0]))
    return (positive_rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives)


def brier_score(labels: ArrayLike, probabilities: ArrayLike) -> float:
    label_array = _binary_labels(labels)
    _, probability_array = _paired(label_array, probabilities)
    return float(np.mean(np.square(probability_array - label_array)))


def expected_calibration_error(
    labels: ArrayLike,
    probabilities: ArrayLike,
    *,
    bins: int = 10,
) -> float:
    """Return ECE with equal-width probability bins."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    label_array = _binary_labels(labels)
    _, probability_array = _paired(label_array, probabilities)
    if np.any(probability_array < 0.0) or np.any(probability_array > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(probability_array, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for bin_index in range(bins):
        selected = indices == bin_index
        count = float(np.sum(selected))
        if count == 0.0:
            continue
        gap = abs(
            float(np.mean(label_array[selected])) - float(np.mean(probability_array[selected]))
        )
        error += (count / label_array.size) * gap
    return error


def average_precision(labels: ArrayLike, scores: ArrayLike) -> float:
    """Area under the precision-recall curve (AP); base rate when degenerate."""

    label_array = _binary_labels(labels)
    _, score_array = _paired(label_array, scores)
    positives = float(np.sum(label_array))
    if positives == 0.0:
        return 0.0
    if positives == float(label_array.size):
        return 1.0
    order = np.argsort(-score_array, kind="mergesort")
    sorted_labels = label_array[order]
    cumulative_positives = np.cumsum(sorted_labels)
    precision = cumulative_positives / np.arange(1, label_array.size + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def negative_log_likelihood(
    labels: ArrayLike,
    probabilities: ArrayLike,
    *,
    epsilon: float = 1.0e-7,
) -> float:
    label_array = _binary_labels(labels)
    _, probability_array = _paired(label_array, probabilities)
    clipped = np.clip(probability_array, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(label_array * np.log(clipped) + (1.0 - label_array) * np.log(1.0 - clipped))
    )


def error_detection_auroc(
    absolute_errors: ArrayLike,
    uncertainties: ArrayLike,
    *,
    error_quantile: float = 0.75,
) -> float:
    """AUROC for flagging high-error predictions from uncertainty alone.

    A prediction is "high error" when its absolute error is at or above the
    given quantile of the error distribution; uncertainty is the score.
    Returns 0.5 when errors are too uniform to define both classes.
    """

    if not 0.0 < error_quantile < 1.0:
        raise ValueError("error_quantile must lie in (0, 1)")
    error_array, uncertainty_array = _paired(absolute_errors, uncertainties)
    threshold = float(np.quantile(error_array, error_quantile))
    labels = (error_array >= threshold).astype(np.float64)
    return auroc(labels, uncertainty_array)


def telescoping_residual(
    per_step_rewards: ArrayLike,
    start_values: ArrayLike,
    end_values: ArrayLike,
    *,
    axis: int = -1,
) -> float:
    """Return RMS residual of ``sum(rewards) == end_value - start_value``."""

    rewards = np.asarray(per_step_rewards, dtype=np.float64)
    if rewards.ndim == 0 or rewards.size == 0:
        raise ValueError("per_step_rewards must be a non-empty array")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("per_step_rewards must contain only finite values")
    starts = np.asarray(start_values, dtype=np.float64)
    ends = np.asarray(end_values, dtype=np.float64)
    residuals = rewards.sum(axis=axis) - (ends - starts)
    if not np.all(np.isfinite(residuals)):
        raise ValueError("start_values and end_values must be finite and broadcast-compatible")
    return float(np.sqrt(np.mean(np.square(residuals))))
