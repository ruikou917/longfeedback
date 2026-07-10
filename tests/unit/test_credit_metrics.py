"""Tests for credit-recovery and probabilistic outcome metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from longfeedback.credit.metrics import (
    credit_recovery_summary,
    kendall_tau,
    normalized_rmse,
    spearman_by_temporal_distance,
)
from longfeedback.evaluation import (
    auroc,
    average_precision,
    brier_score,
    error_detection_auroc,
    expected_calibration_error,
    negative_log_likelihood,
)


def test_kendall_tau_hand_computed_cases() -> None:
    assert kendall_tau([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == 1.0
    assert kendall_tau([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == -1.0
    # One discordant pair among three: (2,3) reversed -> (2 - 1) / 3.
    assert math.isclose(kendall_tau([1.0, 2.0, 3.0], [1.0, 3.0, 2.0]), 1.0 / 3.0)
    assert kendall_tau([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


def test_normalized_rmse_scales_by_target_deviation() -> None:
    expected = np.asarray([0.0, 2.0])
    assert normalized_rmse(expected, expected) == 0.0
    assert math.isclose(normalized_rmse(expected, expected + 1.0), 1.0)
    assert normalized_rmse([1.0, 1.0], [1.0, 1.0]) == 0.0
    assert normalized_rmse([1.0, 1.0], [2.0, 2.0]) == float("inf")


def test_credit_recovery_summary_reports_design_doc_metrics() -> None:
    summary = credit_recovery_summary([1.0, -1.0, 2.0, 0.5], [0.9, -0.8, 1.5, 0.2])
    assert set(summary) == {
        "pearson",
        "spearman",
        "kendall_tau",
        "sign_accuracy",
        "normalized_rmse",
        "examples",
    }
    assert summary["sign_accuracy"] == 1.0
    assert math.isclose(summary["spearman"], 1.0)
    assert summary["examples"] == 4.0


def test_spearman_by_temporal_distance_groups_and_filters() -> None:
    expected = np.asarray([1.0, 2.0, 3.0, 4.0] * 4)
    predicted = np.concatenate((expected[:8], -expected[8:]))
    distances = np.asarray([1] * 8 + [2] * 8)
    grouped = spearman_by_temporal_distance(expected, predicted, distances, min_group_size=4)
    assert math.isclose(grouped["1"], 1.0)
    assert math.isclose(grouped["2"], -1.0)
    # Groups below the minimum size are omitted.
    tiny = spearman_by_temporal_distance(expected, predicted, distances, min_group_size=9)
    assert tiny == {}


def test_auroc_hand_computed_and_single_class_convention() -> None:
    labels = [0.0, 0.0, 1.0, 1.0]
    assert auroc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert math.isclose(auroc(labels, [0.5, 0.5, 0.5, 0.5]), 0.5)
    assert auroc([1.0, 1.0], [0.2, 0.9]) == 0.5


def test_brier_and_ece_hand_computed() -> None:
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert math.isclose(brier_score([1.0, 0.0], [0.5, 0.5]), 0.25)
    assert expected_calibration_error([1.0, 0.0], [1.0, 0.0]) == 0.0
    # All predictions 0.7 in one bin with a 0.5 event rate -> gap 0.2.
    assert math.isclose(
        expected_calibration_error([1.0, 0.0], [0.7, 0.7]),
        0.2,
        abs_tol=1.0e-12,
    )


def test_average_precision_hand_computed() -> None:
    # Ranked: pos, neg, pos -> AP = (1/1 + 2/3) / 2 = 5/6.
    assert math.isclose(average_precision([1.0, 0.0, 1.0], [0.9, 0.8, 0.7]), 5.0 / 6.0)
    assert average_precision([0.0, 0.0], [0.4, 0.6]) == 0.0
    assert average_precision([1.0, 1.0], [0.4, 0.6]) == 1.0


def test_negative_log_likelihood_hand_computed() -> None:
    assert math.isclose(
        negative_log_likelihood([1.0, 0.0], [0.5, 0.5]), math.log(2.0), rel_tol=1.0e-9
    )
    # Confident-correct beats uncertain; clipping keeps it finite.
    assert negative_log_likelihood([1.0], [1.0]) < 1.0e-5
    assert math.isfinite(negative_log_likelihood([1.0], [0.0]))


def test_error_detection_auroc_flags_high_error_predictions() -> None:
    errors = [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 1.3, 1.4]
    aligned_uncertainty = [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 1.3, 1.4]
    inverted_uncertainty = aligned_uncertainty[::-1]
    assert error_detection_auroc(errors, aligned_uncertainty) == 1.0
    assert error_detection_auroc(errors, inverted_uncertainty) == 0.0
    assert error_detection_auroc(errors, [0.5] * 8) == 0.5
    with pytest.raises(ValueError):
        error_detection_auroc(errors, aligned_uncertainty, error_quantile=1.5)


def test_metric_input_validation() -> None:
    with pytest.raises(ValueError):
        kendall_tau([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        auroc([0.5, 1.0], [0.1, 0.2])
    with pytest.raises(ValueError):
        expected_calibration_error([0.0, 1.0], [1.5, 0.2])
    with pytest.raises(ValueError):
        spearman_by_temporal_distance([1.0, 2.0], [1.0, 2.0], [-1, 0])
