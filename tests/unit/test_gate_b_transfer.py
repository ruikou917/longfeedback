"""Tests for the Gate B leave-one-family-out uncertainty transfer criterion."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from longfeedback.config import GateBConfig
from longfeedback.experiments.gate_b import (
    FAMILY_NAMES,
    UncertaintyEvaluation,
    _balanced_accuracy,
    _best_threshold_balanced_accuracy,
    _high_error_labels,
    _standardize,
    leave_one_family_out_transfer,
)


def test_standardize_zero_variance_returns_zeros() -> None:
    constant = np.full(5, 3.0)
    assert np.all(_standardize(constant) == 0.0)


def test_high_error_labels_splits_on_family_median() -> None:
    errors = np.array([1.0, 2.0, 3.0, 4.0])
    labels = _high_error_labels(errors)
    assert list(labels) == [0.0, 0.0, 1.0, 1.0]


def test_best_threshold_balanced_accuracy_separates_perfectly_separable_data() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    threshold, accuracy = _best_threshold_balanced_accuracy(scores, labels)
    assert accuracy == pytest.approx(1.0)
    assert _balanced_accuracy(scores, labels, threshold) == pytest.approx(1.0)


def test_best_threshold_handles_single_class_gracefully() -> None:
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.zeros(3)
    _threshold, accuracy = _best_threshold_balanced_accuracy(scores, labels)
    assert accuracy == pytest.approx(0.5)


def _evaluation(uncertainties: np.ndarray, errors: np.ndarray) -> UncertaintyEvaluation:
    return UncertaintyEvaluation(summary={}, errors=errors, uncertainties=uncertainties)


def test_leave_one_family_out_transfer_detects_a_generalizing_relationship() -> None:
    rng = np.random.default_rng(0)
    raw: dict[str, UncertaintyEvaluation] = {}
    for name in FAMILY_NAMES:
        # Same relationship (uncertainty predicts error) in every family, up
        # to family-specific scale/offset -- standardizing should recover it.
        uncertainty = rng.uniform(0.0, 1.0, size=200) * rng.uniform(1.0, 5.0)
        errors = uncertainty + rng.normal(0.0, 0.05, size=200)
        raw[name] = _evaluation(uncertainty, errors)

    config = GateBConfig()
    result = leave_one_family_out_transfer(raw, config)
    assert result["winning_transfers"] == 4
    assert result["pass"] is True
    for family_result in result["per_family"].values():
        assert family_result["transfer_balanced_accuracy"] > 0.5


def test_leave_one_family_out_transfer_reports_failure_honestly() -> None:
    rng = np.random.default_rng(1)
    raw: dict[str, UncertaintyEvaluation] = {}
    for name in FAMILY_NAMES:
        # Pure noise: uncertainty carries no information about error anywhere.
        uncertainty = rng.uniform(0.0, 1.0, size=200)
        errors = rng.uniform(0.0, 1.0, size=200)
        raw[name] = _evaluation(uncertainty, errors)

    config = GateBConfig()
    result = leave_one_family_out_transfer(raw, config)
    assert result["winning_transfers"] < config.decision.min_winning_transfers
    assert result["pass"] is False
