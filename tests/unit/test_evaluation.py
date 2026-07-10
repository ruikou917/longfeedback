"""Tests for E0 metrics and reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from longfeedback.evaluation import (
    pearson_correlation,
    plot_outcome_vs_credit,
    rmse,
    sign_accuracy,
    spearman_correlation,
    telescoping_residual,
    write_metrics_json,
)


def test_metrics_handle_ties_and_constants() -> None:
    assert spearman_correlation([1, 1, 2, 3], [10, 10, 20, 30]) == pytest.approx(1.0)
    assert spearman_correlation([1, 1, 1], [1, 2, 3]) == 0.0
    assert pearson_correlation([1, 1, 1], [1, 2, 3]) == 0.0
    assert sign_accuracy([0, 2, -3], [0, 1, -1]) == 1.0
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_telescoping_residual_is_zero_for_adjacent_differences() -> None:
    values = np.array([[0.1, 0.3, 0.9], [-0.2, 0.4, 0.0]])
    rewards = np.diff(values, axis=1)

    assert telescoping_residual(rewards, values[:, 0], values[:, -1]) == pytest.approx(0.0)


def test_metrics_json_is_strict_and_deterministic(tmp_path: Path) -> None:
    metrics = {"z": np.float64(2.0), "a": np.array([1, 2])}

    first = write_metrics_json(metrics, tmp_path / "first.json")
    second = write_metrics_json(metrics, tmp_path / "second.json")

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == {"a": [1, 2], "z": 2.0}


def test_plot_output_is_deterministic(tmp_path: Path) -> None:
    first = plot_outcome_vs_credit(
        [0.70, 0.82],
        [0.30, 0.76],
        tmp_path / "first.png",
        labels=["baseline", "docm"],
    )
    second = plot_outcome_vs_credit(
        [0.70, 0.82],
        [0.30, 0.76],
        tmp_path / "second.png",
        labels=["baseline", "docm"],
    )

    assert first.read_bytes() == second.read_bytes()
