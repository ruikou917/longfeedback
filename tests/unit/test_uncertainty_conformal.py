"""Split-conformal calibration and coverage reporting."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from longfeedback.models.uncertainty import (
    ConformalCalibration,
    coverage_report,
    fit_conformal,
)


def test_fit_conformal_uses_finite_sample_quantile() -> None:
    predictions = np.zeros(9, dtype=np.float64)
    targets = np.arange(1, 10, dtype=np.float64) / 10.0
    calibration = fit_conformal(predictions, targets, target_coverage=0.9)
    # ceil((9 + 1) * 0.9) = 9 -> the 9th smallest |residual| = 0.9.
    assert calibration.half_width == pytest.approx(0.9)
    assert calibration.calibration_rows == 9


def test_conformal_round_trip_and_interval_clipping() -> None:
    calibration = ConformalCalibration(target_coverage=0.9, half_width=0.3, calibration_rows=12)
    restored = ConformalCalibration.from_dict(calibration.as_dict())
    assert restored == calibration
    lower, upper = calibration.interval(np.asarray([0.1, 0.9]))
    assert lower[0] == 0.0 and upper[0] == pytest.approx(0.4)
    assert lower[1] == pytest.approx(0.6) and upper[1] == 1.0


def test_coverage_report_counts_hits() -> None:
    calibration = ConformalCalibration(target_coverage=0.8, half_width=0.1, calibration_rows=5)
    mean = np.asarray([0.5, 0.5, 0.5, 0.5])
    reference = np.asarray([0.55, 0.45, 0.75, 0.5])
    report = coverage_report(calibration, mean, reference)
    assert report["empirical_coverage"] == pytest.approx(0.75)
    assert report["mean_interval_width"] == pytest.approx(0.2)


def test_fit_conformal_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        fit_conformal(np.zeros(0), np.zeros(0), target_coverage=0.9)
    with pytest.raises(ValueError):
        fit_conformal(np.zeros(3), np.zeros(3), target_coverage=1.5)
