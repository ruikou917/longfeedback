"""Experiment metrics and reporting utilities."""

from .metrics import (
    auroc,
    average_precision,
    brier_score,
    error_detection_auroc,
    expected_calibration_error,
    negative_log_likelihood,
    pearson_correlation,
    rmse,
    sign_accuracy,
    spearman_correlation,
    telescoping_residual,
)
from .plotting import plot_outcome_vs_credit
from .reporting import write_metrics_json

__all__ = [
    "auroc",
    "average_precision",
    "brier_score",
    "error_detection_auroc",
    "expected_calibration_error",
    "negative_log_likelihood",
    "pearson_correlation",
    "plot_outcome_vs_credit",
    "rmse",
    "sign_accuracy",
    "spearman_correlation",
    "telescoping_residual",
    "write_metrics_json",
]
