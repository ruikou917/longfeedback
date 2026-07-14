"""Deep-ensemble uncertainty with development-only conformal calibration.

Monte Carlo target noise is carried by binomial counts/standard errors in the
dataset; this module represents model uncertainty as between-member
disagreement of independently initialized critics, and calibrates intervals by
split conformal against high-K development targets. Calibration must never see
locked-reference rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from longfeedback.models.candidate_data import CandidateSequenceDataset
from longfeedback.models.candidate_docm import (
    CandidateDelayedOutcomeCreditModel,
    CandidateLossWeights,
    CandidateTrainingSettings,
    Parameterization,
)
from longfeedback.models.encoders import EncoderArchitecture

FloatArray = npt.NDArray[np.float64]


@dataclass
class CriticEnsemble:
    """Independently initialized critics sharing one training configuration."""

    state_dim: int
    action_dim: int
    max_horizon: int
    architecture: EncoderArchitecture
    loss_weights: CandidateLossWeights
    action_value_parameterization: Parameterization = "policy_centered_dueling"
    action_mlp_hidden: int = 128
    target_network_ema: float = 0.99
    members: int = 5
    base_seed: int = 0
    _models: list[CandidateDelayedOutcomeCreditModel] = field(
        default_factory=list, init=False, repr=False
    )

    def fit(
        self,
        dataset: CandidateSequenceDataset,
        *,
        training: CandidateTrainingSettings | None = None,
    ) -> None:
        if self.members <= 1:
            raise ValueError("an ensemble needs at least two members")
        self._models = []
        for member in range(self.members):
            model = CandidateDelayedOutcomeCreditModel(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_horizon=self.max_horizon,
                architecture=self.architecture,
                loss_weights=self.loss_weights,
                action_value_parameterization=self.action_value_parameterization,
                action_mlp_hidden=self.action_mlp_hidden,
                target_network_ema=self.target_network_ema,
                seed=self.base_seed + 1000 * member,
            )
            model.fit(dataset, training=training)
            self._models.append(model)

    def predict_q(self, dataset: CandidateSequenceDataset) -> tuple[FloatArray, FloatArray]:
        """(mean, member standard deviation) of candidate Q predictions."""

        if not self._models:
            raise RuntimeError("fit must be called before predict_q")
        stacked = np.stack([model.predict_branch_q(dataset) for model in self._models], axis=0)
        return stacked.mean(axis=0), stacked.std(axis=0)


@dataclass(frozen=True, slots=True)
class ConformalCalibration:
    """Split-conformal absolute-residual half-width around the ensemble mean."""

    target_coverage: float
    half_width: float
    calibration_rows: int

    def interval(self, mean: FloatArray) -> tuple[FloatArray, FloatArray]:
        lower = np.clip(mean - self.half_width, 0.0, 1.0)
        upper = np.clip(mean + self.half_width, 0.0, 1.0)
        return lower, upper

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "split_conformal_absolute_residual",
            "target_coverage": self.target_coverage,
            "half_width": self.half_width,
            "calibration_rows": self.calibration_rows,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> ConformalCalibration:
        return ConformalCalibration(
            target_coverage=float(payload["target_coverage"]),
            half_width=float(payload["half_width"]),
            calibration_rows=int(payload["calibration_rows"]),
        )


def fit_conformal(
    predictions: FloatArray, targets: FloatArray, *, target_coverage: float
) -> ConformalCalibration:
    """Fit the conformal quantile on development (never locked) residuals."""

    if predictions.shape != targets.shape or predictions.size == 0:
        raise ValueError("predictions and targets must be non-empty and aligned")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be in (0, 1)")
    scores = np.abs(predictions - targets).ravel()
    n = scores.size
    rank = min(n, math.ceil((n + 1) * target_coverage))
    half_width = float(np.sort(scores)[rank - 1])
    return ConformalCalibration(
        target_coverage=target_coverage, half_width=half_width, calibration_rows=n
    )


def coverage_report(
    calibration: ConformalCalibration, mean: FloatArray, reference: FloatArray
) -> dict[str, float]:
    """Empirical coverage and width against held-out high-K references."""

    lower, upper = calibration.interval(mean)
    covered = (reference >= lower) & (reference <= upper)
    return {
        "target_coverage": calibration.target_coverage,
        "empirical_coverage": float(covered.mean()) if covered.size else 0.0,
        "mean_interval_width": float((upper - lower).mean()) if covered.size else 0.0,
        "evaluated_rows": float(covered.size),
    }
