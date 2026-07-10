"""Small NumPy regression baselines with sklearn-like APIs."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _features(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"features must have shape [samples, features], got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("features must contain only finite values")
    return array


def _targets(values: ArrayLike, samples: int) -> tuple[NDArray[np.float64], bool]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError(
            f"targets must have shape [samples] or [samples, targets], got {array.shape}"
        )
    if array.shape[0] != samples:
        raise ValueError(f"feature/target sample mismatch: {samples} != {array.shape[0]}")
    if not np.all(np.isfinite(array)):
        raise ValueError("targets must contain only finite values")
    was_vector = array.ndim == 1
    return (array[:, None] if was_vector else array), was_vector


class MeanOutcomeBaseline:
    """Predict the training-set mean outcome for every sample."""

    def fit(self, features: ArrayLike, targets: ArrayLike) -> MeanOutcomeBaseline:
        x = _features(features)
        y, self._was_vector = _targets(targets, x.shape[0])
        self._mean = y.mean(axis=0)
        self.mean_ = float(self._mean[0]) if self._was_vector else self._mean.copy()
        return self

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        x = _features(features)
        if not hasattr(self, "_mean"):
            raise RuntimeError("MeanOutcomeBaseline must be fitted before prediction")
        prediction = np.broadcast_to(self._mean, (x.shape[0], self._mean.size)).copy()
        return prediction[:, 0] if self._was_vector else prediction


class RidgeBaseline:
    """Deterministic ridge regression for outcome or credit targets."""

    def __init__(self, alpha: float = 1.0, *, fit_intercept: bool = True) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept

    def fit(self, features: ArrayLike, targets: ArrayLike) -> RidgeBaseline:
        x = _features(features)
        y, self._was_vector = _targets(targets, x.shape[0])

        if self.fit_intercept:
            x_offset = x.mean(axis=0)
            y_offset = y.mean(axis=0)
        else:
            x_offset = np.zeros(x.shape[1], dtype=np.float64)
            y_offset = np.zeros(y.shape[1], dtype=np.float64)

        centered_x = x - x_offset
        centered_y = y - y_offset

        if self.alpha == 0:
            coefficients, *_ = np.linalg.lstsq(centered_x, centered_y, rcond=None)
        else:
            gram = centered_x.T @ centered_x
            gram.flat[:: gram.shape[0] + 1] += self.alpha
            coefficients = np.linalg.solve(gram, centered_x.T @ centered_y)

        intercept = y_offset - x_offset @ coefficients
        self._coefficients = coefficients
        self._intercept = intercept
        self.coef_ = coefficients[:, 0].copy() if self._was_vector else coefficients.copy()
        self.intercept_ = float(intercept[0]) if self._was_vector else intercept.copy()
        return self

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        x = _features(features)
        if not hasattr(self, "_coefficients"):
            raise RuntimeError("RidgeBaseline must be fitted before prediction")
        if x.shape[1] != self._coefficients.shape[0]:
            raise ValueError(f"expected {self._coefficients.shape[0]} features, got {x.shape[1]}")
        prediction = x @ self._coefficients + self._intercept
        if self._was_vector:
            return np.asarray(prediction[:, 0], dtype=np.float64)
        return np.asarray(prediction, dtype=np.float64)
