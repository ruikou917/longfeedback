"""RUDDER-style return redistribution from learned prefix values."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from longfeedback.baselines import RidgeBaseline


def _prefix_features(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(
            f"prefix_features must have shape [trajectories, prefixes, features], got {array.shape}"
        )
    if min(array.shape) == 0 or array.shape[1] < 2:
        raise ValueError("prefix_features require at least two non-empty prefixes")
    if not np.all(np.isfinite(array)):
        raise ValueError("prefix_features must contain only finite values")
    return array


def _mask(values: ArrayLike | None, shape: tuple[int, int]) -> NDArray[np.bool_]:
    if values is None:
        return np.ones(shape, dtype=np.bool_)
    mask = np.asarray(values, dtype=np.bool_)
    if mask.shape != shape:
        raise ValueError(f"mask must have shape {shape}, got {mask.shape}")
    if np.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError("mask must describe left-aligned contiguous prefixes")
    return mask


def redistribute_prefix_values(prefix_values: ArrayLike) -> NDArray[np.float64]:
    """Return adjacent prefix-value differences that telescope exactly."""

    values = np.asarray(prefix_values, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[-1] < 2:
        raise ValueError("prefix_values must be a vector or matrix with at least two prefixes")
    if not np.all(np.isfinite(values)):
        raise ValueError("prefix_values must contain only finite values")
    return np.diff(values, axis=-1)


class RudderRedistributor:
    """Fit a prefix outcome predictor and redistribute value differences."""

    def __init__(self, alpha: float = 1.0, *, fit_intercept: bool = True) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(
        self,
        prefix_features: ArrayLike,
        outcomes: ArrayLike,
        mask: ArrayLike | None = None,
    ) -> RudderRedistributor:
        features = _prefix_features(prefix_features)
        trajectories, prefixes, feature_count = features.shape
        targets = np.asarray(outcomes, dtype=np.float64)
        if targets.shape != (trajectories,):
            raise ValueError(f"outcomes must have shape {(trajectories,)}, got {targets.shape}")
        if not np.all(np.isfinite(targets)):
            raise ValueError("outcomes must contain only finite values")

        valid = _mask(mask, (trajectories, prefixes)).reshape(-1)
        if not np.any(valid):
            raise ValueError("mask selects no training prefixes")

        flat_features = features.reshape(-1, feature_count)
        flat_targets = np.repeat(targets, prefixes)
        self.model_ = RidgeBaseline(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
        ).fit(flat_features[valid], flat_targets[valid])
        return self

    def predict_values(
        self,
        prefix_features: ArrayLike,
        mask: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        if not hasattr(self, "model_"):
            raise RuntimeError("RudderRedistributor must be fitted before prediction")
        features = _prefix_features(prefix_features)
        trajectories, prefixes, feature_count = features.shape
        values = self.model_.predict(features.reshape(-1, feature_count)).reshape(
            trajectories, prefixes
        )
        if mask is not None:
            valid = _mask(mask, (trajectories, prefixes))
            values = np.where(valid, values, np.nan)
        return values

    def redistribute(
        self,
        prefix_features: ArrayLike,
        mask: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        features = _prefix_features(prefix_features)
        values = self.predict_values(features)
        rewards = redistribute_prefix_values(values)
        if mask is not None:
            valid = _mask(mask, values.shape)
            rewards = np.where(valid[:, :-1] & valid[:, 1:], rewards, 0.0)
        return rewards
