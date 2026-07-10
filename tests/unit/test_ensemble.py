"""Tests for the bootstrap DOCM ensemble (skipped without torch)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from longfeedback.models import (
    BootstrapEnsemble,
    EncoderArchitecture,
    SequenceDataset,
    TrainingSettings,
    variant_loss_weights,
)

_ARCHITECTURE = EncoderArchitecture(d_model=16, n_layers=1, n_heads=2)
_TRAINING = TrainingSettings(epochs=2, batch_size=16)


def _dataset(seed: int = 0) -> SequenceDataset:
    rng = np.random.default_rng(seed)
    episodes, horizon = 24, 5
    actions = rng.integers(0, 3, size=(episodes, horizon))
    mask = np.ones((episodes, horizon), dtype=np.bool_)
    return SequenceDataset(
        observations=rng.normal(size=(episodes, horizon, 4)),
        actions=actions,
        responses=rng.integers(0, 2, size=(episodes, horizon)).astype(np.float64),
        outcomes=rng.integers(0, 2, size=episodes).astype(np.float64),
        credit_targets=(actions == 1).astype(np.float64),
        credit_mask=mask,
        credit_se=np.full((episodes, horizon), 0.05),
    )


def _ensemble(members: int = 3, seed: int = 0) -> BootstrapEnsemble:
    return BootstrapEnsemble(
        observation_dim=4,
        n_actions=3,
        horizon=5,
        architecture=_ARCHITECTURE,
        loss_weights=variant_loss_weights("docm_credit"),
        members=members,
        seed=seed,
    )


def test_ensemble_requires_at_least_two_members() -> None:
    with pytest.raises(ValueError, match="at least two members"):
        _ensemble(members=1)


def test_ensemble_mean_and_uncertainty_shapes() -> None:
    dataset = _dataset()
    ensemble = _ensemble()
    ensemble.fit(dataset, training=_TRAINING)
    outcome_mean, outcome_std = ensemble.predict_outcome_probability(dataset)
    credit_mean, credit_std = ensemble.predict_logged_credit(dataset)
    assert outcome_mean.shape == outcome_std.shape == (dataset.episodes,)
    assert credit_mean.shape == credit_std.shape == (dataset.episodes, dataset.horizon)
    # Members differ by init seed and bootstrap resample, so disagreement
    # (epistemic uncertainty) must be strictly positive somewhere.
    assert np.all(outcome_std >= 0.0)
    assert float(np.max(outcome_std)) > 0.0
    assert float(np.max(credit_std)) > 0.0


def test_ensemble_is_seed_deterministic() -> None:
    dataset = _dataset()
    first = _ensemble(seed=7)
    second = _ensemble(seed=7)
    first.fit(dataset, training=_TRAINING)
    second.fit(dataset, training=_TRAINING)
    first_mean, first_std = first.predict_outcome_probability(dataset)
    second_mean, second_std = second.predict_outcome_probability(dataset)
    assert np.array_equal(first_mean, second_mean)
    assert np.array_equal(first_std, second_std)
