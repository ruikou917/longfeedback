"""Tests for the DOCM sequence models (skipped without the research extra)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from longfeedback.models import (
    VARIANT_LOSS_WEIGHTS,
    DelayedOutcomeCreditModel,
    EncoderArchitecture,
    SequenceDataset,
    TrainingSettings,
    redistributed_rewards,
    variant_loss_weights,
)

_ARCHITECTURE = EncoderArchitecture(d_model=16, n_layers=1, n_heads=2)
_TRAINING = TrainingSettings(epochs=3, batch_size=16)


def _dataset(seed: int = 0, *, episodes: int = 24, horizon: int = 5) -> SequenceDataset:
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, 3, size=(episodes, horizon))
    mask = np.zeros((episodes, horizon), dtype=np.bool_)
    mask[: episodes // 2] = True
    return SequenceDataset(
        observations=rng.normal(size=(episodes, horizon, 4)),
        actions=actions,
        responses=rng.integers(0, 2, size=(episodes, horizon)).astype(np.float64),
        outcomes=rng.integers(0, 2, size=episodes).astype(np.float64),
        credit_targets=(actions == 1).astype(np.float64),
        credit_mask=mask,
        credit_se=np.full((episodes, horizon), 0.05),
    )


def _model(name: str, seed: int = 0) -> DelayedOutcomeCreditModel:
    return DelayedOutcomeCreditModel(
        observation_dim=4,
        n_actions=3,
        horizon=5,
        architecture=_ARCHITECTURE,
        loss_weights=variant_loss_weights(name),
        seed=seed,
    )


def test_variants_are_capacity_matched() -> None:
    counts = {name: _model(name).parameter_count() for name in VARIANT_LOSS_WEIGHTS}
    assert len(set(counts.values())) == 1


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown DOCM variant"):
        variant_loss_weights("docm_mystery")


def test_fit_and_predict_are_seed_deterministic() -> None:
    dataset = _dataset()
    first = _model("docm_credit", seed=3)
    second = _model("docm_credit", seed=3)
    first.fit(dataset, training=_TRAINING)
    second.fit(dataset, training=_TRAINING)
    assert np.array_equal(
        first.predict_outcome_probability(dataset),
        second.predict_outcome_probability(dataset),
    )
    assert np.array_equal(
        first.predict_logged_credit(dataset),
        second.predict_logged_credit(dataset),
    )


def test_heads_are_causal_at_every_step() -> None:
    dataset = _dataset()
    model = _model("docm_prefix")
    model.fit(dataset, training=_TRAINING)
    values = model.predict_prefix_values(dataset)
    action_values = model.predict_action_values(dataset)

    perturbed_step = 3
    observations = dataset.observations.copy()
    observations[:, perturbed_step, :] += 7.0
    actions = dataset.actions.copy()
    actions[:, perturbed_step] = (actions[:, perturbed_step] + 1) % 3
    perturbed = SequenceDataset(
        observations=observations,
        actions=actions,
        responses=dataset.responses,
        outcomes=dataset.outcomes,
    )
    perturbed_values = model.predict_prefix_values(perturbed)
    perturbed_action_values = model.predict_action_values(perturbed)

    # Prefix values strictly before the perturbed step are untouched; the
    # perturbed step's own value reads its observation and must change.
    assert np.allclose(values[:, :perturbed_step], perturbed_values[:, :perturbed_step])
    assert not np.allclose(values[:, perturbed_step], perturbed_values[:, perturbed_step])
    assert np.allclose(
        action_values[:, :perturbed_step], perturbed_action_values[:, :perturbed_step]
    )


def test_redistributed_rewards_telescope_exactly() -> None:
    dataset = _dataset()
    model = _model("docm_prefix")
    model.fit(dataset, training=_TRAINING)
    values = model.predict_prefix_values(dataset)
    rewards = redistributed_rewards(values)
    assert np.allclose(rewards.sum(axis=1), values[:, -1] - values[:, 0], atol=1.0e-12)


def test_credit_loss_is_masked_out_without_labels() -> None:
    dataset = _dataset()
    unlabeled = SequenceDataset(
        observations=dataset.observations,
        actions=dataset.actions,
        responses=dataset.responses,
        outcomes=dataset.outcomes,
    )
    summary = _model("docm_credit").fit(unlabeled, training=_TRAINING)
    assert "loss_credit" not in summary
    labeled_summary = _model("docm_credit").fit(dataset, training=_TRAINING)
    assert "loss_credit" in labeled_summary


def test_dataset_validation_rejects_misaligned_labels() -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="provided together"):
        SequenceDataset(
            observations=dataset.observations,
            actions=dataset.actions,
            responses=dataset.responses,
            outcomes=dataset.outcomes,
            credit_targets=dataset.credit_targets,
        )
    with pytest.raises(ValueError, match="actions must be"):
        SequenceDataset(
            observations=dataset.observations,
            actions=dataset.actions[:, :-1],
            responses=dataset.responses,
            outcomes=dataset.outcomes,
        )
