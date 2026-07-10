"""Tests for leakage-safe E0 features."""

from __future__ import annotations

import numpy as np

from longfeedback.experiments.features import (
    action_sequence_features,
    deterministic_split,
    prefix_action_features,
)


def test_prefix_features_never_include_future_actions() -> None:
    actions = [1, 2, 1]
    encoded = prefix_action_features(actions, horizon=3, n_actions=3)

    assert np.count_nonzero(encoded[0]) == 0
    assert encoded[1, 1] == 1.0
    assert np.count_nonzero(encoded[1, 3:]) == 0
    assert encoded[2, 3 + 2] == 1.0
    assert np.count_nonzero(encoded[2, 6:]) == 0


def test_full_sequence_features_are_position_aware() -> None:
    encoded = action_sequence_features([[1, 0]], horizon=2, n_actions=3)

    assert encoded.shape == (1, 6)
    assert np.array_equal(encoded[0], np.array([0, 1, 0, 1, 0, 0], dtype=float))


def test_deterministic_split_is_reproducible_and_disjoint() -> None:
    train_a, test_a = deterministic_split(100, train_fraction=0.8, seed=7)
    train_b, test_b = deterministic_split(100, train_fraction=0.8, seed=7)

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(test_a, test_b)
    assert set(train_a).isdisjoint(set(test_a))
    assert sorted(np.concatenate((train_a, test_a)).tolist()) == list(range(100))
