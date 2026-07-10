"""Tests for deterministic diagnostic baselines."""

from __future__ import annotations

import numpy as np

from longfeedback.baselines import MeanOutcomeBaseline, RidgeBaseline
from longfeedback.credit.rudder import RudderRedistributor, redistribute_prefix_values


def test_ridge_exactly_recovers_linear_targets() -> None:
    features = np.array([[-2.0, 1.0], [-1.0, 2.0], [0.0, -1.0], [1.0, 0.5], [2.0, 3.0]])
    targets = 1.5 + 2.0 * features[:, 0] - 3.0 * features[:, 1]

    model = RidgeBaseline(alpha=0.0).fit(features, targets)

    np.testing.assert_allclose(model.predict(features), targets, atol=1e-12)


def test_mean_and_ridge_outputs_are_deterministic() -> None:
    features = np.arange(12, dtype=float).reshape(6, 2)
    targets = np.array([0.0, 1.0, 0.0, 1.0, 2.0, 2.0])
    factories = (MeanOutcomeBaseline, lambda: RidgeBaseline(alpha=0.25))

    for factory in factories:
        first = factory().fit(features, targets).predict(features)
        second = factory().fit(features, targets).predict(features)
        np.testing.assert_array_equal(first, second)


def test_rudder_differences_telescope_exactly() -> None:
    values = np.array([[0.2, 0.5, 0.4, 1.2], [-0.1, 0.0, 0.8, 0.3]])

    rewards = redistribute_prefix_values(values)

    np.testing.assert_allclose(rewards.sum(axis=1), values[:, -1] - values[:, 0])


def test_fitted_rudder_is_deterministic_and_telescopes() -> None:
    base = np.array([[0.0, 1.0], [1.0, -1.0], [2.0, 0.5], [-1.0, 2.0]])
    times = np.arange(4, dtype=float)
    prefixes = np.stack(
        [np.column_stack((base[:, 0], base[:, 1], np.full(base.shape[0], time))) for time in times],
        axis=1,
    )
    outcomes = 2.0 * base[:, 0] - base[:, 1]
    first = RudderRedistributor(alpha=0.0).fit(prefixes, outcomes)
    second = RudderRedistributor(alpha=0.0).fit(prefixes, outcomes)

    first_values = first.predict_values(prefixes)
    first_rewards = first.redistribute(prefixes)

    np.testing.assert_array_equal(first_values, second.predict_values(prefixes))
    np.testing.assert_allclose(
        first_rewards.sum(axis=1),
        first_values[:, -1] - first_values[:, 0],
        atol=1e-12,
    )
