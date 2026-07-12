from __future__ import annotations

import numpy as np

from longfeedback.experiments.e10_hs_day import _cluster_effect_draws


def test_cluster_effect_recovers_known_randomized_difference() -> None:
    users = np.repeat(np.asarray([f"u{index}" for index in range(20)]), 4)
    action = np.tile(np.asarray([False, True, False, True]), 20)
    user_offsets = np.repeat(np.linspace(-1.0, 1.0, 20), 4)
    outcome = user_offsets + 0.25 * action

    estimate, draws = _cluster_effect_draws(
        outcome,
        action,
        users,
        resamples=200,
        seed=9,
    )

    assert np.isclose(estimate, 0.25)
    assert len(draws) == 200
    assert np.allclose(draws, 0.25)
