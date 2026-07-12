from __future__ import annotations

import numpy as np

from longfeedback.experiments.e9 import crossfit_arm_predictions, distal_scores


def test_distal_score_recovers_constant_randomized_effect() -> None:
    action = np.asarray([False, True] * 20)
    probability = np.full(len(action), 0.5)
    mu0 = np.full(len(action), 10.0)
    mu1 = np.full(len(action), 12.0)
    outcome = np.where(action, mu1, mu0)

    scores = distal_scores(outcome, action, probability, mu0, mu1)

    assert np.allclose(scores, 2.0)


def test_crossfit_predictions_hold_out_whole_users() -> None:
    users = np.asarray([f"u{index // 4}" for index in range(40)])
    action = np.asarray([False, True, False, True] * 10)
    feature = np.arange(40, dtype=np.float64) / 40.0
    x = np.column_stack([np.ones(40), feature])
    outcome = 1.0 + feature + 0.5 * action

    mu0, mu1 = crossfit_arm_predictions(x, outcome, action, users, folds=2, alpha=0.01)

    assert mu0.shape == outcome.shape
    assert mu1.shape == outcome.shape
    assert np.all(np.isfinite(mu0))
    assert np.all(np.isfinite(mu1))
    assert np.mean(mu1 - mu0) > 0.4
