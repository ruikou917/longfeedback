from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from longfeedback.experiments.e8 import cluster_bootstrap_slopes, ols_slope


def test_ols_slope_recovers_linear_effect() -> None:
    x = np.asarray([-1.0, 0.0, 1.0, 2.0])
    y = 3.0 + 0.25 * x

    assert np.isclose(ols_slope(x, y), 0.25)


def test_cluster_bootstrap_is_reproducible_and_clusters_users() -> None:
    x = np.asarray([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    y = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    users = np.asarray(["a", "a", "a", "b", "b", "b"])

    first = cluster_bootstrap_slopes(x, y, users, resamples=100, seed=7)
    second = cluster_bootstrap_slopes(x, y, users, resamples=100, seed=7)

    assert np.array_equal(first, second)
    assert first.shape == (100,)
    assert np.all(np.isfinite(first))
