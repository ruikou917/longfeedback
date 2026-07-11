"""Tests for the softmax policy and REINFORCE step (skipped without torch)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from longfeedback.policies import (
    EpisodeBatch,
    SoftmaxPolicy,
    WorldPolicyAdapter,
    collect_episodes,
    mean_kl_divergence,
    reinforce_update,
)
from longfeedback.worlds import ProxyUtilityConfig, ProxyUtilityWorld


def _policy(seed: int = 0) -> SoftmaxPolicy:
    return SoftmaxPolicy(feature_dim=3, n_actions=4, seed=seed)


def test_probabilities_are_normalized_and_sampling_is_deterministic() -> None:
    policy = _policy()
    features = [0.2, -0.4, 1.0]
    probabilities = policy.probabilities(features)
    assert probabilities.shape == (4,)
    assert abs(float(probabilities.sum()) - 1.0) < 1.0e-6
    assert policy.sample(features, 0.31) == policy.sample(features, 0.31)
    assert policy.sample(features, 0.0) == 0 or probabilities[0] < 1.0e-6


def test_kl_is_zero_against_self_and_positive_against_other() -> None:
    policy = _policy(seed=1)
    other = _policy(seed=2)
    observations = np.random.default_rng(0).normal(size=(6, 5, 3))
    assert mean_kl_divergence(policy, policy.clone_detached(), observations) < 1.0e-9
    assert mean_kl_divergence(policy, other, observations) > 0.0


def test_reinforce_learns_the_dominant_action_on_a_bandit() -> None:
    """Reward = 1 when action 2 is chosen; the policy must concentrate on it."""

    torch.manual_seed(0)
    policy = _policy(seed=3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.2)
    rng = np.random.default_rng(0)
    features_batch = np.tile(np.asarray([1.0, 0.0, 0.0]), (32, 1, 1))
    for _ in range(60):
        draws = rng.random(32)
        actions = np.asarray(
            [[policy.sample([1.0, 0.0, 0.0], float(draw))] for draw in draws],
            dtype=np.int64,
        )
        batch = EpisodeBatch(
            observations=features_batch,
            actions=actions,
            responses=np.zeros((32, 1)),
            proxies=np.zeros(32),
            utilities=np.zeros(32),
        )
        rewards = (actions[:, 0] == 2).astype(np.float64)
        reinforce_update(policy, optimizer, batch, rewards, entropy_coefficient=0.0)
    final = policy.probabilities([1.0, 0.0, 0.0])
    assert final[2] > 0.9


def test_behavior_clone_matches_majority_behavior() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(400, 3))
    actions = (features[:, 0] > 0).astype(np.int64)  # action 1 iff feature 0 positive
    clone = SoftmaxPolicy.behavior_clone(features, actions, n_actions=4, epochs=200, seed=0)
    positive = clone.probabilities([2.0, 0.0, 0.0])
    negative = clone.probabilities([-2.0, 0.0, 0.0])
    assert positive[1] > 0.8
    assert negative[0] > 0.8


def test_collect_episodes_is_seed_deterministic_in_world_d() -> None:
    world = ProxyUtilityWorld(ProxyUtilityConfig(horizon=5))
    policy = SoftmaxPolicy(feature_dim=9, n_actions=5, seed=4)

    def feature_fn(observation: object) -> list[float]:
        from longfeedback.experiments.features import world_d_observation_features

        return world_d_observation_features(observation, horizon=5, n_actions=5)  # type: ignore[arg-type]

    adapter = WorldPolicyAdapter(policy=policy, feature_fn=feature_fn)
    first = collect_episodes(world, adapter, episodes=6, seed_base=11)
    second = collect_episodes(world, adapter, episodes=6, seed_base=11)
    assert np.array_equal(first.actions, second.actions)
    assert np.array_equal(first.utilities, second.utilities)
    assert first.observations.shape == (6, 5, 9)


def test_kl_penalty_requires_reference() -> None:
    policy = _policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
    batch = EpisodeBatch(
        observations=np.zeros((4, 2, 3)),
        actions=np.zeros((4, 2), dtype=np.int64),
        responses=np.zeros((4, 2)),
        proxies=np.zeros(4),
        utilities=np.zeros(4),
    )
    with pytest.raises(ValueError, match="reference policy"):
        reinforce_update(policy, optimizer, batch, np.zeros(4), kl_coefficient=0.1, reference=None)
