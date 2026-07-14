"""Shared group-policy objective and trainable candidate policy."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")

import torch

from longfeedback.actors.trainable import TrainableSoftmaxCandidatePolicy
from longfeedback.models.text_embeddings import HashedTextEmbedder
from longfeedback.training.group_policy import (
    GroupPolicySettings,
    PolicyUpdateStep,
    center_advantages,
    clipped_surrogate,
    group_policy_update,
)


def test_center_advantages_conventions() -> None:
    assert center_advantages([1.0]) == [0.0]
    assert center_advantages([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]
    centered = center_advantages([1.0, 3.0])
    assert centered[0] == pytest.approx(-1.0)
    assert centered[1] == pytest.approx(1.0)
    uncentered = center_advantages([1.0, 3.0], normalize=False)
    assert uncentered == [-1.0, 1.0]


def test_clipped_surrogate_clips_large_ratios() -> None:
    new = torch.tensor(math.log(2.0))
    old = torch.tensor(0.0)
    positive = clipped_surrogate(new, old, torch.tensor(1.0), ratio_clip=0.2)
    assert float(positive) == pytest.approx(1.2)
    negative = clipped_surrogate(new, old, torch.tensor(-1.0), ratio_clip=0.2)
    assert float(negative) == pytest.approx(-2.0)


def test_trainable_policy_is_deterministic_and_id_tracks_weights() -> None:
    embedder = HashedTextEmbedder(dim=16)
    policy = TrainableSoftmaxCandidatePolicy(embedder, seed=0)
    twin = TrainableSoftmaxCandidatePolicy(embedder, seed=0)
    assert policy.policy_id == twin.policy_id
    scores = policy.score("prompt", ("press red 1", "smash blue 2"))
    assert scores == twin.score("prompt", ("press red 1", "smash blue 2"))
    before = policy.policy_id
    with torch.no_grad():
        policy.parameters()[0].add_(1.0)
    assert policy.policy_id != before


def test_group_policy_update_increases_good_action_probability() -> None:
    embedder = HashedTextEmbedder(dim=16)
    policy = TrainableSoftmaxCandidatePolicy(embedder, seed=1)
    reference = TrainableSoftmaxCandidatePolicy(embedder, seed=1)
    prompt = "goal: win | observation: panel"
    candidates = ("press red 1", "smash blue 2")
    scores = policy.score(prompt, candidates)
    good = scores.candidates.index("press red 1")
    steps = [
        PolicyUpdateStep(
            prompt=prompt,
            candidates=candidates,
            chosen_index=good,
            old_log_probability=scores.log_probabilities[good],
            advantage=1.0,
        ),
        PolicyUpdateStep(
            prompt=prompt,
            candidates=candidates,
            chosen_index=1 - good,
            old_log_probability=scores.log_probabilities[1 - good],
            advantage=-1.0,
        ),
    ]
    before = policy.score(prompt, candidates).probabilities[good]
    stats = group_policy_update(
        policy,
        reference,
        steps,
        GroupPolicySettings(update_epochs=5, learning_rate=0.1),
    )
    after = policy.score(prompt, candidates).probabilities[good]
    assert after > before
    assert stats["updated_steps"] == 2.0
    # The immutable reference is untouched by the update.
    assert reference.score(prompt, candidates).probabilities[good] == pytest.approx(before)


def test_group_policy_update_handles_empty_batch() -> None:
    embedder = HashedTextEmbedder(dim=16)
    policy = TrainableSoftmaxCandidatePolicy(embedder, seed=2)
    stats = group_policy_update(policy, policy, [], GroupPolicySettings())
    assert stats["updated_steps"] == 0.0
