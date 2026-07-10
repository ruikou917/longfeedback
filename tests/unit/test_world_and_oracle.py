"""Tests for deterministic World A and paired counterfactual credit."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

from longfeedback.credit.oracle import (
    ContinuationMode,
    counterfactual_pair,
    estimate_oracle_credit,
    exact_deterministic_credit,
)
from longfeedback.experiments.e0 import hand_derived_one_step_credit_check
from longfeedback.worlds import (
    FatigueAction,
    FatigueHabitConfig,
    FatigueHabitObservation,
    FatigueHabitWorld,
)


def _episode(world: FatigueHabitWorld, seed: int = 7):
    actions = (
        FatigueAction.HELPFUL,
        FatigueAction.URGENT,
        FatigueAction.NOOP,
        FatigueAction.HELPFUL,
    )
    return world.rollout(actions, world.sample_exogenous(seed))


def test_seeded_world_rollout_is_deterministic_and_immutable() -> None:
    world = FatigueHabitWorld(FatigueHabitConfig(horizon=4))

    first = _episode(world)
    second = _episode(world)

    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.transitions[0].reward = 1.0  # type: ignore[misc]


def test_paired_counterfactual_reuses_noise_and_frozen_actions() -> None:
    world = FatigueHabitWorld(FatigueHabitConfig(horizon=4))
    episode = _episode(world)

    pair = counterfactual_pair(
        world,
        episode,
        step_index=1,
        action=FatigueAction.HELPFUL,
        reference_action=FatigueAction.NOOP,
        continuation_mode=ContinuationMode.FROZEN,
    )

    assert pair.paired_noise_reused
    assert pair.treated_episode.transitions[0].state == pair.reference_episode.transitions[0].state
    assert pair.treated_episode.actions[1:] == pair.reference_episode.actions[1:]
    assert pair.treated_episode.actions[0] is FatigueAction.HELPFUL
    assert pair.reference_episode.actions[0] is FatigueAction.NOOP


def test_deterministic_oracle_has_zero_mc_variance_and_matches_exact() -> None:
    world = FatigueHabitWorld(FatigueHabitConfig(horizon=4))
    episode = _episode(world)
    exact = exact_deterministic_credit(
        world,
        episode,
        step_index=0,
        action=FatigueAction.URGENT,
        reference_action=FatigueAction.NOOP,
    )

    estimate = estimate_oracle_credit(
        world,
        episode,
        step_index=0,
        action=FatigueAction.URGENT,
        reference_action=FatigueAction.NOOP,
        num_rollouts=8,
        base_seed=10,
    )

    assert estimate.credit_utility == pytest.approx(exact.credit_utility)
    assert estimate.monte_carlo_se == 0.0


def test_oracle_matches_independent_hand_derived_nonzero_credit() -> None:
    world = FatigueHabitWorld(FatigueHabitConfig(horizon=4))

    oracle, analytic, absolute_error = hand_derived_one_step_credit_check(world)

    assert analytic != 0.0
    assert oracle == pytest.approx(analytic, abs=1.0e-12)
    assert absolute_error <= 1.0e-12


def test_actions_have_zero_credit_when_action_mechanisms_are_disabled() -> None:
    world = FatigueHabitWorld(FatigueHabitConfig.effects_disabled(horizon=4))
    episode = world.rollout(
        (FatigueAction.NOOP,) * 4,
        world.sample_exogenous(0),
    )

    pair = exact_deterministic_credit(
        world,
        episode,
        step_index=2,
        action=FatigueAction.URGENT,
        reference_action=FatigueAction.NOOP,
    )

    assert pair.credit_utility == 0.0
    assert pair.credit_proxy == 0.0


@dataclass(frozen=True)
class _FatigueAwarePolicy:
    def select_action(
        self,
        observation: FatigueHabitObservation,
        *,
        step_index: int,
        random_value: float,
    ) -> FatigueAction:
        del step_index, random_value
        return FatigueAction.NOOP if observation.fatigue > 0.5 else FatigueAction.HELPFUL


def test_policy_reactive_continuation_requires_and_uses_policy() -> None:
    world = FatigueHabitWorld(FatigueHabitConfig(horizon=4))
    episode = _episode(world)

    with pytest.raises(ValueError, match="future_policy"):
        counterfactual_pair(
            world,
            episode,
            step_index=0,
            action=FatigueAction.URGENT,
            reference_action=FatigueAction.NOOP,
            continuation_mode=ContinuationMode.POLICY_REACTIVE,
        )

    pair = counterfactual_pair(
        world,
        episode,
        step_index=0,
        action=FatigueAction.URGENT,
        reference_action=FatigueAction.NOOP,
        continuation_mode=ContinuationMode.POLICY_REACTIVE,
        future_policy=_FatigueAwarePolicy(),
    )

    assert pair.paired_noise_reused
