"""Tests for World A observability modes and the stochastic preset."""

from __future__ import annotations

import pytest

from longfeedback.worlds import (
    FatigueAction,
    FatigueHabitConfig,
    FatigueHabitWorld,
    FatigueObservability,
)


def _stepped_observation(world: FatigueHabitWorld):
    noise = world.sample_exogenous(5)
    transition = world.step(world.initial_state(), FatigueAction.HELPFUL, noise[0])
    return transition, noise


def test_oracle_observability_reveals_exact_state() -> None:
    world = FatigueHabitWorld(FatigueHabitConfig.stochastic(observability="oracle"))
    transition, _ = _stepped_observation(world)
    observation = transition.next_observation
    assert observation.habit == transition.next_state.habit
    assert observation.fatigue == transition.next_state.fatigue
    assert observation.habituation == transition.next_state.habituation


def test_noisy_observability_offsets_by_seeded_observation_noise() -> None:
    world = FatigueHabitWorld(FatigueHabitConfig.stochastic(observability="noisy"))
    transition, noise = _stepped_observation(world)
    observation = transition.next_observation
    assert observation.habit == transition.next_state.habit + noise[0].habit_observation_noise
    assert observation.fatigue is not None
    assert observation.fatigue == transition.next_state.fatigue + noise[0].fatigue_observation_noise


def test_partial_observability_hides_fatigue_and_habituation() -> None:
    world = FatigueHabitWorld(FatigueHabitConfig.stochastic(observability="partial"))
    transition, _ = _stepped_observation(world)
    observation = transition.next_observation
    assert observation.fatigue is None
    assert observation.habituation is None
    assert observation.last_response == transition.next_state.last_response


def test_observation_noise_stream_does_not_perturb_dynamics_draws() -> None:
    """Enabling observation noise must not change existing E0-style seeds."""

    silent = FatigueHabitWorld(FatigueHabitConfig())
    noisy = FatigueHabitWorld(
        FatigueHabitConfig(observability=FatigueObservability.NOISY, observation_noise_std=0.5)
    )
    for seed in range(5):
        silent_steps = silent.sample_exogenous(seed)
        noisy_steps = noisy.sample_exogenous(seed)
        for left, right in zip(silent_steps, noisy_steps, strict=True):
            assert left.response_noise == right.response_noise
            assert left.habit_noise == right.habit_noise
            assert left.fatigue_noise == right.fatigue_noise
            assert left.policy_draw == right.policy_draw
            assert left.habit_observation_noise == 0.0
            assert right.habit_observation_noise != 0.0


def test_observation_noise_makes_world_nondeterministic() -> None:
    config = FatigueHabitConfig(observation_noise_std=0.1)
    assert not FatigueHabitWorld(config).is_deterministic
    assert FatigueHabitWorld(FatigueHabitConfig()).is_deterministic


def test_invalid_observability_and_noise_are_rejected() -> None:
    with pytest.raises(ValueError):
        FatigueHabitConfig(observability="omniscient")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FatigueHabitConfig(observation_noise_std=-0.1)
