"""Leakage-safe feature construction for experiment models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from longfeedback.worlds import (
    DelayedConversionObservation,
    DelayedConversionWorld,
    FatigueHabitObservation,
    FatigueHabitWorld,
    HiddenIntentObservation,
    HiddenIntentWorld,
    ProxyUtilityObservation,
    ProxyUtilityWorld,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def world_a_observation_features(
    observation: FatigueHabitObservation, *, horizon: int
) -> list[float]:
    """Released World A features; hidden fields are absent, not zero-filled."""

    features = [
        observation.step_index / max(horizon - 1, 1),
        observation.habit,
        observation.last_response,
    ]
    if observation.fatigue is not None:
        features.append(observation.fatigue)
    if observation.habituation is not None:
        features.extend(observation.habituation)
    return features


def world_b_observation_features(
    observation: HiddenIntentObservation, *, horizon: int, n_actions: int
) -> list[float]:
    """Released World B features; the privileged signal must never appear."""

    last_action_onehot = [0.0] * n_actions
    if observation.last_action is not None:
        last_action_onehot[observation.last_action] = 1.0
    return [
        observation.step_index / max(horizon - 1, 1),
        1.0 if observation.last_action is not None else 0.0,
        *last_action_onehot,
        observation.last_response,
        observation.cumulative_progress / horizon,
    ]


def world_c_observation_features(
    observation: DelayedConversionObservation, *, horizon: int, n_actions: int
) -> list[float]:
    """Released World C features; pending impulses must never appear."""

    last_action_onehot = [0.0] * n_actions
    if observation.last_action is not None:
        last_action_onehot[observation.last_action] = 1.0
    return [
        observation.step_index / max(horizon - 1, 1),
        1.0 if observation.converted else 0.0,
        observation.sends / horizon,
        1.0 if observation.last_action is not None else 0.0,
        *last_action_onehot,
        min(observation.steps_since_send, horizon) / horizon,
    ]


def world_d_observation_features(
    observation: ProxyUtilityObservation, *, horizon: int, n_actions: int
) -> list[float]:
    """Released World D features; trust and dependency must never appear."""

    last_action_onehot = [0.0] * n_actions
    if observation.last_action is not None:
        last_action_onehot[observation.last_action] = 1.0
    return [
        observation.step_index / max(horizon - 1, 1),
        observation.engagement,
        observation.noisy_progress / horizon,
        1.0 if observation.last_action is not None else 0.0,
        *last_action_onehot,
    ]


def observation_features_for(world: object, observation: object, *, horizon: int) -> list[float]:
    """Dispatch the released-feature builder for any structural world."""

    if isinstance(world, FatigueHabitWorld):
        assert isinstance(observation, FatigueHabitObservation)
        return world_a_observation_features(observation, horizon=horizon)
    n_actions = len(world.action_space)  # type: ignore[attr-defined]
    if isinstance(world, HiddenIntentWorld):
        assert isinstance(observation, HiddenIntentObservation)
        return world_b_observation_features(observation, horizon=horizon, n_actions=n_actions)
    if isinstance(world, DelayedConversionWorld):
        assert isinstance(observation, DelayedConversionObservation)
        return world_c_observation_features(observation, horizon=horizon, n_actions=n_actions)
    if isinstance(world, ProxyUtilityWorld):
        assert isinstance(observation, ProxyUtilityObservation)
        return world_d_observation_features(observation, horizon=horizon, n_actions=n_actions)
    raise TypeError(f"no released-feature builder for world type {type(world).__name__}")


def action_sequence_features(
    action_sequences: Sequence[Sequence[int]],
    *,
    horizon: int,
    n_actions: int,
) -> FloatArray:
    """Encode complete action sequences as position-aware one-hot features."""

    features = np.zeros((len(action_sequences), horizon * n_actions), dtype=np.float64)
    for row, actions in enumerate(action_sequences):
        if len(actions) != horizon:
            raise ValueError(f"expected horizon {horizon}, got {len(actions)}")
        for step, action in enumerate(actions):
            if action < 0 or action >= n_actions:
                raise ValueError(f"action {action} is outside [0, {n_actions})")
            features[row, step * n_actions + action] = 1.0
    return features


def prefix_action_features(
    actions: Sequence[int],
    *,
    horizon: int,
    n_actions: int,
) -> FloatArray:
    """Encode every prefix without exposing future actions.

    Row ``t`` contains actions strictly before prefix boundary ``t``. Future
    slots remain all-zero, which makes accidental future leakage testable.
    """

    if len(actions) != horizon:
        raise ValueError(f"expected horizon {horizon}, got {len(actions)}")
    encoded = np.zeros((horizon + 1, horizon * n_actions), dtype=np.float64)
    for step, action in enumerate(actions):
        if action < 0 or action >= n_actions:
            raise ValueError(f"action {action} is outside [0, {n_actions})")
        encoded[step + 1] = encoded[step]
        encoded[step + 1, step * n_actions + action] = 1.0
    return encoded


def oracle_credit_features(
    steps: Sequence[int],
    actions: Sequence[int],
    *,
    horizon: int,
    n_actions: int,
) -> FloatArray:
    """Encode step/action pairs for an oracle-supervised linear diagnostic."""

    if len(steps) != len(actions):
        raise ValueError("steps and actions must have equal length")
    features = np.zeros((len(steps), horizon * n_actions), dtype=np.float64)
    for row, (step, action) in enumerate(zip(steps, actions, strict=True)):
        if step < 0 or step >= horizon:
            raise ValueError(f"step {step} is outside [0, {horizon})")
        if action < 0 or action >= n_actions:
            raise ValueError(f"action {action} is outside [0, {n_actions})")
        features[row, step * n_actions + action] = 1.0
    return features


def deterministic_split(
    size: int,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[IntArray, IntArray]:
    """Return stable, non-overlapping train and test row indices."""

    if size < 2:
        raise ValueError("at least two samples are required")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    order = np.random.default_rng(seed).permutation(size).astype(np.int64)
    boundary = min(max(int(size * train_fraction), 1), size - 1)
    return order[:boundary], order[boundary:]
