"""Paired structural counterfactual evaluation with Monte Carlo SE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import sqrt
from statistics import fmean, stdev
from typing import Generic, TypeVar

from longfeedback.worlds.base import Episode, Policy, StructuralWorld, Transition

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
ObservationT = TypeVar("ObservationT")
NoiseT = TypeVar("NoiseT")


class ContinuationMode(StrEnum):
    FROZEN = "frozen"
    POLICY_REACTIVE = "policy_reactive"


@dataclass(frozen=True, slots=True)
class CounterfactualPair(Generic[StateT, ActionT, ObservationT, NoiseT]):
    step_index: int
    action: ActionT
    reference_action: ActionT
    continuation_mode: ContinuationMode
    treated_episode: Episode[StateT, ActionT, ObservationT, NoiseT]
    reference_episode: Episode[StateT, ActionT, ObservationT, NoiseT]
    credit_utility: float
    credit_proxy: float

    @property
    def paired_noise_reused(self) -> bool:
        treated = self.treated_episode.transitions
        reference = self.reference_episode.transitions
        return len(treated) == len(reference) and all(
            left.exogenous is right.exogenous
            for left, right in zip(treated, reference, strict=True)
        )


@dataclass(frozen=True, slots=True)
class OracleCreditEstimate(Generic[ActionT]):
    step_index: int
    action: ActionT
    reference_action: ActionT
    continuation_mode: ContinuationMode
    credit_utility: float
    credit_proxy: float
    monte_carlo_se: float
    proxy_monte_carlo_se: float
    num_rollouts: int
    utility_samples: tuple[float, ...]
    proxy_samples: tuple[float, ...]


def _standard_error(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values) / sqrt(len(values))


def _episode_noise_by_step(
    episode: Episode[StateT, ActionT, ObservationT, NoiseT],
) -> dict[int, NoiseT]:
    return {transition.step_index: transition.exogenous for transition in episode.transitions}


def _sequence_noise_by_step(
    world: StructuralWorld[StateT, ActionT, ObservationT, NoiseT],
    exogenous: Sequence[NoiseT],
    *,
    start_step: int,
) -> dict[int, NoiseT]:
    if len(exogenous) < world.horizon:
        raise ValueError("sampled exogenous sequence must cover the full horizon")
    return {step: exogenous[step] for step in range(start_step, world.horizon)}


def _rollout_arm(
    world: StructuralWorld[StateT, ActionT, ObservationT, NoiseT],
    *,
    initial_state: StateT,
    step_index: int,
    intervention_action: ActionT,
    continuation_mode: ContinuationMode,
    original_actions: Mapping[int, ActionT],
    noise_by_step: Mapping[int, NoiseT],
    future_policy: Policy[ObservationT, ActionT] | None,
    seed: int | None,
) -> Episode[StateT, ActionT, ObservationT, NoiseT]:
    state = initial_state
    initial_observation = world.observe(state)
    transitions: list[Transition[StateT, ActionT, ObservationT, NoiseT]] = []

    for current_step in range(step_index, world.horizon):
        try:
            noise = noise_by_step[current_step]
        except KeyError as error:
            raise ValueError(f"missing exogenous noise for step {current_step}") from error

        if current_step == step_index:
            selected_action = intervention_action
        elif continuation_mode is ContinuationMode.FROZEN:
            try:
                selected_action = original_actions[current_step]
            except KeyError as error:
                raise ValueError(
                    "baseline episode does not contain the complete frozen continuation"
                ) from error
        else:
            if future_policy is None:
                raise ValueError("future_policy is required for policy-reactive continuation")
            selected_action = future_policy.select_action(
                world.observe(state),
                step_index=current_step,
                random_value=world.policy_random_value(noise),
            )

        transition = world.step(state, selected_action, noise)
        if transition.step_index != current_step:
            raise RuntimeError("world returned a transition for the wrong step")
        transitions.append(transition)
        state = transition.next_state

    if not transitions or not transitions[-1].terminated:
        raise RuntimeError("counterfactual rollout did not reach a terminal state")

    return Episode(
        initial_state=initial_state,
        initial_observation=initial_observation,
        transitions=tuple(transitions),
        terminal_proxy=world.terminal_proxy(state),
        terminal_utility=world.terminal_utility(state),
        seed=seed,
    )


def counterfactual_pair(
    world: StructuralWorld[StateT, ActionT, ObservationT, NoiseT],
    episode: Episode[StateT, ActionT, ObservationT, NoiseT],
    *,
    step_index: int,
    action: ActionT,
    reference_action: ActionT,
    continuation_mode: ContinuationMode | str = ContinuationMode.FROZEN,
    future_policy: Policy[ObservationT, ActionT] | None = None,
    exogenous: Sequence[NoiseT] | None = None,
) -> CounterfactualPair[StateT, ActionT, ObservationT, NoiseT]:
    """Evaluate one paired intervention from the observed pre-action state."""

    mode = ContinuationMode(continuation_mode)
    if not 0 <= step_index < world.horizon:
        raise IndexError("step_index is outside the world horizon")
    if mode is ContinuationMode.POLICY_REACTIVE and future_policy is None:
        raise ValueError("future_policy is required for policy-reactive continuation")

    initial_state = episode.state_before(step_index)
    original_actions = {
        transition.step_index: transition.action for transition in episode.transitions
    }
    if exogenous is None:
        noise_by_step = _episode_noise_by_step(episode)
        seed = episode.seed
    else:
        noise_by_step = _sequence_noise_by_step(world, exogenous, start_step=step_index)
        seed = getattr(exogenous, "seed", None)

    treated_episode = _rollout_arm(
        world,
        initial_state=initial_state,
        step_index=step_index,
        intervention_action=action,
        continuation_mode=mode,
        original_actions=original_actions,
        noise_by_step=noise_by_step,
        future_policy=future_policy,
        seed=seed,
    )
    reference_episode = _rollout_arm(
        world,
        initial_state=initial_state,
        step_index=step_index,
        intervention_action=reference_action,
        continuation_mode=mode,
        original_actions=original_actions,
        noise_by_step=noise_by_step,
        future_policy=future_policy,
        seed=seed,
    )
    return CounterfactualPair(
        step_index=step_index,
        action=action,
        reference_action=reference_action,
        continuation_mode=mode,
        treated_episode=treated_episode,
        reference_episode=reference_episode,
        credit_utility=treated_episode.terminal_utility - reference_episode.terminal_utility,
        credit_proxy=treated_episode.terminal_proxy - reference_episode.terminal_proxy,
    )


def exact_deterministic_credit(
    world: StructuralWorld[StateT, ActionT, ObservationT, NoiseT],
    episode: Episode[StateT, ActionT, ObservationT, NoiseT],
    *,
    step_index: int,
    action: ActionT,
    reference_action: ActionT,
    continuation_mode: ContinuationMode | str = ContinuationMode.FROZEN,
    future_policy: Policy[ObservationT, ActionT] | None = None,
) -> CounterfactualPair[StateT, ActionT, ObservationT, NoiseT]:
    """Return exact credit for a deterministic world configuration."""

    if not world.is_deterministic:
        raise ValueError("exact deterministic credit requires zero world noise")
    return counterfactual_pair(
        world,
        episode,
        step_index=step_index,
        action=action,
        reference_action=reference_action,
        continuation_mode=continuation_mode,
        future_policy=future_policy,
    )


def estimate_oracle_credit(
    world: StructuralWorld[StateT, ActionT, ObservationT, NoiseT],
    episode: Episode[StateT, ActionT, ObservationT, NoiseT],
    *,
    step_index: int,
    action: ActionT,
    reference_action: ActionT,
    continuation_mode: ContinuationMode | str = ContinuationMode.FROZEN,
    future_policy: Policy[ObservationT, ActionT] | None = None,
    num_rollouts: int = 128,
    base_seed: int = 0,
) -> OracleCreditEstimate[ActionT]:
    """Estimate conditional credit with paired common random numbers."""

    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive")
    mode = ContinuationMode(continuation_mode)
    utility_samples = []
    proxy_samples = []

    for offset in range(num_rollouts):
        exogenous = world.sample_exogenous(base_seed + offset)
        pair = counterfactual_pair(
            world,
            episode,
            step_index=step_index,
            action=action,
            reference_action=reference_action,
            continuation_mode=mode,
            future_policy=future_policy,
            exogenous=exogenous,
        )
        utility_samples.append(pair.credit_utility)
        proxy_samples.append(pair.credit_proxy)

    return OracleCreditEstimate(
        step_index=step_index,
        action=action,
        reference_action=reference_action,
        continuation_mode=mode,
        credit_utility=fmean(utility_samples),
        credit_proxy=fmean(proxy_samples),
        monte_carlo_se=_standard_error(utility_samples),
        proxy_monte_carlo_se=_standard_error(proxy_samples),
        num_rollouts=num_rollouts,
        utility_samples=tuple(utility_samples),
        proxy_samples=tuple(proxy_samples),
    )
