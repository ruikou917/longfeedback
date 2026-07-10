"""Tests for World B: hidden intent dynamics and logging policies."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from longfeedback.credit.oracle import exact_deterministic_credit
from longfeedback.worlds import (
    HiddenIntentConfig,
    HiddenIntentObservation,
    HiddenIntentWorld,
    IntentAction,
    PrivilegedIntentPolicy,
    RepeatSuccessPolicy,
)


def test_deterministic_step_matches_hand_computation() -> None:
    world = HiddenIntentWorld(HiddenIntentConfig(deterministic=True, horizon=8))
    state = world.initial_state()
    exogenous = world.sample_exogenous(0)

    # Initial intent is 0; FOCUS_A matches, sigmoid(1.2) > 0.5 -> response 1.
    matched = world.step(state, IntentAction.FOCUS_A, exogenous[0])
    assert matched.info_value("response") == 1.0
    assert matched.next_state.progress == 1.0
    assert matched.next_state.last_action == 0

    # FOCUS_B mismatches intent 0; sigmoid(-1.8) < 0.5 -> response 0.
    mismatched = world.step(state, IntentAction.FOCUS_B, exogenous[0])
    assert mismatched.info_value("response") == 0.0
    assert mismatched.next_state.progress == 0.0

    # The deterministic intent cycle shifts every 4 steps and is exogenous:
    # it does not depend on the chosen action.
    assert matched.next_state.intent == mismatched.next_state.intent == 0


def test_deterministic_intent_cycle_is_action_independent() -> None:
    world = HiddenIntentWorld(HiddenIntentConfig(deterministic=True, horizon=12))
    exogenous = world.sample_exogenous(1)
    for actions in (
        [IntentAction.FOCUS_A] * 12,
        [IntentAction.FOCUS_C] * 12,
    ):
        episode = world.rollout(actions, exogenous)
        intents = [transition.state.intent for transition in episode.transitions]
        assert intents == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_terminal_proxy_and_utility_are_distinct() -> None:
    world = HiddenIntentWorld(
        HiddenIntentConfig(deterministic=True, horizon=8, proxy_threshold=3.5)
    )
    exogenous = world.sample_exogenous(0)
    # Intents: 0,0,0,0,1,1,1,1 -> matching actions produce 8 progress.
    actions = [IntentAction.FOCUS_A] * 4 + [IntentAction.FOCUS_B] * 4
    episode = world.rollout(actions, exogenous)
    assert episode.terminal_utility == 8.0
    assert episode.terminal_proxy == 1.0

    weak = world.rollout([IntentAction.FOCUS_C] * 8, exogenous)
    assert weak.terminal_utility == 0.0
    assert weak.terminal_proxy == 0.0


def test_deterministic_matched_action_credit_is_one_progress_unit() -> None:
    world = HiddenIntentWorld(HiddenIntentConfig(deterministic=True, horizon=8))
    episode = world.rollout([IntentAction.FOCUS_A] * 8, world.sample_exogenous(3))

    # Step 5 has intent 1; replacing the mismatched FOCUS_A with FOCUS_B under
    # frozen continuation adds exactly one unit of matched progress.
    pair = exact_deterministic_credit(
        world,
        episode,
        step_index=5,
        action=IntentAction.FOCUS_B,
        reference_action=IntentAction.FOCUS_A,
    )
    assert pair.credit_utility == 1.0
    assert pair.paired_noise_reused


def test_stochastic_rollout_is_seed_deterministic_and_immutable() -> None:
    world = HiddenIntentWorld()
    policy = RepeatSuccessPolicy(epsilon=0.3)
    first = world.rollout_policy(policy, world.sample_exogenous(5))
    second = world.rollout_policy(policy, world.sample_exogenous(5))
    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.transitions[0].reward = 1.0  # type: ignore[misc]


def test_shock_only_pollutes_the_proxy_never_utility() -> None:
    world = HiddenIntentWorld(HiddenIntentConfig(horizon=6, proxy_threshold=2.5))
    for seed in range(20):
        episode = world.rollout_policy(RepeatSuccessPolicy(0.3), world.sample_exogenous(seed))
        assert episode.terminal_utility == sum(
            transition.info_value("response") for transition in episode.transitions
        )


@pytest.mark.parametrize(
    "policy",
    [RepeatSuccessPolicy(epsilon=0.4), PrivilegedIntentPolicy(epsilon=0.2)],
)
def test_policy_probabilities_are_normalized_and_match_log_probability(
    policy: RepeatSuccessPolicy | PrivilegedIntentPolicy,
) -> None:
    observation = HiddenIntentObservation(
        step_index=4,
        last_action=1,
        last_response=1.0,
        cumulative_progress=3.0,
        privileged_signal=2,
    )
    probabilities = policy.probabilities(observation)
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1.0e-12)
    assert all(probability > 0.0 for probability in probabilities)
    if isinstance(policy, RepeatSuccessPolicy):
        log_probability = policy.log_probability(observation, IntentAction.FOCUS_B)
    else:
        log_probability = policy.log_probability_full(observation, IntentAction.FOCUS_B)
    assert math.isclose(log_probability, math.log(probabilities[1]), abs_tol=1.0e-12)


def test_privileged_policy_diverges_from_released_history_policy() -> None:
    """The confounded policy conditions on the hidden signal, the clean one cannot."""

    base = {
        "step_index": 4,
        "last_action": 0,
        "last_response": 0.0,
        "cumulative_progress": 1.0,
    }
    observation_signal_a = HiddenIntentObservation(privileged_signal=0, **base)
    observation_signal_c = HiddenIntentObservation(privileged_signal=2, **base)

    clean = RepeatSuccessPolicy(epsilon=0.2)
    privileged = PrivilegedIntentPolicy(epsilon=0.2)
    # Same released history, different hidden signal: the clean policy is
    # invariant, the privileged policy is not.
    assert clean.probabilities(observation_signal_a) == clean.probabilities(observation_signal_c)
    assert privileged.probabilities(observation_signal_a) != privileged.probabilities(
        observation_signal_c
    )
