"""Property tests for structural-world invariants across Worlds A and B."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from longfeedback.credit.oracle import counterfactual_pair, estimate_oracle_credit
from longfeedback.experiments.features import (
    world_a_observation_features,
    world_b_observation_features,
)
from longfeedback.worlds import (
    FatigueAction,
    FatigueHabitConfig,
    FatigueHabitWorld,
    HiddenIntentConfig,
    HiddenIntentObservation,
    HiddenIntentWorld,
    IntentAction,
    PrivilegedIntentPolicy,
    RepeatSuccessPolicy,
)

_B_ACTIONS = tuple(IntentAction)


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    step_index=st.integers(min_value=0, max_value=7),
    action=st.sampled_from(_B_ACTIONS),
)
@settings(max_examples=40, deadline=None)
def test_world_b_intervention_changes_only_the_intervened_action(
    seed: int, step_index: int, action: IntentAction
) -> None:
    world = HiddenIntentWorld(HiddenIntentConfig(horizon=8))
    episode = world.rollout_policy(RepeatSuccessPolicy(0.4), world.sample_exogenous(seed))
    pair = counterfactual_pair(
        world,
        episode,
        step_index=step_index,
        action=action,
        reference_action=IntentAction.FOCUS_A,
    )
    assert pair.paired_noise_reused
    # Both arms replay the recorded actions after the intervention point, and
    # the exogenous intent path is identical because noise is shared.
    assert (
        pair.treated_episode.actions[step_index + 1 :]
        == (pair.reference_episode.actions[step_index + 1 :])
    )
    treated_intents = [t.state.intent for t in pair.treated_episode.transitions]
    reference_intents = [t.state.intent for t in pair.reference_episode.transitions]
    assert treated_intents == reference_intents
    # Intervening with the reference action itself must give exactly zero.
    null_pair = counterfactual_pair(
        world,
        episode,
        step_index=step_index,
        action=IntentAction.FOCUS_A,
        reference_action=IntentAction.FOCUS_A,
    )
    assert null_pair.credit_utility == 0.0
    assert null_pair.credit_proxy == 0.0


@given(
    step_index=st.integers(min_value=0, max_value=30),
    last_action=st.one_of(st.none(), st.integers(min_value=0, max_value=2)),
    last_response=st.sampled_from([0.0, 1.0]),
    progress=st.floats(min_value=0.0, max_value=30.0, allow_nan=False),
    signal=st.integers(min_value=0, max_value=2),
    epsilon=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
)
@settings(max_examples=60, deadline=None)
def test_world_b_policy_probabilities_sum_to_one(
    step_index: int,
    last_action: int | None,
    last_response: float,
    progress: float,
    signal: int,
    epsilon: float,
) -> None:
    observation = HiddenIntentObservation(
        step_index=step_index,
        last_action=last_action,
        last_response=last_response,
        cumulative_progress=progress,
        privileged_signal=signal,
    )
    for policy in (RepeatSuccessPolicy(epsilon), PrivilegedIntentPolicy(epsilon)):
        probabilities = policy.probabilities(observation)
        assert abs(sum(probabilities) - 1.0) < 1.0e-9
        assert all(probability >= 0.0 for probability in probabilities)


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=25, deadline=None)
def test_partial_observation_features_never_reveal_hidden_state(seed: int) -> None:
    world = FatigueHabitWorld(FatigueHabitConfig.stochastic(observability="partial", horizon=6))
    exogenous = world.sample_exogenous(seed)
    state = world.initial_state()
    for step_index in range(world.horizon):
        observation = world.observe(state)
        features = world_a_observation_features(observation, horizon=world.horizon)
        # Released partial features: normalized step, noisy habit, last response.
        assert len(features) == 3
        assert observation.fatigue is None and observation.habituation is None
        state = world.step(state, FatigueAction.HELPFUL, exogenous[step_index]).next_state


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=25, deadline=None)
def test_world_b_released_features_never_contain_the_privileged_signal(seed: int) -> None:
    world = HiddenIntentWorld(HiddenIntentConfig(horizon=6))
    episode = world.rollout_policy(PrivilegedIntentPolicy(0.2), world.sample_exogenous(seed))
    for transition in episode.transitions:
        observation = transition.observation
        features = world_b_observation_features(observation, horizon=6, n_actions=3)
        signal_free = HiddenIntentObservation(
            step_index=observation.step_index,
            last_action=observation.last_action,
            last_response=observation.last_response,
            cumulative_progress=observation.cumulative_progress,
            privileged_signal=(observation.privileged_signal + 1) % 3,
        )
        # Flipping the hidden signal must not change any released feature.
        assert features == world_b_observation_features(signal_free, horizon=6, n_actions=3)


class RepeatSuccessPolicyForWorldA:
    """Minimal helper policy for the SE-shrinkage check."""

    def select_action(self, observation, *, step_index: int, random_value: float):
        del observation, step_index
        return FatigueAction.HELPFUL if random_value < 0.7 else FatigueAction.NOOP


def test_stochastic_oracle_se_decreases_and_estimates_agree() -> None:
    world = FatigueHabitWorld(FatigueHabitConfig.stochastic(horizon=6))
    episode = world.rollout_policy(RepeatSuccessPolicyForWorldA(), world.sample_exogenous(11))
    small = estimate_oracle_credit(
        world,
        episode,
        step_index=1,
        action=FatigueAction.URGENT,
        reference_action=FatigueAction.NOOP,
        num_rollouts=8,
        base_seed=101,
    )
    large = estimate_oracle_credit(
        world,
        episode,
        step_index=1,
        action=FatigueAction.URGENT,
        reference_action=FatigueAction.NOOP,
        num_rollouts=128,
        base_seed=101,
    )
    assert small.monte_carlo_se > 0.0
    assert large.monte_carlo_se < small.monte_carlo_se
    assert abs(large.credit_utility - small.credit_utility) < 5.0 * small.monte_carlo_se
