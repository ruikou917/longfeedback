"""Tests for World C (delayed conversion) and World D (proxy-utility divergence)."""

from __future__ import annotations

import math
import statistics

import pytest

from longfeedback.credit.oracle import counterfactual_pair, exact_deterministic_credit
from longfeedback.worlds import (
    ConversionAction,
    DelayedConversionConfig,
    DelayedConversionWorld,
    InfluenceAction,
    MixedInfluencePolicy,
    ProxyUtilityConfig,
    ProxyUtilityWorld,
    SpacedOutreachPolicy,
)


class TestDelayedConversion:
    def test_impulse_arrives_after_delay_and_decays(self) -> None:
        world = DelayedConversionWorld(DelayedConversionConfig(deterministic=True))
        state = world.initial_state()
        exogenous = world.sample_exogenous(0)
        # STRONG has delay_geometric_p=0.25 -> deterministic delay 4.
        state = world.step(state, ConversionAction.STRONG, exogenous[0]).next_state
        quality = world.config.impulse_qualities[2]
        assert state.impulses == (
            world.step(
                world.initial_state(), ConversionAction.STRONG, exogenous[0]
            ).next_state.impulses
        )
        assert state.impulses[0].arrival_step == 4
        assert state.impulses[0].quality == quality

        for step_index in range(1, 6):
            hazard = world.hazard(state)
            if step_index <= 4:
                # Impulse not yet arrived at steps 1-4 observation points.
                expected = world.config.base_hazard + (
                    quality * world.config.kernel_decay ** (state.step_index - 4)
                    if state.step_index >= 4
                    else 0.0
                )
                assert math.isclose(hazard, expected, abs_tol=1.0e-12)
            state = world.step(state, ConversionAction.NONE, exogenous[step_index]).next_state

    def test_saturation_reduces_later_impulse_quality(self) -> None:
        world = DelayedConversionWorld(DelayedConversionConfig(deterministic=True))
        state = world.initial_state()
        exogenous = world.sample_exogenous(0)
        state = world.step(state, ConversionAction.STRONG, exogenous[0]).next_state
        state = world.step(state, ConversionAction.STRONG, exogenous[1]).next_state
        first, second = state.impulses
        assert second.quality == pytest.approx(
            first.quality * math.exp(-world.config.saturation_rate)
        )

    def test_late_action_cannot_convert_and_costs_utility(self) -> None:
        world = DelayedConversionWorld(DelayedConversionConfig(deterministic=True))
        actions = [ConversionAction.NONE] * (world.horizon - 1) + [ConversionAction.STRONG]
        episode = world.rollout(actions, world.sample_exogenous(0))
        assert episode.terminal_proxy == 0.0
        assert episode.terminal_utility == pytest.approx(-world.config.action_cost)

    def test_deterministic_frozen_credit_matches_hand_value(self) -> None:
        world = DelayedConversionWorld(DelayedConversionConfig(deterministic=True))
        actions = [ConversionAction.STRONG] + [ConversionAction.NONE] * (world.horizon - 1)
        episode = world.rollout(actions, world.sample_exogenous(0))
        assert episode.terminal_proxy == 1.0
        pair = exact_deterministic_credit(
            world,
            episode,
            step_index=0,
            action=ConversionAction.NONE,
            reference_action=ConversionAction.STRONG,
        )
        # Removing the only converting send loses the conversion value but
        # refunds one action cost.
        assert pair.credit_utility == pytest.approx(
            -(world.config.conversion_value - world.config.action_cost)
        )

    def test_observation_hides_pending_impulses(self) -> None:
        world = DelayedConversionWorld()
        state = world.initial_state()
        state = world.step(state, ConversionAction.STRONG, world.sample_exogenous(3)[0]).next_state
        observation = world.observe(state)
        assert not hasattr(observation, "impulses")
        assert observation.sends == 1

    def test_paired_counterfactual_reuses_noise(self) -> None:
        world = DelayedConversionWorld()
        episode = world.rollout_policy(SpacedOutreachPolicy(0.2), world.sample_exogenous(9))
        pair = counterfactual_pair(
            world,
            episode,
            step_index=2,
            action=ConversionAction.SOFT,
            reference_action=ConversionAction.NONE,
        )
        assert pair.paired_noise_reused

    def test_policy_probabilities_normalized(self) -> None:
        policy = SpacedOutreachPolicy(0.3)
        world = DelayedConversionWorld()
        observation = world.observe(world.initial_state())
        assert math.isclose(sum(policy.probabilities(observation)), 1.0, abs_tol=1.0e-12)


class TestProxyUtility:
    def test_goodhart_gap_fear_raises_proxy_and_lowers_utility(self) -> None:
        """Without this property the overoptimization experiments are void."""

        world = ProxyUtilityWorld()

        def mean_values(action: InfluenceAction) -> tuple[float, float]:
            episodes = [
                world.rollout([action] * world.horizon, world.sample_exogenous(seed))
                for seed in range(200)
            ]
            return (
                statistics.mean(episode.terminal_proxy for episode in episodes),
                statistics.mean(episode.terminal_utility for episode in episodes),
            )

        helpful_proxy, helpful_utility = mean_values(InfluenceAction.HELPFUL)
        fear_proxy, fear_utility = mean_values(InfluenceAction.FEAR)
        assert fear_proxy > helpful_proxy + 0.15
        assert fear_utility < helpful_utility - 5.0

    def test_trust_and_dependency_are_hidden(self) -> None:
        world = ProxyUtilityWorld()
        observation = world.observe(world.initial_state())
        assert not hasattr(observation, "trust")
        assert not hasattr(observation, "dependency")

    def test_deterministic_return_thresholds_on_engagement(self) -> None:
        world = ProxyUtilityWorld(ProxyUtilityConfig(deterministic=True))
        high = world.rollout([InfluenceAction.FEAR] * world.horizon, world.sample_exogenous(0))
        low = world.rollout([InfluenceAction.NOOP] * world.horizon, world.sample_exogenous(0))
        assert high.terminal_proxy == 1.0
        assert low.terminal_proxy == 0.0

    def test_deterministic_credit_fear_versus_helpful_is_negative_utility(self) -> None:
        world = ProxyUtilityWorld(ProxyUtilityConfig(deterministic=True))
        episode = world.rollout(
            [InfluenceAction.HELPFUL] * world.horizon, world.sample_exogenous(0)
        )
        pair = exact_deterministic_credit(
            world,
            episode,
            step_index=4,
            action=InfluenceAction.FEAR,
            reference_action=InfluenceAction.HELPFUL,
        )
        assert pair.credit_utility < 0.0

    def test_interruption_costs_accumulate(self) -> None:
        world = ProxyUtilityWorld(ProxyUtilityConfig(deterministic=True))
        episode = world.rollout([InfluenceAction.URGENT] * world.horizon, world.sample_exogenous(0))
        expected = world.config.interruption_costs[2] * world.horizon
        assert episode.transitions[-1].next_state.cumulative_interruption == pytest.approx(expected)

    def test_policy_probabilities_normalized(self) -> None:
        world = ProxyUtilityWorld()
        policy = MixedInfluencePolicy(0.25)
        observation = world.observe(world.initial_state())
        assert math.isclose(sum(policy.probabilities(observation)), 1.0, abs_tol=1.0e-12)
