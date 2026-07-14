"""Hand-computed fixtures for the direct C3 and GiGPO credit baselines."""

from __future__ import annotations

from dataclasses import replace

import pytest

from longfeedback.actors.mock import MockCandidatePolicy
from longfeedback.budget import BudgetLedger, BudgetLimits
from longfeedback.credit.branching import (
    BranchTargetRow,
    collect_episode,
    derive_seed,
)
from longfeedback.credit.c3 import c3_estimates, c3_logged_action_advantage
from longfeedback.credit.gigpo import (
    GigpoSettings,
    anchor_diagnostics,
    gigpo_step_credits,
)
from longfeedback.environments.fake import FakeTextEnvironment, FakeWorldSettings


def _row(action: str, q_hat: float, probability: float, *, logged: bool) -> BranchTargetRow:
    return BranchTargetRow(
        game_id="g",
        split="train",
        episode_id="e",
        step_index=1,
        state_hash="s",
        replay_prefix_hash="p",
        candidate_action=action,
        candidate_set_id="c",
        candidate_selected_probability=1.0,
        candidate_policy_probability=probability,
        full_policy_distribution_hash="d",
        continuation_policy_id="pi",
        rollout_seed_block_hash="r",
        rollout_count=4,
        success_count=int(q_hat * 4),
        q_hat=q_hat,
        q_se=0.1,
        unforced_rollout_count=4,
        unforced_success_count=2,
        v_hat=0.5,
        v_se=0.1,
        forced_next_state_hash="n",
        forced_next_observation="obs",
        forced_done=False,
        forced_terminal_success=0.0,
        child_unforced_rollout_count=0,
        child_unforced_success_count=0,
        child_v_hat=0.0,
        child_v_se=0.0,
        advantage_hat=q_hat - 0.5,
        target_role="validation",
        selection_probability=1.0,
        stratum="early",
        logged_action=logged,
        prompt_hash="h",
    )


def test_c3_policy_centered_contrast_hand_computed() -> None:
    # Probabilities renormalize to (0.5, 0.25, 0.25); center = 0.55.
    rows = [
        _row("a", 0.8, 0.4, logged=True),
        _row("b", 0.4, 0.2, logged=False),
        _row("c", 0.2, 0.2, logged=False),
    ]
    estimates = c3_estimates(rows)
    assert len(estimates) == 1
    estimate = estimates[0]
    center = 0.5 * 0.8 + 0.25 * 0.4 + 0.25 * 0.2
    for _action, advantage, q_hat in zip(
        estimate.actions, estimate.advantages, estimate.q_hats, strict=True
    ):
        assert advantage == pytest.approx(q_hat - center)
    # Leave-one-out for action a: others renormalize to (0.5, 0.5).
    a_index = estimate.actions.index("a")
    assert estimate.leave_one_out_advantages[a_index] == pytest.approx(0.8 - 0.3)
    assert c3_logged_action_advantage(estimate, "a") == pytest.approx(0.8 - center)
    assert c3_logged_action_advantage(estimate, "missing") is None
    assert estimate.rollouts_used == 12


def test_gigpo_macro_and_micro_credit() -> None:
    env = FakeTextEnvironment(FakeWorldSettings(code_length=2, max_steps=6))
    game = env.list_games("train")[0]
    actor = MockCandidatePolicy(
        seed=3, verb_bias={"press": 1.5, "pull": 1.5, "smash": -2.0, "yank": -2.0}
    )
    ledger = BudgetLedger(
        BudgetLimits(
            max_actor_forward_tokens=10**9,
            max_environment_steps=10**9,
            max_wall_time_seconds=600.0,
        )
    )
    episodes = [
        collect_episode(
            env,
            game,
            actor,
            episode_id=f"e{i}",
            environment_seed=derive_seed("gg", str(i)) % 2**31,
            max_steps=6,
            ledger=ledger,
        )
        for i in range(4)
    ]
    credits = gigpo_step_credits(episodes, GigpoSettings(micro_weight=0.5))
    assert len(credits) == sum(len(episode.steps) for episode in episodes)
    successes = [float(episode.success) for episode in episodes]
    if len(set(successes)) == 1:
        # Declared all-equal convention: zero macro advantage everywhere.
        assert all(credit.macro_advantage == 0.0 for credit in credits)
    for credit in credits:
        if credit.anchor_group_size < 2:
            assert credit.micro_advantage == 0.0
        assert credit.combined_advantage == pytest.approx(
            credit.macro_advantage + 0.5 * credit.micro_advantage
        )
    diagnostics = anchor_diagnostics(credits)
    assert 0.0 <= diagnostics["anchor_availability"] <= 1.0
    # All four episodes start at the same initial state, so the first step
    # of each episode shares an anchor group.
    first_steps = [credit for credit in credits if credit.step_index == 0]
    assert all(credit.anchor_group_size >= 2 for credit in first_steps)


def test_gigpo_settings_validation() -> None:
    with pytest.raises(ValueError):
        GigpoSettings(discount=0.0)
    with pytest.raises(ValueError):
        GigpoSettings(micro_weight=-1.0)


def test_c3_skips_states_with_zero_probability_mass() -> None:
    rows = [replace(_row("a", 0.5, 0.0, logged=True), candidate_policy_probability=0.0)]
    assert c3_estimates(rows) == []
