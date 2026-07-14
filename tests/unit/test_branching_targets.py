"""Branch-state selection, candidate selection, and Monte Carlo targets."""

from __future__ import annotations

import math

import pytest

from longfeedback.actors.mock import MockCandidatePolicy
from longfeedback.budget import BudgetLedger, BudgetLimits
from longfeedback.credit.branching import (
    BranchGenerationSettings,
    BranchSelectionRule,
    CandidateRule,
    binomial_se,
    collect_episode,
    derive_seed,
    derive_unit,
    generate_branch_targets,
    replay_audit,
    select_branch_states,
    select_candidates,
)
from longfeedback.credit.tree_targets import TreeTargetKind, tree_target_for_row
from longfeedback.environments.fake import FakeTextEnvironment, FakeWorldSettings

_ACTOR = MockCandidatePolicy(
    seed=7,
    verb_bias={"press": 0.9, "pull": 0.9, "smash": -0.9, "yank": -0.9},
    noise_scale=0.4,
    prompt_noise_scale=0.2,
)


def _ledger() -> BudgetLedger:
    return BudgetLedger(
        BudgetLimits(
            max_actor_forward_tokens=10**9,
            max_environment_steps=10**9,
            max_wall_time_seconds=3600.0,
        )
    )


def _episodes(count: int = 6) -> list:
    env = FakeTextEnvironment(FakeWorldSettings())
    games = env.list_games("train")
    ledger = _ledger()
    episodes = []
    for index in range(count):
        episode_id = f"t:{index}"
        episodes.append(
            collect_episode(
                env,
                games[index % len(games)],
                _ACTOR,
                episode_id=episode_id,
                environment_seed=derive_seed("env", episode_id) % 2**31,
                max_steps=8,
                ledger=ledger,
            )
        )
    return episodes


def test_derive_streams_are_deterministic_and_independent() -> None:
    assert derive_unit("a", "b") == derive_unit("a", "b")
    assert derive_unit("a", "b") != derive_unit("b", "a")
    assert 0.0 <= derive_unit("x") < 1.0


def test_binomial_se_matches_formula() -> None:
    assert binomial_se(2, 8) == pytest.approx(math.sqrt(0.25 * 0.75 / 8))
    assert binomial_se(0, 0) == 0.0


def test_branch_state_selection_is_stratified_and_deterministic() -> None:
    episodes = _episodes()
    rule = BranchSelectionRule(states_per_episode=3)
    for episode in episodes:
        first = select_branch_states(episode, rule, seed="s")
        second = select_branch_states(episode, rule, seed="s")
        assert first == second
        strata = [selection.stratum for selection in first]
        assert len(strata) == len(set(strata))
        for selection in first:
            assert 0.0 < selection.selection_probability <= 1.0
            step = episode.steps[selection.step_index]
            assert len(step.observation.admissible_actions) >= 2


def test_candidate_selection_contains_logged_action_and_dedupes() -> None:
    episodes = _episodes(2)
    step = episodes[0].steps[0]
    rule = CandidateRule(full_enumeration_limit=2, top_actor_candidates=1, random_candidates=1)
    candidate_set = select_candidates(step, rule, seed="c")
    assert step.decision.action in candidate_set.actions
    assert list(candidate_set.actions) == sorted(set(candidate_set.actions))
    assert candidate_set == select_candidates(step, rule, seed="c")


def test_generate_branch_targets_counts_and_determinism() -> None:
    env = FakeTextEnvironment(FakeWorldSettings())
    episodes = _episodes(4)
    settings = BranchGenerationSettings(
        selection_rule=BranchSelectionRule(states_per_episode=1),
        candidate_rule=CandidateRule(
            full_enumeration_limit=4, top_actor_candidates=1, random_candidates=1
        ),
        forced_rollouts=3,
        unforced_rollouts=4,
        child_value_fraction=1.0,
        child_rollouts=2,
        max_continuation_steps=8,
        target_role="train",
    )
    rows, distributions = generate_branch_targets(
        env, _ACTOR, episodes, settings, seed="g", ledger=_ledger()
    )
    rows_again, _ = generate_branch_targets(
        env, _ACTOR, episodes, settings, seed="g", ledger=_ledger()
    )
    assert rows == rows_again
    assert rows, "expected at least one branch target row"
    state_hashes = {distribution.state_hash for distribution in distributions}
    for row in rows:
        assert row.state_hash in state_hashes
        assert row.rollout_count == 3 and 0 <= row.success_count <= 3
        assert row.q_hat == pytest.approx(row.success_count / row.rollout_count)
        assert row.v_hat == pytest.approx(row.unforced_success_count / 4)
        assert row.advantage_hat == pytest.approx(row.q_hat - row.v_hat)
        if row.forced_done:
            assert row.child_unforced_rollout_count == 0
        else:
            assert row.child_unforced_rollout_count == 2
        assert row.continuation_policy_id == _ACTOR.policy_id


def test_rollout_seed_split_changes_rollouts_not_states() -> None:
    env = FakeTextEnvironment(FakeWorldSettings())
    episodes = _episodes(4)
    settings = BranchGenerationSettings(
        selection_rule=BranchSelectionRule(states_per_episode=1),
        candidate_rule=CandidateRule(
            full_enumeration_limit=4, top_actor_candidates=1, random_candidates=1
        ),
        forced_rollouts=2,
        unforced_rollouts=2,
        child_value_fraction=0.0,
        child_rollouts=1,
        max_continuation_steps=8,
        target_role="validation",
    )
    base, _ = generate_branch_targets(env, _ACTOR, episodes, settings, seed="g", ledger=_ledger())
    other, _ = generate_branch_targets(
        env, _ACTOR, episodes, settings, seed="g", rollout_seed="r", ledger=_ledger()
    )
    assert [(r.state_hash, r.candidate_action) for r in base] == [
        (r.state_hash, r.candidate_action) for r in other
    ]
    assert [r.rollout_seed_block_hash for r in base] != [r.rollout_seed_block_hash for r in other]


def test_replay_audit_full_match_on_deterministic_environment() -> None:
    env = FakeTextEnvironment(FakeWorldSettings())
    episodes = _episodes(3)
    audit = replay_audit(env, episodes, max_prefixes=10)
    assert audit["replay_match_rate"] == 1.0
    assert audit["audited_prefixes"] <= 10


def test_tree_target_kinds() -> None:
    episodes = _episodes(4)
    env = FakeTextEnvironment(FakeWorldSettings())
    settings = BranchGenerationSettings(
        selection_rule=BranchSelectionRule(states_per_episode=2),
        candidate_rule=CandidateRule(
            full_enumeration_limit=4, top_actor_candidates=1, random_candidates=1
        ),
        forced_rollouts=2,
        unforced_rollouts=2,
        child_value_fraction=0.5,
        child_rollouts=2,
        max_continuation_steps=8,
        target_role="train",
    )
    rows, _ = generate_branch_targets(env, _ACTOR, episodes, settings, seed="g", ledger=_ledger())
    kinds = set()
    for row in rows:
        target = tree_target_for_row(row)
        kinds.add(target.kind)
        if row.forced_done:
            assert target.kind is TreeTargetKind.TERMINAL
            assert target.value == row.forced_terminal_success
        elif row.child_unforced_rollout_count > 0:
            assert target.kind is TreeTargetKind.CHILD_DIRECT
            assert target.value == pytest.approx(row.child_v_hat)
        else:
            assert target.kind is TreeTargetKind.BOOTSTRAP
    assert TreeTargetKind.BOOTSTRAP in kinds or TreeTargetKind.CHILD_DIRECT in kinds


def test_tree_target_rejects_locked_reference_rows() -> None:
    episodes = _episodes(2)
    env = FakeTextEnvironment(FakeWorldSettings())
    settings = BranchGenerationSettings(
        selection_rule=BranchSelectionRule(states_per_episode=1),
        candidate_rule=CandidateRule(
            full_enumeration_limit=4, top_actor_candidates=1, random_candidates=1
        ),
        forced_rollouts=2,
        unforced_rollouts=2,
        child_value_fraction=0.0,
        child_rollouts=1,
        max_continuation_steps=8,
        target_role="locked_reference",
    )
    rows, _ = generate_branch_targets(env, _ACTOR, episodes, settings, seed="g", ledger=_ledger())
    with pytest.raises(ValueError):
        tree_target_for_row(rows[0])
