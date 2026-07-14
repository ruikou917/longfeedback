"""Synchronous actor/critic online iteration for E12 (design section 11.2).

The actor and critic are frozen in turn: trajectories and branch targets are
collected under a frozen policy checkpoint, the critic is fitted only on
labels whose continuation policy is fresh enough, then the frozen critic
scores advantages for one shared group-policy update. The optimization shell
is identical for every method; only the credit differs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from longfeedback.actors.trainable import TrainableSoftmaxCandidatePolicy
from longfeedback.budget import BudgetLedger, BudgetLimits
from longfeedback.credit.branching import (
    BranchGenerationSettings,
    BranchSelectionRule,
    BranchTargetRow,
    CandidateRule,
    EpisodeRecord,
    StatePolicyDistribution,
    collect_episode,
    derive_seed,
    full_distribution,
    generate_branch_targets,
)
from longfeedback.credit.c3 import c3_estimates, c3_logged_action_advantage
from longfeedback.credit.gigpo import GigpoSettings, anchor_diagnostics, gigpo_step_credits
from longfeedback.environments.base import EnvironmentClient, GameRef
from longfeedback.models.candidate_data import build_candidate_dataset
from longfeedback.models.candidate_docm import (
    CandidateDelayedOutcomeCreditModel,
    CandidateTrainingSettings,
    candidate_variant_spec,
)
from longfeedback.models.encoders import EncoderArchitecture
from longfeedback.models.text_embeddings import TextEmbeddingProvider
from longfeedback.training.group_policy import (
    GroupPolicySettings,
    PolicyUpdateStep,
    center_advantages,
    group_policy_update,
)

E12_METHODS = (
    "frozen_actor",
    "terminal_grpo",
    "prefix_group",
    "c3_group",
    "gigpo",
    "longfeedback_group",
)
_BRANCH_METHODS = ("c3_group", "longfeedback_group")


@dataclass(frozen=True, slots=True)
class OnlineLoopSettings:
    iterations: int
    base_episodes_per_iteration: int
    iteration_token_budget: int
    max_extra_episodes: int
    max_steps: int
    branch_selection: BranchSelectionRule
    candidate_rule: CandidateRule
    branch_rollouts_per_action: int
    unforced_rollouts: int
    max_policy_lag: int = 0
    evaluation_interval: int = 1
    gigpo: GigpoSettings = field(default_factory=GigpoSettings)
    group_policy: GroupPolicySettings = field(default_factory=GroupPolicySettings)
    critic_training: CandidateTrainingSettings = field(default_factory=CandidateTrainingSettings)
    critic_architecture: EncoderArchitecture = field(default_factory=EncoderArchitecture)
    action_mlp_hidden: int = 64
    policy_center_tolerance: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.base_episodes_per_iteration <= 0:
            raise ValueError("iterations and base episodes must be positive")
        if self.iteration_token_budget <= 0 or self.max_extra_episodes < 0:
            raise ValueError("token budget must be positive")
        if self.max_policy_lag != 0:
            raise ValueError("only max_policy_lag=0 is supported in the primary loop")
        if self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive")


@dataclass
class MethodOutcome:
    method: str
    seed: int
    success_curve: list[dict[str, float]]
    locked_success: float
    ledger: dict[str, Any]
    policy_ids: list[str]
    diagnostics: dict[str, Any]
    budget_exceeded: bool
    final_policy_state: dict[str, Any]


def filter_fresh_rows(
    rows: Sequence[BranchTargetRow], allowed_policy_ids: Sequence[str]
) -> tuple[list[BranchTargetRow], int]:
    """Exclude labels from stale continuation policies (default lag 0)."""

    allowed = set(allowed_policy_ids)
    fresh = [row for row in rows if row.continuation_policy_id in allowed]
    return fresh, len(rows) - len(fresh)


def greedy_success_rate(
    client: EnvironmentClient,
    policy: TrainableSoftmaxCandidatePolicy,
    games: Sequence[GameRef],
    *,
    max_steps: int,
    seed_tag: str,
) -> float:
    """Deterministic argmax evaluation; not charged to the training budget."""

    ledger = BudgetLedger(
        BudgetLimits(
            max_actor_forward_tokens=10**12,
            max_environment_steps=10**12,
            max_wall_time_seconds=10**9,
        )
    )
    successes = 0
    for game in games:
        episode = collect_episode(
            client,
            game,
            _GreedyPolicyView(policy),
            episode_id=f"eval:{seed_tag}:{game.game_id}",
            environment_seed=derive_seed("eval-env", seed_tag, game.game_id) % 2**31,
            max_steps=max_steps,
            ledger=ledger,
        )
        successes += int(episode.success)
    return successes / len(games) if games else 0.0


class _GreedyPolicyView:
    """Wraps a policy so sampling always returns the argmax candidate."""

    def __init__(self, policy: TrainableSoftmaxCandidatePolicy) -> None:
        self._policy = policy

    @property
    def policy_id(self) -> str:
        return f"greedy:{self._policy.policy_id}"

    def score(self, prompt: str, candidates: Sequence[str]) -> Any:
        return self._policy.score(prompt, candidates)

    def sample(self, scores: Any, *, random_value: float) -> Any:
        del random_value
        best = max(range(len(scores.probabilities)), key=lambda i: scores.probabilities[i])
        from longfeedback.actors.base import PolicyDecision

        return PolicyDecision(
            action=scores.candidates[best],
            index=best,
            probability=scores.probabilities[best],
            log_probability=scores.log_probabilities[best],
            entropy=scores.entropy,
            random_value=0.0,
        )


def _probe_rows(
    episodes: Sequence[EpisodeRecord],
) -> tuple[list[BranchTargetRow], list[StatePolicyDistribution]]:
    """Unlabeled rows for every logged step so the critic can score Q/V."""

    rows: list[BranchTargetRow] = []
    distributions: dict[str, StatePolicyDistribution] = {}
    for episode in episodes:
        for step in episode.steps:
            distribution = full_distribution(step)
            distributions[distribution.state_hash] = distribution
            rows.append(
                BranchTargetRow(
                    game_id=episode.game.game_id,
                    split=episode.game.split,
                    episode_id=episode.episode_id,
                    step_index=step.step_index,
                    state_hash=step.observation.state_hash,
                    replay_prefix_hash=step.replay_handle.prefix_hash,
                    candidate_action=step.decision.action,
                    candidate_set_id="probe",
                    candidate_selected_probability=1.0,
                    candidate_policy_probability=step.decision.probability,
                    full_policy_distribution_hash=(distribution.full_policy_distribution_hash),
                    continuation_policy_id=episode.actor_policy_id,
                    rollout_seed_block_hash="probe",
                    rollout_count=0,
                    success_count=0,
                    q_hat=0.0,
                    q_se=0.0,
                    unforced_rollout_count=0,
                    unforced_success_count=0,
                    v_hat=0.0,
                    v_se=0.0,
                    forced_next_state_hash="",
                    forced_next_observation="",
                    forced_done=False,
                    forced_terminal_success=0.0,
                    child_unforced_rollout_count=0,
                    child_unforced_success_count=0,
                    child_v_hat=0.0,
                    child_v_se=0.0,
                    advantage_hat=0.0,
                    target_role="train",
                    selection_probability=1.0,
                    stratum="probe",
                    logged_action=True,
                    prompt_hash=step.prompt_hash,
                )
            )
    # One row per step; sorted to match build_candidate_dataset's branch order.
    rows.sort(key=lambda row: (row.episode_id, row.step_index))
    return rows, list(distributions.values())


def _step_key(episode_id: str, step_index: int) -> tuple[str, int]:
    return (episode_id, step_index)


def _collect_iteration_episodes(
    client: EnvironmentClient,
    policy: TrainableSoftmaxCandidatePolicy,
    games: Sequence[GameRef],
    *,
    method: str,
    seed: int,
    iteration: int,
    count: int,
    start_index: int = 0,
    max_steps: int,
    ledger: BudgetLedger,
) -> list[EpisodeRecord]:
    episodes: list[EpisodeRecord] = []
    for offset in range(count):
        index = start_index + offset
        game = games[index % len(games)]
        episode_id = f"{method}:s{seed}:i{iteration}:e{index}"
        episodes.append(
            collect_episode(
                client,
                game,
                policy,
                episode_id=episode_id,
                environment_seed=derive_seed("e12-env", episode_id) % 2**31,
                max_steps=max_steps,
                ledger=ledger,
            )
        )
    return episodes


def run_online_method(
    method: str,
    client: EnvironmentClient,
    embedder: TextEmbeddingProvider,
    settings: OnlineLoopSettings,
    *,
    seed: int,
    initial_policy_state: dict[str, Any],
    train_games: Sequence[GameRef],
    valid_seen_games: Sequence[GameRef],
    valid_unseen_games: Sequence[GameRef],
) -> MethodOutcome:
    """Run one E12 method under the shared budget and optimization shell."""

    if method not in E12_METHODS:
        raise ValueError(f"unknown E12 method {method!r}")
    policy = TrainableSoftmaxCandidatePolicy(embedder, seed=seed)
    policy.load_state_dict({name: tensor.clone() for name, tensor in initial_policy_state.items()})
    reference = TrainableSoftmaxCandidatePolicy(embedder, seed=seed)
    reference.load_state_dict(
        {name: tensor.clone() for name, tensor in initial_policy_state.items()}
    )

    ledger = BudgetLedger(
        BudgetLimits(
            max_actor_forward_tokens=settings.iteration_token_budget * settings.iterations * 4,
            max_environment_steps=10**9,
            max_wall_time_seconds=10**7,
        )
    )
    success_curve: list[dict[str, float]] = []
    policy_ids: list[str] = [policy.policy_id]
    diagnostics: dict[str, Any] = {
        "stale_rows_excluded": 0,
        "centering_aborts": 0,
        "update_stats": [],
    }
    budget_exceeded = False

    for iteration in range(settings.iterations):
        iteration_start_tokens = ledger.actor_forward_tokens
        frozen_policy_id = policy.policy_id
        if method != "frozen_actor":
            episodes = _collect_iteration_episodes(
                client,
                policy,
                train_games,
                method=method,
                seed=seed,
                iteration=iteration,
                count=settings.base_episodes_per_iteration,
                max_steps=settings.max_steps,
                ledger=ledger,
            )

            rows: list[BranchTargetRow] = []
            distributions: list[StatePolicyDistribution] = []
            if method in _BRANCH_METHODS:
                generation = BranchGenerationSettings(
                    selection_rule=settings.branch_selection,
                    candidate_rule=settings.candidate_rule,
                    forced_rollouts=settings.branch_rollouts_per_action,
                    unforced_rollouts=settings.unforced_rollouts,
                    child_value_fraction=0.25 if method == "longfeedback_group" else 0.0,
                    child_rollouts=(
                        settings.unforced_rollouts if method == "longfeedback_group" else 1
                    ),
                    max_continuation_steps=settings.max_steps,
                    target_role="train",
                )
                rows, distributions = generate_branch_targets(
                    client,
                    policy,
                    episodes,
                    generation,
                    seed=f"e12:{method}:s{seed}:i{iteration}",
                    ledger=ledger,
                )
                rows, stale = filter_fresh_rows(rows, [frozen_policy_id])
                diagnostics["stale_rows_excluded"] += stale
            else:
                # Compute-fair terminal/grouped baselines may spend the same
                # token budget on additional complete trajectories.
                extra = 0
                while (
                    ledger.actor_forward_tokens - iteration_start_tokens
                    < settings.iteration_token_budget
                    and extra < settings.max_extra_episodes
                ):
                    episodes.extend(
                        _collect_iteration_episodes(
                            client,
                            policy,
                            train_games,
                            method=method,
                            seed=seed,
                            iteration=iteration,
                            count=1,
                            start_index=settings.base_episodes_per_iteration + extra,
                            max_steps=settings.max_steps,
                            ledger=ledger,
                        )
                    )
                    extra += 1

            iteration_tokens = ledger.actor_forward_tokens - iteration_start_tokens
            if iteration_tokens > settings.iteration_token_budget:
                budget_exceeded = True

            advantages = _method_advantages(
                method,
                episodes,
                rows,
                distributions,
                embedder,
                settings,
                seed=seed,
                diagnostics=diagnostics,
            )
            if advantages is not None:
                update_steps = _build_update_steps(episodes, advantages)
                stats = group_policy_update(policy, reference, update_steps, settings.group_policy)
                stats["iteration"] = float(iteration)
                diagnostics["update_stats"].append(stats)
        policy_ids.append(policy.policy_id)

        if (iteration + 1) % settings.evaluation_interval == 0:
            success_curve.append(
                {
                    "iteration": float(iteration),
                    "valid_seen_success": greedy_success_rate(
                        client,
                        policy,
                        valid_seen_games,
                        max_steps=settings.max_steps,
                        seed_tag=f"{method}:s{seed}:i{iteration}",
                    ),
                    "actor_forward_tokens": float(ledger.actor_forward_tokens),
                }
            )

    locked_success = greedy_success_rate(
        client,
        policy,
        valid_unseen_games,
        max_steps=settings.max_steps,
        seed_tag=f"{method}:s{seed}:locked",
    )
    return MethodOutcome(
        method=method,
        seed=seed,
        success_curve=success_curve,
        locked_success=locked_success,
        ledger=ledger.as_dict(),
        policy_ids=policy_ids,
        diagnostics=diagnostics,
        budget_exceeded=budget_exceeded,
        final_policy_state=dict(policy.state_dict()),
    )


def _build_update_steps(
    episodes: Sequence[EpisodeRecord], advantages: dict[tuple[str, int], float]
) -> list[PolicyUpdateStep]:
    keyed: list[tuple[tuple[str, int], PolicyUpdateStep]] = []
    values: list[float] = []
    for episode in episodes:
        for step in episode.steps:
            key = _step_key(episode.episode_id, step.step_index)
            if key not in advantages:
                continue
            values.append(advantages[key])
            keyed.append(
                (
                    key,
                    PolicyUpdateStep(
                        prompt=step.prompt_text,
                        candidates=step.scores.candidates,
                        chosen_index=step.decision.index,
                        old_log_probability=step.decision.log_probability,
                        advantage=0.0,
                    ),
                )
            )
    centered = center_advantages(values)
    return [
        PolicyUpdateStep(
            prompt=item.prompt,
            candidates=item.candidates,
            chosen_index=item.chosen_index,
            old_log_probability=item.old_log_probability,
            advantage=value,
        )
        for (_, item), value in zip(keyed, centered, strict=True)
    ]


def _method_advantages(
    method: str,
    episodes: Sequence[EpisodeRecord],
    rows: Sequence[BranchTargetRow],
    distributions: Sequence[StatePolicyDistribution],
    embedder: TextEmbeddingProvider,
    settings: OnlineLoopSettings,
    *,
    seed: int,
    diagnostics: dict[str, Any],
) -> dict[tuple[str, int], float] | None:
    """Per-step advantages; None aborts this iteration's update."""

    if method == "terminal_grpo":
        by_game: dict[str, list[EpisodeRecord]] = defaultdict(list)
        for episode in episodes:
            by_game[episode.game.game_id].append(episode)
        advantages: dict[tuple[str, int], float] = {}
        for game_episodes in by_game.values():
            values = center_advantages([float(e.success) for e in game_episodes])
            for episode, value in zip(game_episodes, values, strict=True):
                for step in episode.steps:
                    advantages[_step_key(episode.episode_id, step.step_index)] = value
        return advantages

    if method == "gigpo":
        credits = gigpo_step_credits(episodes, settings.gigpo)
        diagnostics["gigpo_anchor"] = anchor_diagnostics(credits)
        return {
            _step_key(credit.episode_id, credit.step_index): credit.combined_advantage
            for credit in credits
        }

    if method == "c3_group":
        advantages = {}
        logged_by_state = {
            (row.episode_id, row.step_index): row.candidate_action
            for row in rows
            if row.logged_action
        }
        for estimate in c3_estimates(list(rows)):
            key = (estimate.episode_id, estimate.step_index)
            logged = logged_by_state.get(key)
            if logged is None:
                continue
            advantage = c3_logged_action_advantage(estimate, logged)
            if advantage is not None:
                advantages[key] = advantage
        return advantages

    if method in ("prefix_group", "longfeedback_group"):
        variant = "docm_prefix" if method == "prefix_group" else "docm_dueling_credit"
        weights, parameterization = candidate_variant_spec(variant)
        training_rows = list(rows) if method == "longfeedback_group" else []
        training_distributions = list(distributions) if training_rows else []
        dataset = build_candidate_dataset(
            episodes,
            training_rows,
            training_distributions,
            embedder,
            horizon=settings.max_steps,
        )
        critic = CandidateDelayedOutcomeCreditModel(
            state_dim=embedder.dimension,
            action_dim=embedder.dimension,
            max_horizon=settings.max_steps,
            architecture=settings.critic_architecture,
            loss_weights=weights,
            action_value_parameterization=parameterization,
            action_mlp_hidden=settings.action_mlp_hidden,
            seed=seed,
        )
        critic.fit(dataset, training=settings.critic_training)

        probe_row_list, probe_distributions = _probe_rows(episodes)
        probe_dataset = build_candidate_dataset(
            episodes,
            probe_row_list,
            probe_distributions,
            embedder,
            horizon=settings.max_steps,
        )
        baseline_values = critic.predict_branch_value(probe_dataset)
        advantages = {}
        if method == "prefix_group":
            success_by_episode = {
                episode.episode_id: float(episode.success) for episode in episodes
            }
            for index, row in enumerate(probe_row_list):
                advantages[_step_key(row.episode_id, row.step_index)] = success_by_episode[
                    row.episode_id
                ] - float(baseline_values[index])
            return advantages

        residual = critic.policy_centering_residual(probe_dataset)
        diagnostics["policy_centering_residual"] = residual
        if residual > settings.policy_center_tolerance:
            diagnostics["centering_aborts"] += 1
            return None
        q = critic.predict_branch_q(probe_dataset)
        distribution_by_state = {d.state_hash: d for d in probe_distributions}
        for index, row in enumerate(probe_row_list):
            actions = distribution_by_state[row.state_hash].actions
            slot = actions.index(row.candidate_action)
            advantages[_step_key(row.episode_id, row.step_index)] = float(
                q[index, slot] - baseline_values[index]
            )
        return advantages

    raise ValueError(f"method {method!r} does not produce advantages")
