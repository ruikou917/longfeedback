"""Replay-verified trajectory collection and Monte Carlo Q/V branch targets.

All randomness is derived from independent SHA-256 streams so that adding a
branch never changes the base trajectory, and the same rollout seed block is
shared across candidate actions (paired common random numbers).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from longfeedback.actors.base import (
    CandidatePolicy,
    PolicyDecision,
    PolicyScores,
    canonical_candidates,
    render_prompt,
)
from longfeedback.budget import BudgetLedger
from longfeedback.environments.base import (
    EnvironmentClient,
    EnvObservation,
    GameRef,
    ReplayHandle,
    ReplayMismatchError,
    normalize_action,
)

_SUCCESS_EPSILON = 1.0e-9


def derive_unit(*parts: str) -> float:
    """Deterministic pseudo-uniform in [0, 1) from an independent named stream."""

    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def derive_seed(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def observation_success(observation: EnvObservation) -> bool:
    return observation.done and observation.score >= 1.0 - _SUCCESS_EPSILON


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_index: int
    observation: EnvObservation
    prompt_text: str
    prompt_hash: str
    scores: PolicyScores
    decision: PolicyDecision
    replay_handle: ReplayHandle
    next_observation: EnvObservation


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode_id: str
    game: GameRef
    environment_seed: int
    actor_policy_id: str
    steps: tuple[StepRecord, ...]
    success: bool
    terminal_reason: str
    terminal_score: float
    actor_forward_tokens: int

    @property
    def action_sequence(self) -> tuple[str, ...]:
        return tuple(step.decision.action for step in self.steps)


def collect_episode(
    client: EnvironmentClient,
    game: GameRef,
    actor: CandidatePolicy,
    *,
    episode_id: str,
    environment_seed: int,
    max_steps: int,
    ledger: BudgetLedger,
) -> EpisodeRecord:
    """Roll one base trajectory with the frozen actor and full provenance."""

    observation = client.reset(game, seed=environment_seed)
    history: list[tuple[str, str]] = []
    prefix: list[str] = []
    steps: list[StepRecord] = []
    forward_tokens = 0
    while not observation.done and len(steps) < max_steps:
        prompt = render_prompt(
            goal=observation.goal,
            history=history,
            observation=observation.observation,
            admissible_actions=observation.admissible_actions,
        )
        scores = actor.score(prompt.text, observation.admissible_actions)
        forward_tokens += scores.forward_tokens
        ledger.add_actor_tokens(scores.forward_tokens)
        random_value = derive_unit("actor-sample", episode_id, str(len(steps)))
        decision = actor.sample(scores, random_value=random_value)
        handle = ReplayHandle(
            game_id=game.game_id,
            split=game.split,
            environment_seed=environment_seed,
            action_prefix=tuple(prefix),
            expected_state_hash=observation.state_hash,
        )
        transition = client.step(decision.action)
        ledger.add_environment_steps(1)
        steps.append(
            StepRecord(
                step_index=len(steps),
                observation=observation,
                prompt_text=prompt.text,
                prompt_hash=prompt.prompt_hash,
                scores=scores,
                decision=decision,
                replay_handle=handle,
                next_observation=transition.observation,
            )
        )
        history.append((observation.observation, decision.action))
        prefix.append(decision.action)
        observation = transition.observation

    success = observation_success(observation)
    if success:
        terminal_reason = "success"
    elif observation.done:
        terminal_reason = "environment_termination"
    else:
        terminal_reason = "timeout"
    return EpisodeRecord(
        episode_id=episode_id,
        game=game,
        environment_seed=environment_seed,
        actor_policy_id=actor.policy_id,
        steps=tuple(steps),
        success=success,
        terminal_reason=terminal_reason,
        terminal_score=observation.score,
        actor_forward_tokens=forward_tokens,
    )


@dataclass(frozen=True, slots=True)
class BranchSelectionRule:
    """Outcome-blind stratified branch-state selection."""

    states_per_episode: int = 2
    uniform_weight: float = 0.5
    min_admissible: int = 2

    def __post_init__(self) -> None:
        if self.states_per_episode <= 0:
            raise ValueError("states_per_episode must be positive")
        if not 0.0 <= self.uniform_weight <= 1.0:
            raise ValueError("uniform_weight must be in [0, 1]")
        if self.min_admissible < 2:
            raise ValueError("branch states need at least two admissible actions")


@dataclass(frozen=True, slots=True)
class SelectedBranchState:
    step_index: int
    stratum: str
    selection_probability: float


def select_branch_states(
    episode: EpisodeRecord, rule: BranchSelectionRule, *, seed: str
) -> tuple[SelectedBranchState, ...]:
    """Pick at most one eligible state per normalized-time stratum.

    The rule never inspects branch outcomes: weights mix a uniform component
    with normalized actor entropy recorded at collection time.
    """

    # Every StepRecord is a decision state (non-terminal by construction), so
    # eligibility only filters on the admissible-action count.
    eligible = [
        step for step in episode.steps if len(step.scores.candidates) >= rule.min_admissible
    ]
    if not eligible:
        return ()
    strata_names = ("early", "middle", "late")
    total = len(eligible)
    strata: dict[str, list[StepRecord]] = {name: [] for name in strata_names}
    for position, step in enumerate(eligible):
        stratum = strata_names[min(2, position * 3 // total)]
        strata[stratum].append(step)
    nonempty = [name for name in strata_names if strata[name]]
    chosen_strata = nonempty
    if rule.states_per_episode < len(nonempty):
        order = sorted(
            nonempty,
            key=lambda name: derive_unit("stratum-choice", seed, episode.episode_id, name),
        )
        chosen_strata = sorted(order[: rule.states_per_episode], key=strata_names.index)

    selections: list[SelectedBranchState] = []
    for name in chosen_strata:
        members = strata[name]
        max_entropy = max((math.log(len(step.scores.candidates)) for step in members), default=1.0)
        weights = [
            rule.uniform_weight / len(members)
            + (1.0 - rule.uniform_weight)
            * (step.scores.entropy / max_entropy if max_entropy > 0 else 1.0 / len(members))
            for step in members
        ]
        total_weight = sum(weights)
        probabilities = [weight / total_weight for weight in weights]
        draw = derive_unit("state-choice", seed, episode.episode_id, name)
        cumulative = 0.0
        index = len(members) - 1
        for position, probability in enumerate(probabilities):
            cumulative += probability
            if draw < cumulative:
                index = position
                break
        selections.append(
            SelectedBranchState(
                step_index=members[index].step_index,
                stratum=name,
                selection_probability=probabilities[index],
            )
        )
    return tuple(selections)


@dataclass(frozen=True, slots=True)
class CandidateRule:
    full_enumeration_limit: int = 8
    top_actor_candidates: int = 2
    random_candidates: int = 1


@dataclass(frozen=True, slots=True)
class CandidateSet:
    actions: tuple[str, ...]
    candidate_set_id: str
    selection_probabilities: tuple[float, ...]


def select_candidates(step: StepRecord, rule: CandidateRule, *, seed: str) -> CandidateSet:
    """Deterministic candidate set: logged + top actor + seeded random extras."""

    admissible = canonical_candidates(step.observation.admissible_actions)
    logged = normalize_action(step.decision.action)
    if len(admissible) <= rule.full_enumeration_limit:
        chosen = set(admissible)
        probability = 1.0
    else:
        chosen = {logged}
        by_probability = sorted(
            zip(step.scores.candidates, step.scores.probabilities, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        for candidate, _ in by_probability[: rule.top_actor_candidates]:
            chosen.add(candidate)
        remaining = [candidate for candidate in admissible if candidate not in chosen]
        ranked = sorted(
            remaining,
            key=lambda candidate: derive_unit("candidate-choice", seed, candidate),
        )
        needed = min(rule.random_candidates, len(ranked))
        for candidate in ranked[:needed]:
            chosen.add(candidate)
        probability = needed / len(remaining) if remaining else 1.0
    actions = tuple(sorted(chosen))
    candidate_set_id = hashlib.sha256(
        json.dumps([seed, list(actions)], separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    probabilities = tuple(
        1.0
        if (
            len(admissible) <= rule.full_enumeration_limit
            or action == logged
            or action
            in {
                candidate
                for candidate, _ in sorted(
                    zip(step.scores.candidates, step.scores.probabilities, strict=True),
                    key=lambda item: (-item[1], item[0]),
                )[: rule.top_actor_candidates]
            }
        )
        else probability
        for action in actions
    )
    return CandidateSet(
        actions=actions,
        candidate_set_id=candidate_set_id,
        selection_probabilities=probabilities,
    )


@dataclass(frozen=True, slots=True)
class ContinuationResult:
    success: bool
    terminal_reason: str
    length: int
    forced_next: EnvObservation | None


def _restore_with_history(
    client: EnvironmentClient, handle: ReplayHandle
) -> tuple[list[tuple[str, str]], EnvObservation]:
    """Reset-and-replay that also rebuilds the observable history for prompts."""

    observation = client.reset(
        GameRef(game_id=handle.game_id, split=handle.split), seed=handle.environment_seed
    )
    history: list[tuple[str, str]] = []
    for action in handle.action_prefix:
        if observation.done:
            raise ReplayMismatchError(
                f"episode terminated before prefix completed for {handle.game_id}"
            )
        history.append((observation.observation, action))
        observation = client.step(action).observation
    if observation.state_hash != handle.expected_state_hash:
        raise ReplayMismatchError(
            f"state hash mismatch for {handle.game_id}: expected "
            f"{handle.expected_state_hash}, got {observation.state_hash}"
        )
    return history, observation


def rollout_continuation(
    client: EnvironmentClient,
    handle: ReplayHandle,
    policy: CandidatePolicy,
    *,
    forced_action: str | None,
    seed_block: str,
    max_steps: int,
    ledger: BudgetLedger,
    kind: str,
) -> ContinuationResult:
    """Restore the prefix, optionally force one action, then follow the policy."""

    history, observation = _restore_with_history(client, handle)
    ledger.add_environment_steps(len(handle.action_prefix))
    length = 0
    forced_next: EnvObservation | None = None
    if forced_action is not None:
        history.append((observation.observation, normalize_action(forced_action)))
        observation = client.step(forced_action).observation
        ledger.add_environment_steps(1)
        length += 1
        forced_next = observation
    while not observation.done and length < max_steps:
        prompt = render_prompt(
            goal=observation.goal,
            history=history,
            observation=observation.observation,
            admissible_actions=observation.admissible_actions,
        )
        scores = policy.score(prompt.text, observation.admissible_actions)
        ledger.add_actor_tokens(scores.forward_tokens)
        decision = policy.sample(
            scores, random_value=derive_unit("continuation", seed_block, str(length))
        )
        history.append((observation.observation, decision.action))
        observation = client.step(decision.action).observation
        ledger.add_environment_steps(1)
        length += 1
    ledger.count_continuation(kind)
    success = observation_success(observation)
    if success:
        terminal_reason = "success"
    elif observation.done:
        terminal_reason = "environment_termination"
    else:
        terminal_reason = "timeout"
    return ContinuationResult(
        success=success,
        terminal_reason=terminal_reason,
        length=length,
        forced_next=forced_next,
    )


@dataclass(frozen=True, slots=True)
class BranchTargetRow:
    """One (state, candidate action) Monte Carlo target row (schema 8.3)."""

    game_id: str
    split: str
    episode_id: str
    step_index: int
    state_hash: str
    replay_prefix_hash: str
    candidate_action: str
    candidate_set_id: str
    candidate_selected_probability: float
    candidate_policy_probability: float
    full_policy_distribution_hash: str
    continuation_policy_id: str
    rollout_seed_block_hash: str
    rollout_count: int
    success_count: int
    q_hat: float
    q_se: float
    unforced_rollout_count: int
    unforced_success_count: int
    v_hat: float
    v_se: float
    forced_next_state_hash: str
    forced_next_observation: str
    forced_done: bool
    forced_terminal_success: float
    child_unforced_rollout_count: int
    child_unforced_success_count: int
    child_v_hat: float
    child_v_se: float
    advantage_hat: float
    target_role: str
    selection_probability: float
    stratum: str
    logged_action: bool
    prompt_hash: str


@dataclass(frozen=True, slots=True)
class StatePolicyDistribution:
    """Full frozen-actor distribution at one branch state (side table)."""

    state_hash: str
    full_policy_distribution_hash: str
    actions: tuple[str, ...]
    probabilities: tuple[float, ...]


def binomial_se(successes: int, count: int) -> float:
    if count <= 0:
        return 0.0
    rate = successes / count
    return math.sqrt(rate * (1.0 - rate) / count)


def full_distribution(step: StepRecord) -> StatePolicyDistribution:
    payload = json.dumps(
        {
            "actions": list(step.scores.candidates),
            "probabilities": [round(p, 12) for p in step.scores.probabilities],
        },
        separators=(",", ":"),
    )
    return StatePolicyDistribution(
        state_hash=step.observation.state_hash,
        full_policy_distribution_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        actions=step.scores.candidates,
        probabilities=step.scores.probabilities,
    )


@dataclass(frozen=True, slots=True)
class BranchGenerationSettings:
    selection_rule: BranchSelectionRule
    candidate_rule: CandidateRule
    forced_rollouts: int
    unforced_rollouts: int
    child_value_fraction: float
    child_rollouts: int
    max_continuation_steps: int
    target_role: str

    def __post_init__(self) -> None:
        if self.forced_rollouts <= 0 or self.unforced_rollouts <= 0:
            raise ValueError("rollout counts must be positive")
        if not 0.0 <= self.child_value_fraction <= 1.0:
            raise ValueError("child_value_fraction must be in [0, 1]")
        if self.target_role not in ("train", "validation", "locked_reference"):
            raise ValueError("target_role must be train, validation, or locked_reference")


def generate_branch_targets(
    client: EnvironmentClient,
    continuation_policy: CandidatePolicy,
    episodes: Sequence[EpisodeRecord],
    settings: BranchGenerationSettings,
    *,
    seed: str,
    rollout_seed: str | None = None,
    ledger: BudgetLedger,
) -> tuple[list[BranchTargetRow], list[StatePolicyDistribution]]:
    """Generate replay-verified Monte Carlo Q/V targets for selected states.

    ``seed`` drives outcome-blind state/candidate selection; ``rollout_seed``
    (defaulting to ``seed``) drives continuation randomness, so a baseline can
    re-estimate the same states with independent rollouts.
    """

    roll = rollout_seed if rollout_seed is not None else seed
    rows: list[BranchTargetRow] = []
    distributions: dict[str, StatePolicyDistribution] = {}
    for episode in episodes:
        selections = select_branch_states(episode, settings.selection_rule, seed=seed)
        for selection in selections:
            step = episode.steps[selection.step_index]
            handle = step.replay_handle
            state_key = f"{episode.episode_id}:{selection.step_index}"
            candidate_set = select_candidates(
                step, settings.candidate_rule, seed=f"{seed}:{state_key}"
            )
            distribution = full_distribution(step)
            distributions[distribution.state_hash] = distribution

            try:
                unforced_successes = 0
                for rollout in range(settings.unforced_rollouts):
                    result = rollout_continuation(
                        client,
                        handle,
                        continuation_policy,
                        forced_action=None,
                        seed_block=f"{roll}:v:{state_key}:{rollout}",
                        max_steps=settings.max_continuation_steps,
                        ledger=ledger,
                        kind=f"{settings.target_role}:unforced_v",
                    )
                    unforced_successes += int(result.success)

                v_hat = unforced_successes / settings.unforced_rollouts
                probability_by_action = dict(
                    zip(step.scores.candidates, step.scores.probabilities, strict=True)
                )
                candidate_rows: list[BranchTargetRow] = []
                for action, selected_probability in zip(
                    candidate_set.actions, candidate_set.selection_probabilities, strict=True
                ):
                    successes = 0
                    forced_next: EnvObservation | None = None
                    for rollout in range(settings.forced_rollouts):
                        # Rollout seed blocks are shared across candidate
                        # actions: paired common random numbers.
                        result = rollout_continuation(
                            client,
                            handle,
                            continuation_policy,
                            forced_action=action,
                            seed_block=f"{roll}:q:{state_key}:{rollout}",
                            max_steps=settings.max_continuation_steps,
                            ledger=ledger,
                            kind=f"{settings.target_role}:forced_q",
                        )
                        successes += int(result.success)
                        forced_next = result.forced_next
                    assert forced_next is not None
                    q_hat = successes / settings.forced_rollouts

                    child_count = 0
                    child_successes = 0
                    wants_child = (
                        not forced_next.done
                        and settings.child_rollouts > 0
                        and derive_unit("child-select", seed, state_key, action)
                        < settings.child_value_fraction
                    )
                    if wants_child:
                        child_handle = replace(
                            handle,
                            action_prefix=(*handle.action_prefix, action),
                            expected_state_hash=forced_next.state_hash,
                        )
                        for rollout in range(settings.child_rollouts):
                            result = rollout_continuation(
                                client,
                                child_handle,
                                continuation_policy,
                                forced_action=None,
                                seed_block=f"{roll}:cv:{state_key}:{action}:{rollout}",
                                max_steps=settings.max_continuation_steps,
                                ledger=ledger,
                                kind=f"{settings.target_role}:child_v",
                            )
                            child_count += 1
                            child_successes += int(result.success)

                    seed_block_hash = hashlib.sha256(f"{roll}:q:{state_key}".encode()).hexdigest()[
                        :16
                    ]
                    candidate_rows.append(
                        BranchTargetRow(
                            game_id=episode.game.game_id,
                            split=episode.game.split,
                            episode_id=episode.episode_id,
                            step_index=selection.step_index,
                            state_hash=step.observation.state_hash,
                            replay_prefix_hash=handle.prefix_hash,
                            candidate_action=action,
                            candidate_set_id=candidate_set.candidate_set_id,
                            candidate_selected_probability=selected_probability,
                            candidate_policy_probability=probability_by_action[action],
                            full_policy_distribution_hash=(
                                distribution.full_policy_distribution_hash
                            ),
                            continuation_policy_id=continuation_policy.policy_id,
                            rollout_seed_block_hash=seed_block_hash,
                            rollout_count=settings.forced_rollouts,
                            success_count=successes,
                            q_hat=q_hat,
                            q_se=binomial_se(successes, settings.forced_rollouts),
                            unforced_rollout_count=settings.unforced_rollouts,
                            unforced_success_count=unforced_successes,
                            v_hat=v_hat,
                            v_se=binomial_se(unforced_successes, settings.unforced_rollouts),
                            forced_next_state_hash=forced_next.state_hash,
                            forced_next_observation=forced_next.observation,
                            forced_done=forced_next.done,
                            forced_terminal_success=float(observation_success(forced_next)),
                            child_unforced_rollout_count=child_count,
                            child_unforced_success_count=child_successes,
                            child_v_hat=child_successes / child_count if child_count else 0.0,
                            child_v_se=binomial_se(child_successes, child_count),
                            advantage_hat=q_hat - v_hat,
                            target_role=settings.target_role,
                            selection_probability=selection.selection_probability,
                            stratum=selection.stratum,
                            logged_action=action == normalize_action(step.decision.action),
                            prompt_hash=step.prompt_hash,
                        )
                    )
                rows.extend(candidate_rows)
            except ReplayMismatchError:
                ledger.count_replay_failure()
                continue
    return rows, list(distributions.values())


def replay_audit(
    client: EnvironmentClient,
    episodes: Sequence[EpisodeRecord],
    *,
    max_prefixes: int,
) -> dict[str, float | int]:
    """Restore sampled prefixes and report the exact-match rate."""

    handles: list[ReplayHandle] = [
        step.replay_handle for episode in episodes for step in episode.steps
    ]
    handles.sort(key=lambda handle: derive_unit("replay-audit", handle.prefix_hash))
    audited = handles[: max_prefixes if max_prefixes > 0 else len(handles)]
    matches = 0
    for handle in audited:
        try:
            _restore_with_history(client, handle)
            matches += 1
        except ReplayMismatchError:
            continue
    return {
        "audited_prefixes": len(audited),
        "matched_prefixes": matches,
        "replay_match_rate": matches / len(audited) if audited else 1.0,
    }
