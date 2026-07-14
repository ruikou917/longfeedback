"""Direct fixed-history branching credit (C3-style baseline).

C3 is a non-amortized estimator: it forms policy-centered contrasts directly
from forced-action Monte Carlo estimates at labeled branch states and makes no
predictions at unbranched states. The probabilities are renormalized over the
labeled candidate subset; that scope restriction is reported, not hidden.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from longfeedback.credit.branching import BranchTargetRow


@dataclass(frozen=True, slots=True)
class C3StateEstimate:
    state_hash: str
    episode_id: str
    step_index: int
    actions: tuple[str, ...]
    q_hats: tuple[float, ...]
    policy_probabilities: tuple[float, ...]
    advantages: tuple[float, ...]
    leave_one_out_advantages: tuple[float, ...]
    rollouts_used: int


def c3_estimates(rows: Sequence[BranchTargetRow]) -> list[C3StateEstimate]:
    """Group rows by state and compute policy-centered direct contrasts."""

    by_state: dict[tuple[str, int, str], list[BranchTargetRow]] = defaultdict(list)
    for row in rows:
        by_state[(row.episode_id, row.step_index, row.state_hash)].append(row)

    estimates: list[C3StateEstimate] = []
    for (episode_id, step_index, state_hash), state_rows in sorted(by_state.items()):
        ordered = sorted(state_rows, key=lambda row: row.candidate_action)
        total_probability = sum(row.candidate_policy_probability for row in ordered)
        if total_probability <= 0.0:
            continue
        weights = [row.candidate_policy_probability / total_probability for row in ordered]
        q_hats = [row.q_hat for row in ordered]
        center = sum(weight * q for weight, q in zip(weights, q_hats, strict=True))
        advantages = [q - center for q in q_hats]
        leave_one_out: list[float] = []
        for index in range(len(ordered)):
            remaining = 1.0 - weights[index]
            if remaining <= 0.0:
                leave_one_out.append(0.0)
                continue
            other = sum(weights[j] * q_hats[j] for j in range(len(weights)) if j != index)
            leave_one_out.append(q_hats[index] - other / remaining)
        estimates.append(
            C3StateEstimate(
                state_hash=state_hash,
                episode_id=episode_id,
                step_index=step_index,
                actions=tuple(row.candidate_action for row in ordered),
                q_hats=tuple(q_hats),
                policy_probabilities=tuple(weights),
                advantages=tuple(advantages),
                leave_one_out_advantages=tuple(leave_one_out),
                rollouts_used=sum(row.rollout_count for row in ordered),
            )
        )
    return estimates


def c3_logged_action_advantage(estimate: C3StateEstimate, logged_action: str) -> float | None:
    """Advantage of the logged action, or None when it lacks direct samples.

    Rows without the logged action are excluded under a rule frozen before
    evaluation; their unused budget is reported, never reassigned.
    """

    for action, advantage in zip(estimate.actions, estimate.advantages, strict=True):
        if action == logged_action:
            return advantage
    return None
