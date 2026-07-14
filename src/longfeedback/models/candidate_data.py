"""Variable-horizon candidate-action dataset for the candidate DOCM.

The dataset carries two granularities: padded episode sequences (terminal and
prefix supervision) and branch rows (direct V, branch Q, and tree targets).
Every branch row stores the **full admissible action set** with the frozen
actor's probabilities so policy-centered Q can center over the entire set,
while Monte Carlo labels exist only for the evaluated candidate subset
(``q_count == 0`` marks unlabeled candidates).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from longfeedback.credit.branching import (
    BranchTargetRow,
    EpisodeRecord,
    StatePolicyDistribution,
)
from longfeedback.models.text_embeddings import TextEmbeddingProvider

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class CandidateSequenceDataset:
    """Padded episodes plus branch-state candidate targets."""

    state_embeddings: FloatArray
    action_embeddings: FloatArray
    step_mask: BoolArray
    outcomes: FloatArray
    branch_episode: IntArray
    branch_step: IntArray
    candidate_embeddings: FloatArray
    candidate_mask: BoolArray
    candidate_probabilities: FloatArray
    q_success: IntArray
    q_count: IntArray
    v_success: IntArray
    v_count: IntArray
    forced_done: BoolArray
    forced_success: FloatArray
    child_v_success: IntArray
    child_v_count: IntArray
    child_state_embeddings: FloatArray

    def __post_init__(self) -> None:
        if self.state_embeddings.ndim != 3 or self.action_embeddings.ndim != 3:
            raise ValueError("episode embeddings must be [episodes, horizon, dim]")
        episodes, horizon, _ = self.state_embeddings.shape
        if self.action_embeddings.shape[:2] != (episodes, horizon):
            raise ValueError("action embeddings must align with state embeddings")
        if self.step_mask.shape != (episodes, horizon) or self.outcomes.shape != (episodes,):
            raise ValueError("step_mask/outcomes shapes are inconsistent")
        branches = self.branch_episode.shape[0]
        if self.branch_step.shape != (branches,):
            raise ValueError("branch_step must be [branches]")
        if self.candidate_embeddings.ndim != 3 or self.candidate_embeddings.shape[0] != branches:
            raise ValueError("candidate embeddings must be [branches, candidates, dim]")
        candidates = self.candidate_embeddings.shape[1]
        for name in ("candidate_mask", "candidate_probabilities", "q_success", "q_count"):
            if getattr(self, name).shape != (branches, candidates):
                raise ValueError(f"{name} must be [branches, candidates]")
        for name in ("forced_done", "forced_success", "child_v_success", "child_v_count"):
            if getattr(self, name).shape != (branches, candidates):
                raise ValueError(f"{name} must be [branches, candidates]")
        if self.v_success.shape != (branches,) or self.v_count.shape != (branches,):
            raise ValueError("v_success and v_count must be [branches]")
        if self.child_state_embeddings.shape[:2] != (branches, candidates):
            raise ValueError("child_state_embeddings must be [branches, candidates, dim]")
        if branches:
            masked_probabilities = np.where(self.candidate_mask, self.candidate_probabilities, 0.0)
            totals = masked_probabilities.sum(axis=1)
            if not np.allclose(totals, 1.0, atol=1.0e-6):
                raise ValueError(
                    "candidate probabilities must cover the full admissible set; "
                    "missing full actor distributions are a hard data error"
                )
            if np.any(self.q_count[~self.candidate_mask] > 0):
                raise ValueError("padded candidates cannot carry labels")

    @property
    def episodes(self) -> int:
        return int(self.state_embeddings.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.state_embeddings.shape[1])

    @property
    def branches(self) -> int:
        return int(self.branch_episode.shape[0])

    @property
    def max_candidates(self) -> int:
        return int(self.candidate_embeddings.shape[1])

    @property
    def state_dim(self) -> int:
        return int(self.state_embeddings.shape[2])

    @property
    def action_dim(self) -> int:
        return int(self.action_embeddings.shape[2])


def build_candidate_dataset(
    episodes: Sequence[EpisodeRecord],
    rows: Sequence[BranchTargetRow],
    distributions: Sequence[StatePolicyDistribution],
    embedder: TextEmbeddingProvider,
    *,
    horizon: int | None = None,
) -> CandidateSequenceDataset:
    """Embed episodes and branch rows into padded arrays.

    State text is ``goal | current observation``; action text is the complete
    normalized command, so held-out commands are scorable from their meaning.
    """

    if not episodes:
        raise ValueError("at least one episode is required")
    episode_index: dict[str, int] = {
        episode.episode_id: index for index, episode in enumerate(episodes)
    }
    max_len = max(len(episode.steps) for episode in episodes)
    resolved_horizon = horizon or max_len
    if max_len > resolved_horizon:
        raise ValueError(f"episode length {max_len} exceeds horizon {resolved_horizon}")
    dim_state = embedder.dimension
    dim_action = embedder.dimension

    n = len(episodes)
    state_embeddings = np.zeros((n, resolved_horizon, dim_state), dtype=np.float64)
    action_embeddings = np.zeros((n, resolved_horizon, dim_action), dtype=np.float64)
    step_mask = np.zeros((n, resolved_horizon), dtype=np.bool_)
    outcomes = np.zeros(n, dtype=np.float64)
    goals: dict[str, str] = {}
    for index, episode in enumerate(episodes):
        outcomes[index] = float(episode.success)
        goal = episode.steps[0].observation.goal if episode.steps else ""
        goals[episode.episode_id] = goal
        for position, step in enumerate(episode.steps):
            state_embeddings[index, position] = embedder.embed(
                f"{goal} | {step.observation.observation}"
            )
            action_embeddings[index, position] = embedder.embed(step.decision.action)
            step_mask[index, position] = True

    distribution_by_state: Mapping[str, StatePolicyDistribution] = {
        distribution.state_hash: distribution for distribution in distributions
    }
    grouped: dict[tuple[str, int], list[BranchTargetRow]] = {}
    for row in rows:
        grouped.setdefault((row.episode_id, row.step_index), []).append(row)
    keys = sorted(grouped)
    branches = len(keys)
    max_candidates = 1
    for key in keys:
        state_hash = grouped[key][0].state_hash
        if state_hash not in distribution_by_state:
            raise ValueError(
                f"missing full actor distribution for state {state_hash}; "
                "policy-centered Q is undefined"
            )
        max_candidates = max(max_candidates, len(distribution_by_state[state_hash].actions))

    branch_episode = np.zeros(branches, dtype=np.int64)
    branch_step = np.zeros(branches, dtype=np.int64)
    candidate_embeddings = np.zeros((branches, max_candidates, dim_action), dtype=np.float64)
    candidate_mask = np.zeros((branches, max_candidates), dtype=np.bool_)
    candidate_probabilities = np.zeros((branches, max_candidates), dtype=np.float64)
    q_success = np.zeros((branches, max_candidates), dtype=np.int64)
    q_count = np.zeros((branches, max_candidates), dtype=np.int64)
    v_success = np.zeros(branches, dtype=np.int64)
    v_count = np.zeros(branches, dtype=np.int64)
    forced_done = np.zeros((branches, max_candidates), dtype=np.bool_)
    forced_success = np.zeros((branches, max_candidates), dtype=np.float64)
    child_v_success = np.zeros((branches, max_candidates), dtype=np.int64)
    child_v_count = np.zeros((branches, max_candidates), dtype=np.int64)
    child_state_embeddings = np.zeros((branches, max_candidates, dim_state), dtype=np.float64)

    for branch, key in enumerate(keys):
        state_rows = grouped[key]
        first = state_rows[0]
        distribution = distribution_by_state[first.state_hash]
        branch_episode[branch] = episode_index[first.episode_id]
        branch_step[branch] = first.step_index
        goal = goals[first.episode_id]
        row_by_action = {row.candidate_action: row for row in state_rows}
        for slot, (action, probability) in enumerate(
            zip(distribution.actions, distribution.probabilities, strict=True)
        ):
            candidate_embeddings[branch, slot] = embedder.embed(action)
            candidate_mask[branch, slot] = True
            candidate_probabilities[branch, slot] = probability
            labeled_row = row_by_action.get(action)
            if labeled_row is None:
                continue
            q_success[branch, slot] = labeled_row.success_count
            q_count[branch, slot] = labeled_row.rollout_count
            forced_done[branch, slot] = labeled_row.forced_done
            forced_success[branch, slot] = labeled_row.forced_terminal_success
            child_v_success[branch, slot] = labeled_row.child_unforced_success_count
            child_v_count[branch, slot] = labeled_row.child_unforced_rollout_count
            child_state_embeddings[branch, slot] = embedder.embed(
                f"{goal} | {labeled_row.forced_next_observation}"
            )
        v_success[branch] = first.unforced_success_count
        v_count[branch] = first.unforced_rollout_count

    return CandidateSequenceDataset(
        state_embeddings=state_embeddings,
        action_embeddings=action_embeddings,
        step_mask=step_mask,
        outcomes=outcomes,
        branch_episode=branch_episode,
        branch_step=branch_step,
        candidate_embeddings=candidate_embeddings,
        candidate_mask=candidate_mask,
        candidate_probabilities=candidate_probabilities,
        q_success=q_success,
        q_count=q_count,
        v_success=v_success,
        v_count=v_count,
        forced_done=forced_done,
        forced_success=forced_success,
        child_v_success=child_v_success,
        child_v_count=child_v_count,
        child_state_embeddings=child_state_embeddings,
    )


def assert_no_locked_rows(rows: Sequence[BranchTargetRow]) -> None:
    """Hard guard: locked-reference rows must never enter a training path."""

    for row in rows:
        if row.target_role == "locked_reference":
            raise ValueError("locked_reference rows reached a training code path; run invalidated")
