"""Semantic candidate-action DOCM with bounded policy-centered dueling Q.

This model sits alongside (never replaces) the fixed-vocabulary
``DelayedOutcomeCreditModel``. It consumes variable-horizon episodes of
state/action text embeddings and scores full semantic candidate-action
embeddings with three heads:

1. terminal-outcome head;
2. prefix-value head trained on direct unforced continuation targets; and
3. a candidate action-value head whose policy-centered dueling construction
   makes ``V == sum_a pi(a|h) Q(h, a)`` hold architecturally.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from longfeedback.models.candidate_data import CandidateSequenceDataset
from longfeedback.models.encoders import CausalTransformerEncoder, EncoderArchitecture

FloatArray = npt.NDArray[np.float64]

CANDIDATE_VARIANTS = (
    "docm_outcome",
    "docm_prefix",
    "docm_dueling_credit",
    "docm_dueling_no_tree",
    "docm_independent_q",
)

Parameterization = Literal["policy_centered_dueling", "independent"]


@dataclass(frozen=True, slots=True)
class CandidateLossWeights:
    """Loss switches; zero disables a supervision family entirely."""

    outcome: float = 1.0
    direct_v: float = 0.0
    branch_q: float = 0.0
    tree: float = 0.0
    prefix_mc: float = 0.0
    hindsight_aux: float = 0.0

    def __post_init__(self) -> None:
        for name in ("outcome", "direct_v", "branch_q", "tree", "prefix_mc", "hindsight_aux"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} weight cannot be negative")
        if self.outcome + self.direct_v + self.branch_q + self.tree + self.prefix_mc <= 0.0:
            raise ValueError("at least one primary loss weight must be positive")
        if self.hindsight_aux > 0.0:
            raise NotImplementedError(
                "the semantic hindsight auxiliary is not part of the CPU milestone"
            )


def candidate_variant_spec(name: str) -> tuple[CandidateLossWeights, Parameterization]:
    """Capacity-matched variant table (design section 9.6)."""

    if name == "docm_outcome":
        return CandidateLossWeights(outcome=1.0), "policy_centered_dueling"
    if name == "docm_prefix":
        return CandidateLossWeights(outcome=1.0, prefix_mc=1.0), "policy_centered_dueling"
    if name == "docm_dueling_credit":
        return (
            CandidateLossWeights(outcome=1.0, direct_v=1.0, branch_q=1.0, tree=0.25),
            "policy_centered_dueling",
        )
    if name == "docm_dueling_no_tree":
        return (
            CandidateLossWeights(outcome=1.0, direct_v=1.0, branch_q=1.0),
            "policy_centered_dueling",
        )
    if name == "docm_independent_q":
        return (
            CandidateLossWeights(outcome=1.0, direct_v=1.0, branch_q=1.0, tree=0.25),
            "independent",
        )
    raise ValueError(f"unknown candidate variant {name!r}")


# Absolute ceiling on the dueling scale. The probability bounds are almost
# always the binding constraint; this cap only prevents a near-zero policy
# mean of `d` from admitting an astronomically large scale that would amplify
# floating-point centering error past the 1e-6 identity contract.
_SCALE_CEILING = 10.0


def policy_centered_q(
    u: Tensor,
    v_logit: Tensor,
    scale_logit: Tensor,
    probabilities: Tensor,
    mask: Tensor,
    *,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Bounded probability-space dueling construction (design section 9.3).

    Returns ``(q, v, scale)`` in the input dtype; internally computed in
    float64 so the architectural identity ``V == sum_a pi(a|h) Q`` holds to
    1e-6 even for extreme logits. The double centering (before and after
    ``tanh``) keeps the policy mean exactly zero, and the shared scale is
    bounded so every Q stays in ``[epsilon, 1 - epsilon]``. Padded candidates
    and zero-probability actions never enter the center.
    """

    dtype = u.dtype
    u = u.double()
    v_logit = v_logit.double()
    scale_logit = scale_logit.double()
    probabilities = probabilities.double()
    probs = torch.where(mask, probabilities, torch.zeros_like(probabilities))
    raw = torch.where(mask, u, torch.zeros_like(u))
    first_center = (probs * raw).sum(dim=-1, keepdim=True)
    squashed = torch.tanh(raw - first_center)
    second_center = (probs * squashed).sum(dim=-1, keepdim=True)
    direction = torch.where(mask, squashed - second_center, torch.zeros_like(squashed))

    v = torch.sigmoid(v_logit)
    v_column = v.unsqueeze(-1)
    infinity = torch.full_like(direction, float("inf"))
    upper_terms = torch.where(
        mask & (direction > 0), (1.0 - epsilon - v_column) / direction, infinity
    )
    lower_terms = torch.where(mask & (direction < 0), (v_column - epsilon) / (-direction), infinity)
    scale_max = torch.minimum(upper_terms.min(dim=-1).values, lower_terms.min(dim=-1).values)
    scale_max = torch.where(torch.isinf(scale_max), torch.zeros_like(scale_max), scale_max)
    scale_max = torch.clamp(scale_max, min=0.0, max=_SCALE_CEILING)
    scale = torch.sigmoid(scale_logit) * scale_max
    q = v_column + scale.unsqueeze(-1) * direction
    return q.to(dtype), v.to(dtype), scale.to(dtype)


class _CandidateDocmNet(nn.Module):
    """Interleaved state/action embedding tokens with three heads."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        max_horizon: int,
        architecture: EncoderArchitecture,
        action_mlp_hidden: int,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim <= 0 or max_horizon <= 0:
            raise ValueError("state_dim, action_dim, and max_horizon must be positive")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_horizon = max_horizon
        token_dim = 2 + state_dim + action_dim
        # +2 positions allow an appended candidate action token plus one child
        # observation token for tree bootstrap targets.
        self.encoder = CausalTransformerEncoder(
            input_dim=token_dim,
            max_length=2 * max_horizon + 2,
            architecture=architecture,
        )
        d_model = architecture.d_model
        self.outcome_head = nn.Linear(d_model, 1)
        self.value_head = nn.Linear(d_model, 1)
        self.scale_head = nn.Linear(d_model, 1)
        self.action_projection = nn.Linear(action_dim, d_model)
        self.advantage_mlp = nn.Sequential(
            nn.Linear(4 * d_model, action_mlp_hidden),
            nn.Tanh(),
            nn.Linear(action_mlp_hidden, 1),
        )

    def state_token(self, states: Tensor) -> Tensor:
        batch = states.shape[:-1]
        token = torch.zeros(*batch, 2 + self.state_dim + self.action_dim, dtype=states.dtype)
        token[..., 0] = 1.0
        token[..., 2 : 2 + self.state_dim] = states
        return token

    def action_token(self, actions: Tensor) -> Tensor:
        batch = actions.shape[:-1]
        token = torch.zeros(*batch, 2 + self.state_dim + self.action_dim, dtype=actions.dtype)
        token[..., 1] = 1.0
        token[..., 2 + self.state_dim :] = actions
        return token

    def episode_tokens(self, states: Tensor, actions: Tensor, mask: Tensor) -> Tensor:
        batch, horizon = mask.shape
        tokens = torch.zeros(
            batch, 2 * horizon, 2 + self.state_dim + self.action_dim, dtype=states.dtype
        )
        step_mask = mask.to(states.dtype).unsqueeze(-1)
        tokens[:, 0::2, :] = self.state_token(states) * step_mask
        tokens[:, 1::2, :] = self.action_token(actions) * step_mask
        return tokens

    def encode_episodes(self, states: Tensor, actions: Tensor, mask: Tensor) -> Tensor:
        """Causal encoding; right padding cannot influence real positions."""

        encoded: Tensor = self.encoder(self.episode_tokens(states, actions, mask))
        return encoded

    def raw_advantage(self, hidden: Tensor, candidate_embeddings: Tensor) -> Tensor:
        """``u(h, a)`` for every candidate; hidden [B, d], candidates [B, C, da]."""

        projected = self.action_projection(candidate_embeddings)
        expanded = hidden.unsqueeze(1).expand_as(projected)
        features = torch.cat(
            [expanded, projected, expanded * projected, torch.abs(expanded - projected)],
            dim=-1,
        )
        scores: Tensor = self.advantage_mlp(features).squeeze(-1)
        return scores


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


@dataclass(frozen=True, slots=True)
class CandidateTrainingSettings:
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.grad_clip <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer settings")


@dataclass
class _BranchTensors:
    states: Tensor
    actions: Tensor
    step_mask: Tensor
    outcomes: Tensor
    lengths: Tensor
    branch_episode: Tensor
    branch_step: Tensor
    candidates: Tensor
    candidate_mask: Tensor
    probabilities: Tensor
    q_success: Tensor
    q_count: Tensor
    v_success: Tensor
    v_count: Tensor
    forced_done: Tensor
    forced_success: Tensor
    child_success: Tensor
    child_count: Tensor
    child_states: Tensor


@dataclass
class CandidateDelayedOutcomeCreditModel:
    """Trains and evaluates one capacity-matched candidate-DOCM variant."""

    state_dim: int
    action_dim: int
    max_horizon: int
    architecture: EncoderArchitecture = field(default_factory=EncoderArchitecture)
    loss_weights: CandidateLossWeights = field(default_factory=CandidateLossWeights)
    action_value_parameterization: Parameterization = "policy_centered_dueling"
    action_mlp_hidden: int = 128
    target_network_ema: float = 0.99
    probability_epsilon: float = 1.0e-4
    policy_center_tolerance: float = 1.0e-6
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_network_ema < 1.0:
            raise ValueError("target_network_ema must be in [0, 1)")
        if not 0.0 < self.probability_epsilon < 0.5:
            raise ValueError("probability_epsilon must be in (0, 0.5)")
        _seed_torch(self.seed)
        self._net = _CandidateDocmNet(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_horizon=self.max_horizon,
            architecture=self.architecture,
            action_mlp_hidden=self.action_mlp_hidden,
        )
        self._target_net: _CandidateDocmNet | None = None

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self._net.parameters())

    def _tensors(self, dataset: CandidateSequenceDataset) -> _BranchTensors:
        def as_float(array: npt.NDArray[np.float64]) -> Tensor:
            return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

        def as_long(array: npt.NDArray[np.int64]) -> Tensor:
            return torch.from_numpy(np.ascontiguousarray(array, dtype=np.int64))

        def as_bool(array: npt.NDArray[np.bool_]) -> Tensor:
            return torch.from_numpy(np.ascontiguousarray(array, dtype=np.bool_))

        return _BranchTensors(
            states=as_float(dataset.state_embeddings),
            actions=as_float(dataset.action_embeddings),
            step_mask=as_bool(dataset.step_mask),
            outcomes=as_float(dataset.outcomes),
            lengths=as_long(dataset.step_mask.sum(axis=1).astype(np.int64)),
            branch_episode=as_long(dataset.branch_episode),
            branch_step=as_long(dataset.branch_step),
            candidates=as_float(dataset.candidate_embeddings),
            candidate_mask=as_bool(dataset.candidate_mask),
            probabilities=as_float(dataset.candidate_probabilities),
            q_success=as_float(dataset.q_success.astype(np.float64)),
            q_count=as_float(dataset.q_count.astype(np.float64)),
            v_success=as_float(dataset.v_success.astype(np.float64)),
            v_count=as_float(dataset.v_count.astype(np.float64)),
            forced_done=as_bool(dataset.forced_done),
            forced_success=as_float(dataset.forced_success),
            child_success=as_float(dataset.child_v_success.astype(np.float64)),
            child_count=as_float(dataset.child_v_count.astype(np.float64)),
            child_states=as_float(dataset.child_state_embeddings),
        )

    def _branch_hidden(
        self, net: _CandidateDocmNet, tensors: _BranchTensors, branch_indices: Tensor
    ) -> Tensor:
        """Hidden state at the branch observation token for selected rows."""

        episode_ids = tensors.branch_episode[branch_indices]
        unique_episodes, inverse = torch.unique(episode_ids, return_inverse=True)
        encoded = net.encode_episodes(
            tensors.states[unique_episodes],
            tensors.actions[unique_episodes],
            tensors.step_mask[unique_episodes],
        )
        positions = 2 * tensors.branch_step[branch_indices]
        return encoded[inverse, positions, :]

    def _branch_q(
        self,
        net: _CandidateDocmNet,
        tensors: _BranchTensors,
        branch_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """(q, v, u) at the selected branch rows under the parameterization."""

        hidden = self._branch_hidden(net, tensors, branch_indices)
        candidates = tensors.candidates[branch_indices]
        mask = tensors.candidate_mask[branch_indices]
        probabilities = tensors.probabilities[branch_indices]
        u = net.raw_advantage(hidden, candidates)
        v_logit = net.value_head(hidden).squeeze(-1)
        if self.action_value_parameterization == "independent":
            q = torch.sigmoid(u)
            return q, torch.sigmoid(v_logit), u
        scale_logit = net.scale_head(hidden).squeeze(-1)
        q, v, _ = policy_centered_q(
            u,
            v_logit,
            scale_logit,
            probabilities,
            mask,
            epsilon=self.probability_epsilon,
        )
        return q, v, u

    def _appended_sequences(
        self,
        tensors: _BranchTensors,
        *,
        include_child: bool,
        only_rows: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Flattened prefix+candidate(+child) sequences for real candidates.

        Returns (tokens, read_positions, row_index, slot_index). No post-action
        environment observation is appended unless ``include_child``.
        """

        reads: list[int] = []
        row_ids: list[int] = []
        slot_ids: list[int] = []
        branch_rows = (
            torch.arange(tensors.branch_episode.shape[0]) if only_rows is None else only_rows
        )
        episode_tokens = self._net.episode_tokens(
            tensors.states, tensors.actions, tensors.step_mask
        )
        max_len = 0
        sequences: list[Tensor] = []
        for row in branch_rows.tolist():
            episode = int(tensors.branch_episode[row])
            step = int(tensors.branch_step[row])
            prefix = episode_tokens[episode, : 2 * step + 1, :]
            for slot in range(tensors.candidate_mask.shape[1]):
                if not bool(tensors.candidate_mask[row, slot]):
                    continue
                pieces = [
                    prefix,
                    self._net.action_token(tensors.candidates[row, slot]).unsqueeze(0),
                ]
                if include_child:
                    pieces.append(
                        self._net.state_token(tensors.child_states[row, slot]).unsqueeze(0)
                    )
                sequence = torch.cat(pieces, dim=0)
                sequences.append(sequence)
                reads.append(sequence.shape[0] - 1)
                row_ids.append(row)
                slot_ids.append(slot)
                max_len = max(max_len, sequence.shape[0])
        if not sequences:
            empty = torch.zeros(0, dtype=torch.long)
            return torch.zeros(0, 1, episode_tokens.shape[-1]), empty, empty, empty
        tokens = torch.zeros(len(sequences), max_len, episode_tokens.shape[-1])
        for index, sequence in enumerate(sequences):
            tokens[index, : sequence.shape[0], :] = sequence
        return (
            tokens,
            torch.tensor(reads, dtype=torch.long),
            torch.tensor(row_ids, dtype=torch.long),
            torch.tensor(slot_ids, dtype=torch.long),
        )

    def _binomial_nll(self, probabilities: Tensor, successes: Tensor, counts: Tensor) -> Tensor:
        """Binomial negative log likelihood normalized by total rollout count."""

        total = counts.sum()
        if float(total) <= 0.0:
            return torch.zeros(())
        clamped = torch.clamp(probabilities, 1.0e-6, 1.0 - 1.0e-6)
        log_likelihood = successes * torch.log(clamped) + (counts - successes) * torch.log(
            1.0 - clamped
        )
        return -(log_likelihood * (counts > 0)).sum() / total

    def fit(
        self,
        dataset: CandidateSequenceDataset,
        *,
        training: CandidateTrainingSettings | None = None,
    ) -> dict[str, float]:
        settings = training or CandidateTrainingSettings()
        _seed_torch(self.seed)
        tensors = self._tensors(dataset)
        weights = self.loss_weights

        use_tree = weights.tree > 0.0 and dataset.branches > 0
        if use_tree:
            self._target_net = copy.deepcopy(self._net)
            for parameter in self._target_net.parameters():
                parameter.requires_grad_(False)
            labeled = tensors.q_count > 0
            bootstrap_mask = labeled & ~tensors.forced_done & (tensors.child_count <= 0)
            bootstrap_rows = bootstrap_mask.any(dim=1).nonzero().squeeze(-1)
            (
                bootstrap_tokens,
                bootstrap_reads,
                bootstrap_row_ids,
                bootstrap_slot_ids,
            ) = self._appended_sequences(tensors, include_child=True, only_rows=bootstrap_rows)
            keep = bootstrap_mask[bootstrap_row_ids, bootstrap_slot_ids]
            bootstrap_tokens = bootstrap_tokens[keep]
            bootstrap_reads = bootstrap_reads[keep]
            bootstrap_row_ids = bootstrap_row_ids[keep]
            bootstrap_slot_ids = bootstrap_slot_ids[keep]

        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        generator = torch.Generator().manual_seed(self.seed)
        components: dict[str, float] = {}
        final_loss = 0.0
        self._net.train()
        for _ in range(settings.epochs):
            bootstrap_values: Tensor | None = None
            if use_tree and bootstrap_tokens.shape[0] > 0:
                assert self._target_net is not None
                with torch.no_grad():
                    encoded = self._target_net.encoder(bootstrap_tokens)
                    picked = encoded[torch.arange(encoded.shape[0]), bootstrap_reads, :]
                    bootstrap_values = torch.sigmoid(
                        self._target_net.value_head(picked).squeeze(-1)
                    )

            epoch_loss = 0.0
            batches = 0
            order = torch.randperm(dataset.episodes, generator=generator)
            for start in range(0, dataset.episodes, settings.batch_size):
                batch = order[start : start + settings.batch_size]
                loss, parts = self._episode_loss(tensors, batch)
                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()  # type: ignore[no-untyped-call]
                    nn.utils.clip_grad_norm_(self._net.parameters(), settings.grad_clip)
                    optimizer.step()
                epoch_loss += float(loss.detach())
                batches += 1
                components.update(parts)

            if dataset.branches > 0 and (
                weights.direct_v > 0 or weights.branch_q > 0 or weights.tree > 0
            ):
                branch_order = torch.randperm(dataset.branches, generator=generator)
                for start in range(0, dataset.branches, settings.batch_size):
                    branch_batch = branch_order[start : start + settings.batch_size]
                    loss, parts = self._branch_loss(
                        tensors,
                        branch_batch,
                        bootstrap_values=bootstrap_values,
                        bootstrap_row_ids=bootstrap_row_ids if use_tree else None,
                        bootstrap_slot_ids=bootstrap_slot_ids if use_tree else None,
                    )
                    if loss.requires_grad:
                        optimizer.zero_grad()
                        loss.backward()  # type: ignore[no-untyped-call]
                        nn.utils.clip_grad_norm_(self._net.parameters(), settings.grad_clip)
                        optimizer.step()
                    epoch_loss += float(loss.detach())
                    batches += 1
                    components.update(parts)

            final_loss = epoch_loss / max(1, batches)
            if use_tree:
                assert self._target_net is not None
                with torch.no_grad():
                    for target, online in zip(
                        self._target_net.parameters(), self._net.parameters(), strict=True
                    ):
                        target.mul_(self.target_network_ema).add_(
                            online, alpha=1.0 - self.target_network_ema
                        )
        self._net.eval()
        return {"loss": final_loss, **{f"loss_{k}": v for k, v in components.items()}}

    def _episode_loss(
        self, tensors: _BranchTensors, batch: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        weights = self.loss_weights
        parts: dict[str, float] = {}
        encoded = self._net.encode_episodes(
            tensors.states[batch], tensors.actions[batch], tensors.step_mask[batch]
        )
        lengths = tensors.lengths[batch]
        final_positions = torch.clamp(2 * lengths - 1, min=0)
        final_hidden = encoded[torch.arange(encoded.shape[0]), final_positions, :]
        outcome_logit = self._net.outcome_head(final_hidden).squeeze(-1)
        outcomes = tensors.outcomes[batch]
        total = torch.zeros(())
        outcome_loss = nn.functional.binary_cross_entropy_with_logits(outcome_logit, outcomes)
        parts["outcome"] = float(outcome_loss.detach())
        total = total + weights.outcome * outcome_loss
        if weights.prefix_mc > 0.0:
            observation_hidden = encoded[:, 0::2, :]
            prefix_logits = self._net.value_head(observation_hidden).squeeze(-1)
            mask = tensors.step_mask[batch].to(prefix_logits.dtype)
            targets = outcomes.unsqueeze(1).expand_as(prefix_logits)
            elementwise = nn.functional.binary_cross_entropy_with_logits(
                prefix_logits, targets, reduction="none"
            )
            prefix_loss = (elementwise * mask).sum() / torch.clamp(mask.sum(), min=1.0)
            parts["prefix_mc"] = float(prefix_loss.detach())
            total = total + weights.prefix_mc * prefix_loss
        return total, parts

    def _branch_loss(
        self,
        tensors: _BranchTensors,
        branch_batch: Tensor,
        *,
        bootstrap_values: Tensor | None,
        bootstrap_row_ids: Tensor | None,
        bootstrap_slot_ids: Tensor | None,
    ) -> tuple[Tensor, dict[str, float]]:
        weights = self.loss_weights
        parts: dict[str, float] = {}
        q, v, _ = self._branch_q(self._net, tensors, branch_batch)
        total = torch.zeros(())
        if weights.direct_v > 0.0:
            v_loss = self._binomial_nll(
                v, tensors.v_success[branch_batch], tensors.v_count[branch_batch]
            )
            parts["direct_v"] = float(v_loss.detach())
            total = total + weights.direct_v * v_loss
        if weights.branch_q > 0.0:
            q_loss = self._binomial_nll(
                q, tensors.q_success[branch_batch], tensors.q_count[branch_batch]
            )
            parts["branch_q"] = float(q_loss.detach())
            total = total + weights.branch_q * q_loss
        if weights.tree > 0.0:
            targets = torch.zeros_like(q)
            tree_weights = torch.zeros_like(q)
            labeled = tensors.q_count[branch_batch] > 0
            forced_done = tensors.forced_done[branch_batch]
            terminal = labeled & forced_done
            targets = torch.where(terminal, tensors.forced_success[branch_batch], targets)
            tree_weights = torch.where(terminal, torch.ones_like(tree_weights), tree_weights)
            child_counts = tensors.child_count[branch_batch]
            child_direct = labeled & ~forced_done & (child_counts > 0)
            child_rate = torch.where(
                child_counts > 0,
                tensors.child_success[branch_batch] / torch.clamp(child_counts, min=1.0),
                torch.zeros_like(child_counts),
            )
            child_se_sq = child_rate * (1.0 - child_rate) / torch.clamp(child_counts, min=1.0)
            child_weight = 1.0 / (child_se_sq + 1.0e-2)
            child_weight = child_weight / child_weight.max().clamp(min=1.0)
            targets = torch.where(child_direct, child_rate, targets)
            tree_weights = torch.where(child_direct, child_weight, tree_weights)
            if (
                bootstrap_values is not None
                and bootstrap_row_ids is not None
                and bootstrap_slot_ids is not None
            ):
                full_bootstrap = torch.zeros_like(tensors.forced_success)
                full_mask = torch.zeros_like(tensors.forced_done)
                full_bootstrap[bootstrap_row_ids, bootstrap_slot_ids] = bootstrap_values
                full_mask[bootstrap_row_ids, bootstrap_slot_ids] = True
                batch_bootstrap = full_bootstrap[branch_batch]
                batch_mask = full_mask[branch_batch]
                targets = torch.where(batch_mask, batch_bootstrap.detach(), targets)
                tree_weights = torch.where(batch_mask, torch.ones_like(tree_weights), tree_weights)
            weight_total = tree_weights.sum()
            if float(weight_total) > 0.0:
                tree_loss = (tree_weights * torch.square(q - targets)).sum() / weight_total
                parts["tree"] = float(tree_loss.detach())
                total = total + weights.tree * tree_loss
        return total, parts

    def predict_branch_q(self, dataset: CandidateSequenceDataset) -> FloatArray:
        """Q_hat for every real candidate; padded entries are zero."""

        tensors = self._tensors(dataset)
        self._net.eval()
        with torch.no_grad():
            q, _, _ = self._branch_q(self._net, tensors, torch.arange(dataset.branches))
            q = torch.where(tensors.candidate_mask, q, torch.zeros_like(q))
        return np.asarray(q.numpy(), dtype=np.float64)

    def predict_branch_value(self, dataset: CandidateSequenceDataset) -> FloatArray:
        tensors = self._tensors(dataset)
        self._net.eval()
        with torch.no_grad():
            _, v, _ = self._branch_q(self._net, tensors, torch.arange(dataset.branches))
        return np.asarray(v.numpy(), dtype=np.float64)

    def predict_terminal_probability(self, dataset: CandidateSequenceDataset) -> FloatArray:
        tensors = self._tensors(dataset)
        self._net.eval()
        with torch.no_grad():
            encoded = self._net.encode_episodes(tensors.states, tensors.actions, tensors.step_mask)
            positions = torch.clamp(2 * tensors.lengths - 1, min=0)
            hidden = encoded[torch.arange(encoded.shape[0]), positions, :]
            probability = torch.sigmoid(self._net.outcome_head(hidden).squeeze(-1))
        return np.asarray(probability.numpy(), dtype=np.float64)

    def predict_appended_candidate_scores(
        self, dataset: CandidateSequenceDataset, *, head: Literal["outcome", "value"]
    ) -> FloatArray:
        """Score candidates by appending the action token only (section 9.10)."""

        tensors = self._tensors(dataset)
        self._net.eval()
        with torch.no_grad():
            tokens, reads, row_ids, slot_ids = self._appended_sequences(
                tensors, include_child=False
            )
            scores = np.zeros((dataset.branches, dataset.max_candidates), dtype=np.float64)
            if tokens.shape[0] > 0:
                encoded = self._net.encoder(tokens)
                hidden = encoded[torch.arange(encoded.shape[0]), reads, :]
                selected_head = (
                    self._net.outcome_head if head == "outcome" else self._net.value_head
                )
                values = torch.sigmoid(selected_head(hidden).squeeze(-1))
                scores[row_ids.numpy(), slot_ids.numpy()] = values.numpy()
        return scores

    def policy_centering_residual(self, dataset: CandidateSequenceDataset) -> float:
        """max |V_hat - sum_a pi(a|h) Q_hat(h, a)| over branch rows."""

        if dataset.branches == 0:
            return 0.0
        q = self.predict_branch_q(dataset)
        v = self.predict_branch_value(dataset)
        probabilities = np.where(dataset.candidate_mask, dataset.candidate_probabilities, 0.0)
        centered = (probabilities * q).sum(axis=1)
        return float(np.max(np.abs(centered - v)))
