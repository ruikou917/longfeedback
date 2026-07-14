"""A small trainable candidate policy for CPU smoke runs of E12.

This plays the role of the LoRA LLM actor in the fake environment: it scores
each candidate command from frozen text embeddings of the leakage-safe prompt
and the candidate, and it is updated by the shared group-policy objective.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from longfeedback.actors.base import (
    PolicyDecision,
    PolicyScores,
    canonical_candidates,
    sample_from_scores,
    softmax_scores,
)
from longfeedback.models.text_embeddings import TextEmbeddingProvider


class _ScorerNet(nn.Module):
    def __init__(self, embed_dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: Tensor, actions: Tensor) -> Tensor:
        expanded = state.unsqueeze(0).expand(actions.shape[0], -1)
        features = torch.cat([expanded, actions, expanded * actions], dim=-1)
        scores: Tensor = self.mlp(features).squeeze(-1)
        return scores


@dataclass
class TrainableSoftmaxCandidatePolicy:
    """Differentiable candidate scorer implementing :class:`CandidatePolicy`."""

    embedder: TextEmbeddingProvider
    hidden: int = 32
    temperature: float = 1.0
    seed: int = 0
    _net: _ScorerNet = field(init=False, repr=False)

    def __post_init__(self) -> None:
        generator = torch.Generator().manual_seed(self.seed)
        self._net = _ScorerNet(self.embedder.dimension, self.hidden)
        with torch.no_grad():
            for parameter in self._net.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter, generator=generator)
                else:
                    parameter.zero_()

    @property
    def policy_id(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in sorted(self._net.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(parameter.numpy(), dtype=np.float32).tobytes())
        return f"toy-softmax:{digest.hexdigest()[:16]}"

    def parameters(self) -> list[nn.Parameter]:
        return list(self._net.parameters())

    def _embed_state(self, prompt: str) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(self.embedder.embed(prompt), dtype=np.float32))

    def _embed_candidates(self, candidates: tuple[str, ...]) -> Tensor:
        rows = [
            np.ascontiguousarray(self.embedder.embed(candidate), dtype=np.float32)
            for candidate in candidates
        ]
        return torch.from_numpy(np.stack(rows, axis=0))

    def candidate_log_probabilities(
        self, prompt: str, candidates: Sequence[str]
    ) -> tuple[tuple[str, ...], Tensor]:
        """Differentiable log-probabilities over the canonical candidate order."""

        canonical = canonical_candidates(candidates)
        scores = self._net(self._embed_state(prompt), self._embed_candidates(canonical))
        return canonical, torch.log_softmax(scores / self.temperature, dim=-1)

    def score(self, prompt: str, candidates: Sequence[str]) -> PolicyScores:
        canonical = canonical_candidates(candidates)
        with torch.no_grad():
            raw = self._net(self._embed_state(prompt), self._embed_candidates(canonical))
        raw_scores = [float(value) for value in raw]
        token_counts = [len(candidate.split()) for candidate in canonical]
        forward_tokens = len(prompt.split()) + sum(token_counts)
        return softmax_scores(
            canonical,
            raw_scores,
            token_counts,
            forward_tokens=forward_tokens,
            temperature=self.temperature,
        )

    def sample(self, scores: PolicyScores, *, random_value: float) -> PolicyDecision:
        return sample_from_scores(scores, random_value=random_value)

    def state_dict(self) -> dict[str, Tensor]:
        return {name: tensor.clone() for name, tensor in self._net.state_dict().items()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self._net.load_state_dict(state)
