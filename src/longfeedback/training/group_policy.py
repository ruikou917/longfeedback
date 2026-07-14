"""Shared categorical group-policy objective (design section 11.3).

Every E12 method uses this optimization shell unchanged; the compared methods
differ only in how per-step advantages are computed. Only candidate-action
policy probabilities are optimized; prompts and environment text receive no
direct loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from longfeedback.actors.trainable import TrainableSoftmaxCandidatePolicy

_STD_FLOOR = 1.0e-8


@dataclass(frozen=True, slots=True)
class GroupPolicySettings:
    ratio_clip: float = 0.2
    kl_coefficient: float = 0.01
    entropy_coefficient: float = 0.01
    update_epochs: int = 2
    learning_rate: float = 0.05
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio_clip < 1.0:
            raise ValueError("ratio_clip must be in (0, 1)")
        if self.kl_coefficient < 0.0 or self.entropy_coefficient < 0.0:
            raise ValueError("kl and entropy coefficients cannot be negative")
        if self.update_epochs <= 0 or self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("update_epochs, learning_rate, and max_grad_norm must be positive")


@dataclass(frozen=True, slots=True)
class PolicyUpdateStep:
    """One logged decision prepared for the policy update."""

    prompt: str
    candidates: tuple[str, ...]
    chosen_index: int
    old_log_probability: float
    advantage: float


def center_advantages(values: list[float], *, normalize: bool = True) -> list[float]:
    """Batch centering with the declared degenerate-group convention (zeros)."""

    if len(values) < 2:
        return [0.0 for _ in values]
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    if not normalize:
        return centered
    std = (sum(value**2 for value in centered) / len(values)) ** 0.5
    if std < _STD_FLOOR:
        return [0.0 for _ in values]
    return [value / std for value in centered]


def clipped_surrogate(
    new_log_probability: Tensor,
    old_log_probability: Tensor,
    advantage: Tensor,
    *,
    ratio_clip: float,
) -> Tensor:
    ratio = torch.exp(new_log_probability - old_log_probability)
    clipped = torch.clamp(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)
    surrogate: Tensor = torch.minimum(ratio * advantage, clipped * advantage)
    return surrogate


def group_policy_update(
    policy: TrainableSoftmaxCandidatePolicy,
    reference: TrainableSoftmaxCandidatePolicy,
    steps: list[PolicyUpdateStep],
    settings: GroupPolicySettings,
) -> dict[str, float]:
    """Run the fixed number of clipped update epochs; returns diagnostics.

    KL is measured against the immutable initial reference actor, matching
    the design's regularization target.
    """

    if not steps:
        return {"updated_steps": 0.0, "final_loss": 0.0, "mean_kl": 0.0, "mean_entropy": 0.0}
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings.learning_rate)
    final_loss = 0.0
    mean_kl = 0.0
    mean_entropy = 0.0
    for _ in range(settings.update_epochs):
        surrogates: list[Tensor] = []
        kls: list[Tensor] = []
        entropies: list[Tensor] = []
        for step in steps:
            _, new_log_probs = policy.candidate_log_probabilities(step.prompt, step.candidates)
            with torch.no_grad():
                _, ref_log_probs = reference.candidate_log_probabilities(
                    step.prompt, step.candidates
                )
            new_probs = torch.exp(new_log_probs)
            surrogates.append(
                clipped_surrogate(
                    new_log_probs[step.chosen_index],
                    torch.tensor(step.old_log_probability),
                    torch.tensor(step.advantage),
                    ratio_clip=settings.ratio_clip,
                )
            )
            kls.append((new_probs * (new_log_probs - ref_log_probs)).sum())
            entropies.append(-(new_probs * new_log_probs).sum())
        surrogate = torch.stack(surrogates).mean()
        kl = torch.stack(kls).mean()
        entropy = torch.stack(entropies).mean()
        loss = -surrogate + settings.kl_coefficient * kl - settings.entropy_coefficient * entropy
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(policy.parameters(), settings.max_grad_norm)
        optimizer.step()
        final_loss = float(loss.detach())
        mean_kl = float(kl.detach())
        mean_entropy = float(entropy.detach())
    return {
        "updated_steps": float(len(steps)),
        "final_loss": final_loss,
        "mean_kl": mean_kl,
        "mean_entropy": mean_entropy,
    }
