"""Deterministic mock candidate policy for smoke runs and CI."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from longfeedback.actors.base import (
    PolicyDecision,
    PolicyScores,
    canonical_candidates,
    sample_from_scores,
    softmax_scores,
)


def _unit_hash(*parts: str) -> float:
    """Deterministic pseudo-uniform value in [0, 1) from string parts."""

    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass(frozen=True)
class MockCandidatePolicy:
    """Hash-noise scorer with optional verb biases.

    ``verb_bias`` shifts the raw score of any candidate whose first word
    matches, which lets smoke configurations place the actor's success rate
    inside the E11 signal-gate range without training anything.
    """

    seed: int = 0
    temperature: float = 1.0
    noise_scale: float = 0.5
    prompt_noise_scale: float = 0.25
    verb_bias: Mapping[str, float] = field(default_factory=dict)

    @property
    def policy_id(self) -> str:
        bias = ",".join(f"{verb}={value}" for verb, value in sorted(self.verb_bias.items()))
        return (
            f"mock:seed={self.seed}:temperature={self.temperature}:"
            f"noise={self.noise_scale}:prompt_noise={self.prompt_noise_scale}:bias={bias}"
        )

    def score(self, prompt: str, candidates: Sequence[str]) -> PolicyScores:
        canonical = canonical_candidates(candidates)
        raw_scores = []
        token_counts = []
        for candidate in canonical:
            verb = candidate.split()[0] if candidate else ""
            score = float(self.verb_bias.get(verb, 0.0))
            score += self.noise_scale * (2.0 * _unit_hash(str(self.seed), candidate) - 1.0)
            score += self.prompt_noise_scale * (
                2.0 * _unit_hash(str(self.seed), prompt, candidate) - 1.0
            )
            raw_scores.append(score)
            token_counts.append(len(candidate.split()))
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
