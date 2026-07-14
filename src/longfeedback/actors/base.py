"""Candidate-policy protocol and the leakage-safe prompt contract."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from longfeedback.environments.base import normalize_action, normalize_text

_INSTRUCTIONS = (
    "you are choosing the next command. pick exactly one command from the "
    "candidate list that best advances the goal."
)


def canonical_candidates(candidates: Sequence[str]) -> tuple[str, ...]:
    """Normalize, deduplicate, and deterministically sort candidate commands."""

    return tuple(sorted({normalize_action(candidate) for candidate in candidates}))


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """Prompt text plus a canonical hash for the run manifest."""

    text: str
    prompt_hash: str


def render_prompt(
    *,
    goal: str,
    history: Sequence[tuple[str, str]],
    observation: str,
    admissible_actions: Sequence[str],
) -> RenderedPrompt:
    """Render the decision prompt from information available before action t.

    ``history`` holds ``(observation, action)`` pairs through ``t - 1``. The
    prompt never contains future observations, expert plans, terminal success,
    branch outcomes, or reference targets.
    """

    lines = [f"goal: {normalize_text(goal)}"]
    for past_observation, past_action in history:
        lines.append(f"observation: {normalize_text(past_observation)}")
        lines.append(f"action: {normalize_action(past_action)}")
    lines.append(f"observation: {normalize_text(observation)}")
    lines.append("candidates: " + " | ".join(canonical_candidates(admissible_actions)))
    lines.append(f"instructions: {_INSTRUCTIONS}")
    text = "\n".join(lines)
    return RenderedPrompt(text=text, prompt_hash=hashlib.sha256(text.encode("utf-8")).hexdigest())


@dataclass(frozen=True, slots=True)
class PolicyScores:
    """Length-normalized candidate scores and the derived distribution."""

    candidates: tuple[str, ...]
    raw_scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    log_probabilities: tuple[float, ...]
    token_counts: tuple[int, ...]
    forward_tokens: int
    temperature: float

    def __post_init__(self) -> None:
        n = len(self.candidates)
        if not n:
            raise ValueError("PolicyScores needs at least one candidate")
        for name in ("raw_scores", "probabilities", "log_probabilities", "token_counts"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must match the candidate count")
        if abs(sum(self.probabilities) - 1.0) > 1.0e-6:
            raise ValueError("candidate probabilities must sum to one")

    @property
    def entropy(self) -> float:
        pairs = zip(self.probabilities, self.log_probabilities, strict=True)
        return -sum(p * lp for p, lp in pairs if p > 0.0)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """A sampled action with full RNG provenance."""

    action: str
    index: int
    probability: float
    log_probability: float
    entropy: float
    random_value: float


@runtime_checkable
class CandidatePolicy(Protocol):
    """Scores every admissible command; sampling is inverse-CDF given a uniform."""

    @property
    def policy_id(self) -> str: ...

    def score(self, prompt: str, candidates: Sequence[str]) -> PolicyScores: ...

    def sample(self, scores: PolicyScores, *, random_value: float) -> PolicyDecision: ...


def softmax_scores(
    candidates: tuple[str, ...],
    raw_scores: Sequence[float],
    token_counts: Sequence[int],
    *,
    forward_tokens: int,
    temperature: float,
) -> PolicyScores:
    """Build :class:`PolicyScores` from raw (already length-normalized) scores."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scaled = [score / temperature for score in raw_scores]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = sum(weights)
    probabilities = tuple(weight / total for weight in weights)
    log_probabilities = tuple(value - peak - math.log(total) for value in scaled)
    return PolicyScores(
        candidates=candidates,
        raw_scores=tuple(raw_scores),
        probabilities=probabilities,
        log_probabilities=log_probabilities,
        token_counts=tuple(token_counts),
        forward_tokens=forward_tokens,
        temperature=temperature,
    )


def sample_from_scores(scores: PolicyScores, *, random_value: float) -> PolicyDecision:
    """Inverse-CDF sampling over the canonical candidate order."""

    if not 0.0 <= random_value < 1.0:
        raise ValueError("random_value must be in [0, 1)")
    cumulative = 0.0
    index = len(scores.candidates) - 1
    for position, probability in enumerate(scores.probabilities):
        cumulative += probability
        if random_value < cumulative:
            index = position
            break
    return PolicyDecision(
        action=scores.candidates[index],
        index=index,
        probability=scores.probabilities[index],
        log_probability=scores.log_probabilities[index],
        entropy=scores.entropy,
        random_value=random_value,
    )
