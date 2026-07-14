"""Explicit rollout/compute budget accounting shared by E11 and E12."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    max_actor_forward_tokens: int
    max_environment_steps: int
    max_wall_time_seconds: float
    max_api_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if min(self.max_actor_forward_tokens, self.max_environment_steps) <= 0:
            raise ValueError("token and environment-step limits must be positive")
        if self.max_wall_time_seconds <= 0 or self.max_api_cost_usd < 0:
            raise ValueError("wall-time must be positive and api cost non-negative")


@dataclass
class BudgetLedger:
    """Counts every base, group, and branch continuation against hard limits."""

    limits: BudgetLimits
    actor_forward_tokens: int = 0
    environment_steps: int = 0
    api_cost_usd: float = 0.0
    continuations: dict[str, int] = field(default_factory=dict)
    replay_integrity_failures: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    def add_actor_tokens(self, tokens: int) -> None:
        self.actor_forward_tokens += int(tokens)

    def add_environment_steps(self, steps: int) -> None:
        self.environment_steps += int(steps)

    def count_continuation(self, kind: str) -> None:
        self.continuations[kind] = self.continuations.get(kind, 0) + 1

    def count_replay_failure(self) -> None:
        self.replay_integrity_failures += 1

    @property
    def wall_time_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def exceeded(self) -> str | None:
        """Return the first exhausted budget dimension, or None."""

        if self.actor_forward_tokens > self.limits.max_actor_forward_tokens:
            return "actor_forward_tokens"
        if self.environment_steps > self.limits.max_environment_steps:
            return "environment_steps"
        if self.wall_time_seconds > self.limits.max_wall_time_seconds:
            return "wall_time_seconds"
        if self.api_cost_usd > self.limits.max_api_cost_usd:
            return "api_cost_usd"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "limits": {
                "max_actor_forward_tokens": self.limits.max_actor_forward_tokens,
                "max_environment_steps": self.limits.max_environment_steps,
                "max_wall_time_seconds": self.limits.max_wall_time_seconds,
                "max_api_cost_usd": self.limits.max_api_cost_usd,
            },
            "actor_forward_tokens": self.actor_forward_tokens,
            "environment_steps": self.environment_steps,
            "api_cost_usd": self.api_cost_usd,
            "continuations": dict(sorted(self.continuations.items())),
            "replay_integrity_failures": self.replay_integrity_failures,
            "wall_time_seconds": self.wall_time_seconds,
        }
