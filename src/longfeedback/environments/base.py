"""Environment protocol, canonical hashing, and exact replay for E11/E12.

The experiment code never imports a concrete environment package directly; it
talks to :class:`EnvironmentClient`. A prefix of actions plus the environment
seed is a complete replay handle because every supported environment is
deterministic given ``(game, seed, action prefix)``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


def normalize_text(text: str) -> str:
    """Canonical text form: lowercase, collapsed whitespace, stripped."""

    return " ".join(text.lower().split())


def normalize_action(action: str) -> str:
    return normalize_text(action)


@dataclass(frozen=True, slots=True)
class GameRef:
    """A game inside one official split; ``game_id`` is globally unique."""

    game_id: str
    split: str


@dataclass(frozen=True, slots=True)
class EnvObservation:
    game_id: str
    goal: str
    observation: str
    admissible_actions: tuple[str, ...]
    step_index: int
    done: bool
    score: float
    state_hash: str


@dataclass(frozen=True, slots=True)
class EnvTransition:
    """The normalized action taken and the observation it produced."""

    action: str
    observation: EnvObservation


def state_hash(
    *,
    game_id: str,
    step_index: int,
    observation: str,
    admissible_actions: tuple[str, ...],
    score: float,
    done: bool,
) -> str:
    """Canonical SHA-256 state fingerprint used to verify exact replay."""

    payload = json.dumps(
        {
            "game_id": game_id,
            "step_index": step_index,
            "observation": normalize_text(observation),
            "admissible_actions": sorted(normalize_action(a) for a in admissible_actions),
            "score": round(float(score), 6),
            "done": bool(done),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def replay_prefix_hash(game_id: str, environment_seed: int, action_prefix: tuple[str, ...]) -> str:
    payload = json.dumps(
        {
            "game_id": game_id,
            "environment_seed": int(environment_seed),
            "action_prefix": [normalize_action(a) for a in action_prefix],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class EnvironmentClient(Protocol):
    """Synchronous single-environment client; one client per rollout worker."""

    def list_games(self, split: str) -> tuple[GameRef, ...]: ...

    def reset(self, game: GameRef, *, seed: int) -> EnvObservation: ...

    def step(self, action: str) -> EnvTransition: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ReplayHandle:
    """Reset-and-replay restoration record for one branch state."""

    game_id: str
    split: str
    environment_seed: int
    action_prefix: tuple[str, ...]
    expected_state_hash: str

    @property
    def prefix_hash(self) -> str:
        return replay_prefix_hash(self.game_id, self.environment_seed, self.action_prefix)

    def to_json(self) -> str:
        return json.dumps(
            {
                "game_id": self.game_id,
                "split": self.split,
                "environment_seed": self.environment_seed,
                "action_prefix": list(self.action_prefix),
                "expected_state_hash": self.expected_state_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def from_json(payload: str) -> ReplayHandle:
        raw = json.loads(payload)
        return ReplayHandle(
            game_id=str(raw["game_id"]),
            split=str(raw["split"]),
            environment_seed=int(raw["environment_seed"]),
            action_prefix=tuple(str(a) for a in raw["action_prefix"]),
            expected_state_hash=str(raw["expected_state_hash"]),
        )


class ReplayMismatchError(RuntimeError):
    """The restored state hash disagreed with the handle.

    A mismatch aborts the branch and is counted as a replay-integrity failure;
    it must never be recorded as a failed task rollout.
    """


def restore_replay_handle(client: EnvironmentClient, handle: ReplayHandle) -> EnvObservation:
    """Reset the game and replay the prefix, verifying the final state hash."""

    observation = client.reset(
        GameRef(game_id=handle.game_id, split=handle.split), seed=handle.environment_seed
    )
    for action in handle.action_prefix:
        if observation.done:
            raise ReplayMismatchError(
                f"episode terminated before prefix completed for {handle.game_id}"
            )
        observation = client.step(action).observation
    if observation.state_hash != handle.expected_state_hash:
        raise ReplayMismatchError(
            f"state hash mismatch for {handle.game_id}: "
            f"expected {handle.expected_state_hash}, got {observation.state_hash}"
        )
    return observation
