"""Text-only ALFWorld clients behind the :class:`EnvironmentClient` protocol.

Two interchangeable backends:

- :class:`InProcessAlfworldClient` imports TextWorld/ALFWorld lazily in this
  process (Linux/compatible Python environments only).
- :class:`SubprocessAlfworldClient` speaks line-delimited JSON to a dedicated
  worker process (``scripts/alfworld/worker.py``), which may run in a separate
  Python environment. This is the portability default; each parallel rollout
  worker owns one environment process.

Neither backend is exercised by the CPU test suite (CI uses the fake
environment); both require validation against installed ALFWorld data before
any real E11 run.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from longfeedback.environments.base import (
    EnvObservation,
    EnvTransition,
    GameRef,
    normalize_action,
    state_hash,
)

_SPLITS = ("train", "valid_seen", "valid_unseen")


@dataclass(frozen=True, slots=True)
class AlfworldSettings:
    """Locations and limits shared by both backends."""

    data_dir: Path
    max_steps: int = 50
    request_timeout_seconds: float = 120.0
    worker_command: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.request_timeout_seconds <= 0:
            raise ValueError("max_steps and request_timeout_seconds must be positive")


def list_game_files(data_dir: Path, split: str) -> list[Path]:
    if split not in _SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {_SPLITS}")
    root = data_dir / "json_2.1.1" / split
    if not root.is_dir():
        raise FileNotFoundError(f"ALFWorld split directory not found: {root}")
    return sorted(root.glob("**/*.tw-pddl"))


def _observation_from_payload(payload: dict[str, Any]) -> EnvObservation:
    admissible = tuple(str(a) for a in payload["admissible_actions"])
    return EnvObservation(
        game_id=str(payload["game_id"]),
        goal=str(payload["goal"]),
        observation=str(payload["observation"]),
        admissible_actions=admissible,
        step_index=int(payload["step_index"]),
        done=bool(payload["done"]),
        score=float(payload["score"]),
        state_hash=str(payload["state_hash"]),
    )


@dataclass
class InProcessAlfworldClient:
    """Runs TextWorld/ALFWorld in this process. Imports are lazy."""

    settings: AlfworldSettings
    _env: Any = field(default=None, init=False, repr=False)
    _game_id: str = field(default="", init=False, repr=False)
    _goal: str = field(default="", init=False, repr=False)
    _step_index: int = field(default=0, init=False, repr=False)
    _done: bool = field(default=False, init=False, repr=False)

    def list_games(self, split: str) -> tuple[GameRef, ...]:
        files = list_game_files(self.settings.data_dir, split)
        return tuple(
            GameRef(
                game_id=str(path.relative_to(self.settings.data_dir)),
                split=split,
            )
            for path in files
        )

    def reset(self, game: GameRef, *, seed: int) -> EnvObservation:
        import textworld
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler,
        )

        game_file = self.settings.data_dir / game.game_id
        request_infos = textworld.EnvInfos(
            admissible_commands=True, won=True, lost=True, extras=["gamefile"]
        )
        if self._env is not None:
            self._env.close()
        self._env = textworld.start(str(game_file), request_infos, wrappers=[AlfredDemangler()])
        self._env.seed(seed)
        game_state = self._env.reset()
        self._game_id = game.game_id
        self._goal = str(game_state.objective or "")
        self._step_index = 0
        self._done = False
        return self._observe(str(game_state.feedback or ""), game_state)

    def step(self, action: str) -> EnvTransition:
        if self._env is None or self._done:
            raise RuntimeError("reset must be called before step, and the episode must be live")
        normalized = normalize_action(action)
        game_state, _, done = self._env.step(normalized)
        self._step_index += 1
        self._done = bool(done) or self._step_index >= self.settings.max_steps
        observation = self._observe(str(game_state.feedback or ""), game_state)
        return EnvTransition(action=normalized, observation=observation)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _observe(self, feedback: str, game_state: Any) -> EnvObservation:
        won = bool(getattr(game_state, "won", False))
        done = self._done or won or bool(getattr(game_state, "lost", False))
        self._done = done
        admissible = (
            ()
            if done
            else tuple(
                sorted(
                    normalize_action(command) for command in game_state.admissible_commands or ()
                )
            )
        )
        score = 1.0 if won else 0.0
        return EnvObservation(
            game_id=self._game_id,
            goal=self._goal,
            observation=feedback,
            admissible_actions=admissible,
            step_index=self._step_index,
            done=done,
            score=score,
            state_hash=state_hash(
                game_id=self._game_id,
                step_index=self._step_index,
                observation=feedback,
                admissible_actions=admissible,
                score=score,
                done=done,
            ),
        )


@dataclass
class SubprocessAlfworldClient:
    """Line-delimited JSON client for a dedicated ALFWorld worker process."""

    settings: AlfworldSettings
    _process: subprocess.Popen[str] | None = field(default=None, init=False, repr=False)

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is None or self._process.poll() is not None:
            command = self.settings.worker_command or (
                "python",
                "scripts/alfworld/worker.py",
                "--data-dir",
                str(self.settings.data_dir),
                "--max-steps",
                str(self.settings.max_steps),
            )
            self._process = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        return self._process

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._ensure_process()
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("ALFWorld worker terminated unexpectedly")
        response: dict[str, Any] = json.loads(line)
        if not response.get("ok", False):
            raise RuntimeError(f"ALFWorld worker error: {response.get('error', 'unknown')}")
        result: dict[str, Any] = response["result"]
        return result

    def list_games(self, split: str) -> tuple[GameRef, ...]:
        result = self._request({"op": "list_games", "split": split})
        return tuple(GameRef(game_id=str(g), split=split) for g in result["games"])

    def reset(self, game: GameRef, *, seed: int) -> EnvObservation:
        result = self._request(
            {"op": "reset", "game_id": game.game_id, "split": game.split, "seed": seed}
        )
        return _observation_from_payload(result)

    def step(self, action: str) -> EnvTransition:
        result = self._request({"op": "step", "action": normalize_action(action)})
        return EnvTransition(
            action=normalize_action(action),
            observation=_observation_from_payload(result),
        )

    def close(self) -> None:
        if self._process is not None:
            with contextlib.suppress(RuntimeError, BrokenPipeError, OSError):
                self._request({"op": "close"})
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
