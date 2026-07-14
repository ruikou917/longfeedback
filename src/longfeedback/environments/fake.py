"""A tiny deterministic text environment for CI and CPU smoke runs.

Each game is a "signal panel" puzzle derived deterministically from its game
ID. At panel ``p`` the observation names a glowing color; the admissible
commands contain exactly one advancing command for that color, some harmless
inspection commands, and some trap commands that end the episode in failure.
The dynamics depend only on ``(game_id, action history)``, so reset-and-replay
restoration is exact by construction.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from longfeedback.environments.base import (
    EnvObservation,
    EnvTransition,
    GameRef,
    normalize_action,
    state_hash,
)

_COLORS = (
    "red",
    "blue",
    "green",
    "amber",
    "violet",
    "teal",
    "coral",
    "ivory",
    "olive",
    "silver",
    "crimson",
    "indigo",
)
_ADVANCE_VERBS = ("press", "pull")
_NOOP_VERBS = ("inspect", "examine")
_TRAP_VERBS = ("smash", "yank")


@dataclass(frozen=True, slots=True)
class FakeWorldSettings:
    train_games: int = 12
    valid_seen_games: int = 6
    valid_unseen_games: int = 6
    code_length: int = 3
    noop_actions: int = 2
    trap_actions: int = 1
    max_steps: int = 10

    def __post_init__(self) -> None:
        if min(self.train_games, self.valid_seen_games, self.valid_unseen_games) <= 0:
            raise ValueError("every split needs at least one game")
        if self.code_length <= 0 or self.max_steps < self.code_length:
            raise ValueError("code_length must be positive and reachable within max_steps")
        if self.noop_actions < 0 or self.trap_actions < 0:
            raise ValueError("noop_actions and trap_actions cannot be negative")


@dataclass(frozen=True, slots=True)
class _Panel:
    color: str
    advance: str
    noops: tuple[str, ...]
    traps: tuple[str, ...]

    @property
    def commands(self) -> tuple[str, ...]:
        return (self.advance, *self.noops, *self.traps)


@dataclass(frozen=True, slots=True)
class _FakeGame:
    game_id: str
    goal: str
    panels: tuple[_Panel, ...]


def _derive_game(game_id: str, settings: FakeWorldSettings) -> _FakeGame:
    digest = hashlib.sha256(game_id.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    panels: list[_Panel] = []
    for _ in range(settings.code_length):
        colors = rng.sample(_COLORS, 1 + settings.noop_actions + settings.trap_actions)
        glowing = colors[0]
        slot = rng.randint(1, 9)
        advance = f"{rng.choice(_ADVANCE_VERBS)} {glowing} {slot}"
        noops = tuple(
            f"{rng.choice(_NOOP_VERBS)} {color} {rng.randint(1, 9)}"
            for color in colors[1 : 1 + settings.noop_actions]
        )
        traps = tuple(
            f"{rng.choice(_TRAP_VERBS)} {color} {rng.randint(1, 9)}"
            for color in colors[1 + settings.noop_actions :]
        )
        panels.append(_Panel(color=glowing, advance=advance, noops=noops, traps=traps))
    goal = f"activate all {settings.code_length} signal panels without breaking anything"
    return _FakeGame(game_id=game_id, goal=goal, panels=tuple(panels))


@dataclass
class _EpisodeState:
    game: _FakeGame
    seed: int
    progress: int = 0
    step_index: int = 0
    done: bool = False
    trapped: bool = False


@dataclass
class FakeTextEnvironment:
    """Deterministic :class:`EnvironmentClient` implementation for tests."""

    settings: FakeWorldSettings = field(default_factory=FakeWorldSettings)
    _state: _EpisodeState | None = field(default=None, init=False, repr=False)

    def list_games(self, split: str) -> tuple[GameRef, ...]:
        counts = {
            "train": self.settings.train_games,
            "valid_seen": self.settings.valid_seen_games,
            "valid_unseen": self.settings.valid_unseen_games,
        }
        if split not in counts:
            raise ValueError(f"unknown split {split!r}")
        return tuple(
            GameRef(game_id=f"fake/{split}/{index:03d}", split=split)
            for index in range(counts[split])
        )

    def reset(self, game: GameRef, *, seed: int) -> EnvObservation:
        self._state = _EpisodeState(game=_derive_game(game.game_id, self.settings), seed=seed)
        return self._observe()

    def step(self, action: str) -> EnvTransition:
        state = self._state
        if state is None:
            raise RuntimeError("reset must be called before step")
        if state.done:
            raise RuntimeError("cannot step a finished episode")
        normalized = normalize_action(action)
        panel = state.game.panels[state.progress]
        admissible = {normalize_action(command) for command in panel.commands}
        if normalized not in admissible:
            raise ValueError(f"inadmissible action {normalized!r}")
        state.step_index += 1
        if normalized == normalize_action(panel.advance):
            state.progress += 1
        elif normalized in {normalize_action(command) for command in panel.traps}:
            state.trapped = True
            state.done = True
        if state.progress >= self.settings.code_length:
            state.done = True
        if state.step_index >= self.settings.max_steps:
            state.done = True
        return EnvTransition(action=normalized, observation=self._observe())

    def close(self) -> None:
        self._state = None

    def _observe(self) -> EnvObservation:
        state = self._state
        assert state is not None
        total = self.settings.code_length
        score = state.progress / total
        if state.done:
            if state.trapped:
                observation = "a trap discharged and the console went dark"
            elif state.progress >= total:
                observation = "every panel is active; the console hums with success"
            else:
                observation = "time ran out before every panel was active"
            admissible: tuple[str, ...] = ()
        else:
            panel = state.game.panels[state.progress]
            observation = (
                f"panel {state.progress + 1} of {total}: the {panel.color} light is glowing"
            )
            admissible = tuple(sorted(normalize_action(command) for command in panel.commands))
        return EnvObservation(
            game_id=state.game.game_id,
            goal=state.game.goal,
            observation=observation,
            admissible_actions=admissible,
            step_index=state.step_index,
            done=state.done,
            score=score,
            state_hash=state_hash(
                game_id=state.game.game_id,
                step_index=state.step_index,
                observation=observation,
                admissible_actions=admissible,
                score=score,
                done=state.done,
            ),
        )


def uniform_policy_success_probability(
    settings: FakeWorldSettings, *, progress: int, steps_taken: int
) -> float:
    """Exact success probability of a uniform-random admissible policy.

    Used by tests to compare learned action values against dynamic-programming
    ground truth on the fake world.
    """

    total_actions = 1 + settings.noop_actions + settings.trap_actions
    cache: dict[tuple[int, int], float] = {}

    def value(current: int, step: int) -> float:
        if current >= settings.code_length:
            return 1.0
        if step >= settings.max_steps:
            return 0.0
        key = (current, step)
        if key not in cache:
            advance = value(current + 1, step + 1)
            stay = value(current, step + 1)
            cache[key] = (advance + settings.noop_actions * stay) / total_actions
        return cache[key]

    return value(progress, steps_taken)
