"""Environment protocol, canonical hashing, fake world, and exact replay."""

from __future__ import annotations

import pytest

from longfeedback.environments.base import (
    GameRef,
    ReplayHandle,
    ReplayMismatchError,
    normalize_text,
    restore_replay_handle,
    state_hash,
)
from longfeedback.environments.fake import (
    FakeTextEnvironment,
    FakeWorldSettings,
    uniform_policy_success_probability,
)


def test_normalize_text_collapses_case_and_whitespace() -> None:
    assert normalize_text("  Press   RED 3\n") == "press red 3"


def test_state_hash_is_invariant_to_admissible_order() -> None:
    kwargs = {
        "game_id": "g",
        "step_index": 2,
        "observation": "panel 1",
        "score": 0.5,
        "done": False,
    }
    first = state_hash(admissible_actions=("b", "a"), **kwargs)
    second = state_hash(admissible_actions=("a", "b"), **kwargs)
    assert first == second


def test_state_hash_changes_with_step_index() -> None:
    kwargs = {
        "game_id": "g",
        "observation": "panel 1",
        "admissible_actions": ("a", "b"),
        "score": 0.5,
        "done": False,
    }
    assert state_hash(step_index=1, **kwargs) != state_hash(step_index=2, **kwargs)


def test_replay_handle_json_round_trip() -> None:
    handle = ReplayHandle(
        game_id="fake/train/000",
        split="train",
        environment_seed=3,
        action_prefix=("press red 1", "inspect blue 2"),
        expected_state_hash="abc",
    )
    assert ReplayHandle.from_json(handle.to_json()) == handle
    assert handle.prefix_hash == ReplayHandle.from_json(handle.to_json()).prefix_hash


def test_fake_environment_is_deterministic() -> None:
    env = FakeTextEnvironment()
    game = env.list_games("train")[0]
    first = env.reset(game, seed=1)
    second = env.reset(game, seed=1)
    assert first == second
    action = first.admissible_actions[0]
    after_first = env.reset(game, seed=1) and env.step(action).observation
    after_second = env.reset(game, seed=1) and env.step(action).observation
    assert after_first == after_second


def test_fake_environment_success_and_trap_paths() -> None:
    settings = FakeWorldSettings(code_length=2, noop_actions=1, trap_actions=1, max_steps=5)
    env = FakeTextEnvironment(settings)
    game = env.list_games("train")[0]
    observation = env.reset(game, seed=0)
    while not observation.done:
        advance = next(
            action
            for action in observation.admissible_actions
            if action.split()[0] in ("press", "pull")
        )
        observation = env.step(advance).observation
    assert observation.score == 1.0

    observation = env.reset(game, seed=0)
    trap = next(
        action
        for action in observation.admissible_actions
        if action.split()[0] in ("smash", "yank")
    )
    observation = env.step(trap).observation
    assert observation.done and observation.score == 0.0


def test_exact_replay_restores_matching_state() -> None:
    env = FakeTextEnvironment()
    game = env.list_games("train")[1]
    observation = env.reset(game, seed=5)
    prefix = []
    for _ in range(2):
        action = sorted(observation.admissible_actions)[0]
        prefix.append(action)
        observation = env.step(action).observation
    handle = ReplayHandle(
        game_id=game.game_id,
        split=game.split,
        environment_seed=5,
        action_prefix=tuple(prefix),
        expected_state_hash=observation.state_hash,
    )
    restored = restore_replay_handle(env, handle)
    assert restored.state_hash == observation.state_hash


def test_replay_mismatch_raises_instead_of_failing_task() -> None:
    env = FakeTextEnvironment()
    game = env.list_games("train")[1]
    env.reset(game, seed=5)
    handle = ReplayHandle(
        game_id=game.game_id,
        split=game.split,
        environment_seed=5,
        action_prefix=(),
        expected_state_hash="not-the-real-hash",
    )
    with pytest.raises(ReplayMismatchError):
        restore_replay_handle(env, handle)


def test_unknown_split_rejected() -> None:
    env = FakeTextEnvironment()
    with pytest.raises(ValueError):
        env.list_games("test")
    assert env.reset(GameRef("fake/train/000", "train"), seed=0).step_index == 0


def test_uniform_policy_dp_is_a_probability_and_monotone() -> None:
    settings = FakeWorldSettings()
    start = uniform_policy_success_probability(settings, progress=0, steps_taken=0)
    later = uniform_policy_success_probability(settings, progress=0, steps_taken=3)
    almost = uniform_policy_success_probability(
        settings, progress=settings.code_length - 1, steps_taken=0
    )
    assert 0.0 < start < 1.0
    assert later <= start
    assert almost > start
