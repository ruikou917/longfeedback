"""Prompt contract and candidate-policy behavior (torch-free)."""

from __future__ import annotations

import math

from longfeedback.actors.base import (
    canonical_candidates,
    render_prompt,
    sample_from_scores,
    softmax_scores,
)
from longfeedback.actors.mock import MockCandidatePolicy


def test_prompt_contains_only_past_information() -> None:
    history = [("panel 1: red glowing", "press red 1")]
    prompt = render_prompt(
        goal="activate panels",
        history=history,
        observation="panel 2: blue glowing",
        admissible_actions=("press blue 2", "smash red 1"),
    )
    # The prompt is a pure function of the past; editing what would come
    # afterwards cannot change it because it is never an input.
    again = render_prompt(
        goal="activate panels",
        history=list(history),
        observation="panel 2: blue glowing",
        admissible_actions=("smash red 1", "press blue 2"),
    )
    assert prompt.text == again.text
    assert prompt.prompt_hash == again.prompt_hash
    assert "future" not in prompt.text


def test_prompt_hash_changes_with_history() -> None:
    base = render_prompt(goal="g", history=[], observation="o", admissible_actions=("a", "b"))
    extended = render_prompt(
        goal="g",
        history=[("o", "a")],
        observation="o2",
        admissible_actions=("a", "b"),
    )
    assert base.prompt_hash != extended.prompt_hash


def test_canonical_candidates_dedupes_and_sorts() -> None:
    assert canonical_candidates(("B x", "a Y", "b X")) == ("a y", "b x")


def test_candidate_order_does_not_change_probabilities() -> None:
    policy = MockCandidatePolicy(seed=3, verb_bias={"press": 0.5})
    first = policy.score("prompt", ("press red 1", "smash blue 2", "inspect teal 3"))
    second = policy.score("prompt", ("inspect teal 3", "press red 1", "smash blue 2"))
    assert first.candidates == second.candidates
    assert first.probabilities == second.probabilities


def test_mock_policy_is_deterministic_and_normalized() -> None:
    policy = MockCandidatePolicy(seed=0)
    scores = policy.score("some prompt", ("press red 1", "smash blue 2"))
    again = policy.score("some prompt", ("press red 1", "smash blue 2"))
    assert scores == again
    assert abs(sum(scores.probabilities) - 1.0) < 1.0e-9
    for probability, log_probability in zip(
        scores.probabilities, scores.log_probabilities, strict=True
    ):
        assert math.isclose(math.log(probability), log_probability, rel_tol=1.0e-9)


def test_inverse_cdf_sampling_provenance() -> None:
    scores = softmax_scores(("a", "b"), [0.0, 0.0], [1, 1], forward_tokens=2, temperature=1.0)
    low = sample_from_scores(scores, random_value=0.25)
    high = sample_from_scores(scores, random_value=0.75)
    assert low.action == "a" and high.action == "b"
    assert low.random_value == 0.25
    assert math.isclose(low.probability, 0.5, rel_tol=1.0e-9)
    assert math.isclose(low.entropy, math.log(2.0), rel_tol=1.0e-9)


def test_temperature_sharpens_distribution() -> None:
    sharp = MockCandidatePolicy(seed=1, temperature=0.25, verb_bias={"press": 1.0})
    soft = MockCandidatePolicy(seed=1, temperature=4.0, verb_bias={"press": 1.0})
    candidates = ("press red 1", "smash blue 2")
    sharp_scores = sharp.score("p", candidates)
    soft_scores = soft.score("p", candidates)
    press_index = sharp_scores.candidates.index("press red 1")
    assert sharp_scores.probabilities[press_index] > soft_scores.probabilities[press_index]
