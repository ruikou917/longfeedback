"""Tests for rule-based future-feedback labelers (torch/pyarrow-free)."""

from __future__ import annotations

from longfeedback.outcomes.rules import is_repetition, negative_signal, positive_signal


def test_negative_signals_hit_explicit_corrections() -> None:
    for message in (
        "That's wrong, the capital is Canberra.",
        "this doesn't work at all",
        "No, I asked for Python not Java",
        "you didn't answer my question",
        "still doesn't compile, I get an error",
        "That is incorrect.",
        "not what I asked for",
    ):
        assert negative_signal(message), message


def test_negative_signals_avoid_benign_messages() -> None:
    for message in (
        "Can you write a poem about the sea?",
        "Tell me more about that.",
        "What is the capital of Australia?",
        "Nothing beats a sunny day.",  # 'no' inside a word must not fire
        "That november trip was great.",
    ):
        assert not negative_signal(message), message


def test_positive_signals() -> None:
    assert positive_signal("Thanks, that works now!")
    assert positive_signal("perfect, exactly what I needed")
    assert not positive_signal("Can you explain how it works?")


def test_repetition_requires_real_overlap_and_length() -> None:
    request = "please write a python function that sorts a list of tuples by the second element"
    paraphrase = "write a python function that sorts a list of tuples by second element please"
    unrelated = "what is the weather like in berlin today in the winter season"
    assert is_repetition(paraphrase, request)
    assert not is_repetition(unrelated, request)
    # Short messages never count as repetitions.
    assert not is_repetition("sort it", "sort it")
