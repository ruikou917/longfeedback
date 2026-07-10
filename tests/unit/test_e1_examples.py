"""Tests for E1 example construction and leakage safety (needs torch import chain)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from longfeedback.experiments.e1 import (
    ConversationSteps,
    build_examples,
)


def _conversation(*user_messages: str, split: str = "train") -> ConversationSteps:
    return ConversationSteps(
        trajectory_id="test:conv",
        split=split,
        user_messages=tuple(user_messages),
        assistant_messages=tuple(f"assistant reply {index}" for index in range(len(user_messages))),
    )


def test_labels_read_only_future_user_messages() -> None:
    conversation = _conversation(
        "please sort this list in python for me quickly today",
        "that's wrong, it crashes",
        "thanks, works now",
    )
    examples = build_examples(conversation, window=4, repetition_threshold=0.6)
    assert len(examples) == 2  # last turn has no future evidence
    # Turn 0 fails at the next user turn; turn 1 does not.
    assert examples[0].fail_next == 1.0
    assert examples[1].fail_next == 0.0
    assert examples[0].fail_any == 1.0


def test_features_are_past_only() -> None:
    base = _conversation("first question here today", "second question", "third question")
    negated = _conversation("first question here today", "second question", "that's wrong")
    base_examples = build_examples(base, window=4, repetition_threshold=0.6)
    negated_examples = build_examples(negated, window=4, repetition_threshold=0.6)
    # Changing the FUTURE message flips the label of turn 1 but must leave
    # turn 1's features untouched.
    assert negated_examples[1].fail_next != base_examples[1].fail_next
    assert np.array_equal(negated_examples[1].window_features, base_examples[1].window_features)
    assert np.array_equal(negated_examples[1].trivial_features, base_examples[1].trivial_features)


def test_window_padding_is_left_aligned_zero_rows() -> None:
    conversation = _conversation("only one user turn", "and a second one")
    examples = build_examples(conversation, window=4, repetition_threshold=0.6)
    features = examples[0].window_features
    assert features.shape == (4, 11)
    assert np.all(features[:3] == 0.0)  # padding rows
    assert features[3, 0] == 1.0  # presence flag on the real row
