"""Tests for the source-generic conversation adapter core (torch/pyarrow-free)."""

from __future__ import annotations

import pytest

from longfeedback.data.conversations import (
    ConversationRecord,
    ConversationTurn,
    conversation_exclusion_reason,
    conversation_to_trajectory,
    sanitize_text,
    split_by_conversation_hash,
)
from longfeedback.schema import EventType, PrefixBoundary

_FILTERS = {
    "language": "English",
    "min_assistant_turns": 2,
    "max_message_chars": 100,
    "exclude_flagged": True,
    "include_redacted": True,
}


def _record(
    *,
    turns: tuple[tuple[str, str], ...],
    conversation_id: str = "conv-1",
    language: str = "English",
    flagged: bool = False,
    redacted: bool = False,
) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=conversation_id,
        turns=tuple(ConversationTurn(role=role, content=content) for role, content in turns),
        assistant_model="test-model",
        language=language,
        flagged=flagged,
        redacted=redacted,
        source="test-source",
        source_row_id="shard:0",
    )


_VALID_TURNS = (
    ("user", "first question"),
    ("assistant", "first answer"),
    ("user", "second question"),
    ("assistant", "second answer"),
)


def test_valid_conversation_is_kept() -> None:
    assert conversation_exclusion_reason(_record(turns=_VALID_TURNS), **_FILTERS) is None


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (_record(turns=_VALID_TURNS, flagged=True), "moderation_flagged"),
        (_record(turns=_VALID_TURNS, language="Portuguese"), "language"),
        (_record(turns=_VALID_TURNS[:2]), "too_few_assistant_turns"),
        (_record(turns=()), "empty"),
        (_record(turns=(("assistant", "hi"), ("user", "hello"))), "role_alternation"),
        (
            _record(turns=(("user", "a"), ("assistant", "b"), ("assistant", "c"), ("user", "d"))),
            "role_alternation",
        ),
        (_record(turns=(("user", "  "), ("assistant", "b"))), "empty_message"),
        (_record(turns=(("user", "a"), ("assistant", "x" * 101))), "message_too_long"),
    ],
)
def test_exclusion_reasons(record: ConversationRecord, reason: str) -> None:
    assert conversation_exclusion_reason(record, **_FILTERS) == reason


def test_trailing_user_message_is_dropped_not_rejected() -> None:
    record = _record(turns=(*_VALID_TURNS, ("user", "unanswered follow-up")))
    assert conversation_exclusion_reason(record, **_FILTERS) is None
    trajectory, _ = conversation_to_trajectory(record, conversation_index=0)
    assert len(trajectory.events) == len(_VALID_TURNS)


def test_redacted_conversations_can_be_excluded() -> None:
    record = _record(turns=_VALID_TURNS, redacted=True)
    filters = {**_FILTERS, "include_redacted": False}
    assert conversation_exclusion_reason(record, **filters) == "redacted"


def test_sanitize_text_redacts_high_precision_pii() -> None:
    text, hits = sanitize_text("mail me at jane.doe+x@example.co.uk or 415-555-0143 at 10.0.0.1")
    assert hits == 3
    assert "[EMAIL]" in text and "[PHONE]" in text and "[IP]" in text
    clean, none = sanitize_text("no personal data here, just version 3.5 and $12.99")
    assert none == 0 and clean == "no personal data here, just version 3.5 and $12.99"


def test_trajectory_events_are_one_per_message_and_ordered() -> None:
    record = _record(turns=_VALID_TURNS)
    trajectory, _ = conversation_to_trajectory(record, conversation_index=3)

    assert [event.event_type for event in trajectory.events] == [
        EventType.OBSERVATION,
        EventType.ACTION,
        EventType.OBSERVATION,
        EventType.ACTION,
    ]
    assert [event.step_index for event in trajectory.events] == [0, 0, 1, 1]
    assert trajectory.metadata["synthetic_time"] is True
    assert trajectory.behavior_policy_id == "test-model"

    # The scoring prefix H_t includes user message t and the action, never
    # any future message.
    prefix = trajectory.model_input_events(0, boundary=PrefixBoundary.THROUGH_ACTION)
    contents = [event.payload["content"] for event in prefix]
    assert contents == ["first question", "first answer"]


def test_split_is_stable_disjoint_and_subset_independent() -> None:
    ids = [f"conversation-{index}" for index in range(500)]
    full = {cid: split_by_conversation_hash(cid, seed=1) for cid in ids}
    subset = {cid: split_by_conversation_hash(cid, seed=1) for cid in ids[:50]}
    assert all(full[cid] == subset[cid] for cid in ids[:50])

    counts = {"train": 0, "validation": 0, "test": 0}
    for split in full.values():
        counts[split] += 1
    assert counts["train"] > counts["validation"] > 0
    assert counts["test"] > 0

    reseeded = {cid: split_by_conversation_hash(cid, seed=2) for cid in ids}
    assert reseeded != full


def test_split_fraction_validation() -> None:
    with pytest.raises(ValueError):
        split_by_conversation_hash("x", seed=0, train_fraction=0.9, validation_fraction=0.2)
