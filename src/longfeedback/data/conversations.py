"""Source-generic conversion of conversations into canonical trajectories.

This module is deliberately dependency-light (no pyarrow, no torch) so the
leakage-safe conversion and split logic is exercised by the torch-free test
path. Source-specific readers live next to it (for example ``lmsys.py``).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from longfeedback.schema import Event, EventType, ObservationRegime, Trajectory

PII_FILTER_VERSION = "regex-v1"

_SYNTHETIC_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)

# High-precision patterns only: a missed redaction is worse than an over-wide
# one here, but false positives corrupt text used for modeling, so each
# pattern targets an unambiguous shape.
_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (
        re.compile(
            r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3}[ .-]\d{3,4}[ .-]\d{4}(?!\d)"
        ),
        "[PHONE]",
    ),
    (re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"), "[IP]"),
)

SplitName = Literal["train", "validation", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    """One source conversation before canonical conversion."""

    conversation_id: str
    turns: tuple[ConversationTurn, ...]
    assistant_model: str | None
    language: str | None
    flagged: bool
    redacted: bool
    source: str
    source_row_id: str


def sanitize_text(text: str) -> tuple[str, int]:
    """Apply the local high-precision PII pass; return text and hit count."""

    replacements = 0
    for pattern, token in _PII_PATTERNS:
        text, count = pattern.subn(token, text)
        replacements += count
    return text, replacements


def paired_turns(record: ConversationRecord) -> tuple[tuple[ConversationTurn, ...], int]:
    """Return strictly alternating user/assistant pairs and dropped-turn count.

    A single trailing user message with no assistant reply is dropped (and
    counted); any other role-order violation makes the conversation invalid
    and is reported by :func:`conversation_exclusion_reason`.
    """

    turns = record.turns
    dropped = 0
    if turns and len(turns) % 2 == 1 and turns[-1].role == "user":
        turns = turns[:-1]
        dropped = 1
    return turns, dropped


def conversation_exclusion_reason(
    record: ConversationRecord,
    *,
    language: str | None,
    min_assistant_turns: int,
    max_message_chars: int,
    exclude_flagged: bool,
    include_redacted: bool,
) -> str | None:
    """Return an exclusion reason, or None when the conversation is kept."""

    if exclude_flagged and record.flagged:
        return "moderation_flagged"
    if not include_redacted and record.redacted:
        return "redacted"
    if language is not None and record.language != language:
        return "language"
    turns, _ = paired_turns(record)
    if not turns:
        return "empty"
    if len(turns) % 2 != 0:
        return "role_alternation"
    for index, turn in enumerate(turns):
        expected = "user" if index % 2 == 0 else "assistant"
        if turn.role != expected:
            return "role_alternation"
        if not turn.content.strip():
            return "empty_message"
        if len(turn.content) > max_message_chars:
            return "message_too_long"
    if len(turns) // 2 < min_assistant_turns:
        return "too_few_assistant_turns"
    return None


def conversation_to_trajectory(
    record: ConversationRecord,
    *,
    conversation_index: int,
) -> tuple[Trajectory, int]:
    """Convert a kept conversation into a canonical trajectory.

    Step ``t`` holds one OBSERVATION event (user message ``t``) followed by one
    ACTION event (assistant reply ``t``); every message appears exactly once,
    so ``Trajectory.model_input_events(t, THROUGH_ACTION)`` is exactly the
    scoring prefix H_t. Event times are synthetic (the source has none) and are
    flagged as such in the trajectory metadata.

    Returns the trajectory and the number of PII redactions applied.
    """

    turns, _ = paired_turns(record)
    if not turns or len(turns) % 2 != 0:
        raise ValueError("conversation_to_trajectory requires validated paired turns")

    trajectory_id = f"{record.source}:{record.conversation_id}"
    base_time = _SYNTHETIC_BASE_TIME + timedelta(minutes=conversation_index)
    events: list[Event] = []
    redactions = 0
    for message_index, turn in enumerate(turns):
        step_index = message_index // 2
        event_type = EventType.OBSERVATION if turn.role == "user" else EventType.ACTION
        content, hits = sanitize_text(turn.content)
        redactions += hits
        events.append(
            Event(
                trajectory_id=trajectory_id,
                event_id=f"{trajectory_id}:{message_index:04d}",
                event_time=base_time + timedelta(seconds=message_index),
                step_index=step_index,
                event_type=event_type,
                payload={"role": turn.role, "content": content},
                source=record.source,
                source_row_id=record.source_row_id,
                policy_id=record.assistant_model,
                policy_version=None,
            )
        )

    trajectory = Trajectory(
        trajectory_id=trajectory_id,
        events=tuple(events),
        start_time=base_time,
        end_time=events[-1].event_time,
        behavior_policy_id=record.assistant_model,
        observation_regime=ObservationRegime.CLEAN,
        metadata={
            "source": record.source,
            "assistant_model": record.assistant_model,
            "language": record.language,
            "redacted_by_source": record.redacted,
            "synthetic_time": True,
            "pii_filter_version": PII_FILTER_VERSION,
            "pii_redactions": redactions,
        },
    )
    return trajectory, redactions


def split_by_conversation_hash(
    conversation_id: str,
    *,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> SplitName:
    """Assign a stable split from the conversation identity alone.

    Membership depends only on ``(conversation_id, seed)``, never on scan
    order or subset size, and a conversation can never straddle splits.
    """

    if not 0.0 < train_fraction < 1.0 or not 0.0 <= validation_fraction < 1.0:
        raise ValueError("fractions must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave room for test")
    digest = hashlib.sha256(f"{seed}:{conversation_id}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < train_fraction:
        return "train"
    if position < train_fraction + validation_fraction:
        return "validation"
    return "test"
