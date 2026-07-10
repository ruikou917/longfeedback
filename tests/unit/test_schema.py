"""Tests for canonical event, trajectory, outcome, and credit contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from longfeedback.schema import (
    CensoringStatus,
    CreditContinuation,
    DelayedOutcomeExample,
    Event,
    EventType,
    OracleCreditExample,
    PrefixBoundary,
    Trajectory,
)


def _event(
    *,
    event_id: str,
    step: int,
    seconds: int,
    event_type: EventType,
) -> Event:
    return Event(
        trajectory_id="trajectory-1",
        event_id=event_id,
        event_time=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds),
        step_index=step,
        event_type=event_type,
        payload={"value": seconds},
        source="test",
    )


def test_event_timestamps_are_aware_and_normalized_to_utc() -> None:
    event = Event(
        trajectory_id="trajectory-1",
        event_id="event-1",
        event_time=datetime(2026, 1, 1, 2, tzinfo=timezone(timedelta(hours=2))),
        step_index=0,
        event_type=EventType.OBSERVATION,
        source="test",
    )

    assert event.event_time == datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError):
        Event(
            trajectory_id="trajectory-1",
            event_id="event-2",
            event_time=datetime(2026, 1, 1),
            step_index=0,
            event_type=EventType.OBSERVATION,
            source="test",
        )


def test_trajectory_rejects_unordered_events() -> None:
    later = _event(event_id="later", step=1, seconds=2, event_type=EventType.OBSERVATION)
    earlier = _event(event_id="earlier", step=0, seconds=1, event_type=EventType.OBSERVATION)

    with pytest.raises(ValidationError, match="ordered"):
        Trajectory(
            trajectory_id="trajectory-1",
            events=(later, earlier),
            start_time=datetime(2026, 1, 1, tzinfo=UTC),
            censoring_status=CensoringStatus.OBSERVED,
        )


def test_model_input_prefix_excludes_future_and_outcome_events() -> None:
    events = (
        _event(event_id="obs-0", step=0, seconds=0, event_type=EventType.OBSERVATION),
        _event(event_id="act-0", step=0, seconds=1, event_type=EventType.ACTION),
        _event(event_id="response-0", step=0, seconds=2, event_type=EventType.USER_RESPONSE),
        _event(event_id="obs-1", step=1, seconds=3, event_type=EventType.OBSERVATION),
        _event(event_id="out", step=1, seconds=4, event_type=EventType.OUTCOME),
    )
    trajectory = Trajectory(
        trajectory_id="trajectory-1",
        events=events,
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=4),
    )

    prefix = trajectory.model_input_events(0)

    assert [event.event_id for event in prefix] == ["obs-0", "act-0"]
    assert all(event.event_type is not EventType.OUTCOME for event in prefix)

    after_response = trajectory.model_input_events(
        0,
        boundary=PrefixBoundary.AFTER_RESPONSE,
    )
    assert [event.event_id for event in after_response] == [
        "obs-0",
        "act-0",
        "response-0",
    ]


def test_censored_example_cannot_be_silently_labeled_negative() -> None:
    with pytest.raises(ValidationError, match="censored"):
        DelayedOutcomeExample(
            trajectory_id="trajectory-1",
            prefix_end_step=0,
            observations=({},),
            actions=({},),
            responses=({},),
            terminal_outcome=0,
            outcome_type="future_failure",
            censored=True,
        )


def test_oracle_credit_requires_explicit_continuation_semantics() -> None:
    example = OracleCreditExample(
        trajectory_id="trajectory-1",
        step_index=2,
        action=1,
        reference_action=0,
        future_policy_id="behavior:v1",
        continuation_id="actions:sha256:abc123",
        continuation_mode=CreditContinuation.FROZEN,
        credit_utility=0.4,
        credit_proxy=0.0,
        monte_carlo_se=0.01,
    )

    assert example.continuation_mode is CreditContinuation.FROZEN

    with pytest.raises(ValidationError, match="future_policy_id"):
        OracleCreditExample(
            trajectory_id="trajectory-1",
            step_index=2,
            action=1,
            reference_action=0,
            continuation_id="policy:missing",
            continuation_mode=CreditContinuation.POLICY_REACTIVE,
            credit_utility=0.4,
            credit_proxy=0.0,
            monte_carlo_se=0.01,
        )
