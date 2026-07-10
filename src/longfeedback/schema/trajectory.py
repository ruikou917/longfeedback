"""Canonical trajectory snapshots and leakage-safe prefixes."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import AwareDatetime, Field, JsonValue, field_validator, model_validator

from longfeedback.schema.base import FrozenRecord
from longfeedback.schema.event import Event, EventType


class ObservationRegime(StrEnum):
    ORACLE = "oracle"
    CLEAN = "clean"
    RANDOMIZED = "randomized"
    PARTIAL = "partial"
    HIDDEN_CONFOUNDING = "hidden_confounding"


class CensoringStatus(StrEnum):
    OBSERVED = "observed"
    RIGHT_CENSORED = "right_censored"
    AMBIGUOUS = "ambiguous"
    PENDING = "pending"


class PrefixBoundary(StrEnum):
    """Event-time boundary for a model-scoring prefix."""

    BEFORE_ACTION = "before_action"
    THROUGH_ACTION = "through_action"
    AFTER_RESPONSE = "after_response"


class Trajectory(FrozenRecord):
    """An immutable, validated view over ordered events."""

    trajectory_id: str = Field(min_length=1)
    entity_key_hash: str | None = None
    events: tuple[Event, ...]
    start_time: AwareDatetime
    end_time: AwareDatetime | None = None
    behavior_policy_id: str | None = None
    observation_regime: ObservationRegime = ObservationRegime.CLEAN
    censoring_status: CensoringStatus = CensoringStatus.OBSERVED
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    schema_version: str = "0.1"

    @field_validator("start_time", "end_time")
    @classmethod
    def normalize_time(cls, value: datetime | None) -> datetime | None:
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def validate_event_snapshot(self) -> Trajectory:
        if not self.events:
            raise ValueError("trajectory must contain at least one event")
        if any(event.trajectory_id != self.trajectory_id for event in self.events):
            raise ValueError("all events must belong to trajectory_id")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event IDs must be unique within a trajectory")
        ordered = sorted(
            self.events,
            key=lambda event: (event.event_time, event.step_index, event.event_id),
        )
        if list(self.events) != ordered:
            raise ValueError("events must be ordered by event time, step, and event ID")
        if self.events[0].event_time < self.start_time:
            raise ValueError("event occurs before trajectory start")
        if self.end_time is not None:
            if self.end_time < self.start_time:
                raise ValueError("end_time cannot precede start_time")
            if self.events[-1].event_time > self.end_time:
                raise ValueError("event occurs after trajectory end")
        return self

    def model_input_events(
        self,
        prefix_end_step: int,
        *,
        boundary: PrefixBoundary = PrefixBoundary.THROUGH_ACTION,
    ) -> tuple[Event, ...]:
        """Return a scoring prefix with an explicit within-step boundary.

        The default includes the candidate action at ``prefix_end_step`` but
        excludes the user response produced after that action.
        """

        if prefix_end_step < 0:
            raise ValueError("prefix_end_step must be non-negative")
        allowed_at_boundary = {
            PrefixBoundary.BEFORE_ACTION: {
                EventType.OBSERVATION,
                EventType.SYSTEM_EVENT,
            },
            PrefixBoundary.THROUGH_ACTION: {
                EventType.OBSERVATION,
                EventType.SYSTEM_EVENT,
                EventType.ACTION,
            },
            PrefixBoundary.AFTER_RESPONSE: {
                EventType.OBSERVATION,
                EventType.SYSTEM_EVENT,
                EventType.ACTION,
                EventType.USER_RESPONSE,
            },
        }[boundary]
        return tuple(
            event
            for event in self.events
            if event.event_type is not EventType.OUTCOME
            and (
                event.step_index < prefix_end_step
                or (event.step_index == prefix_end_step and event.event_type in allowed_at_boundary)
            )
        )
