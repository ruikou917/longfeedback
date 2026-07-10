"""Canonical event records."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import AwareDatetime, Field, JsonValue, field_validator

from longfeedback.schema.base import FrozenRecord


class EventType(StrEnum):
    OBSERVATION = "observation"
    ACTION = "action"
    USER_RESPONSE = "user_response"
    SYSTEM_EVENT = "system_event"
    OUTCOME = "outcome"


class Event(FrozenRecord):
    """One immutable, source-attributed event in a trajectory."""

    trajectory_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event_time: AwareDatetime
    step_index: int = Field(ge=0)
    event_type: EventType
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    source: str = Field(min_length=1)
    source_row_id: str | None = None
    policy_id: str | None = None
    policy_version: str | None = None
    schema_version: str = "0.1"

    @field_validator("event_time")
    @classmethod
    def normalize_event_time(cls, value: datetime) -> datetime:
        """Normalize every aware timestamp to UTC at the boundary."""

        return value.astimezone(UTC)
