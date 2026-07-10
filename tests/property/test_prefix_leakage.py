"""Property tests for the no-future-information invariant."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st

from longfeedback.schema import Event, EventType, Trajectory


@given(
    horizon=st.integers(min_value=2, max_value=20),
    prefix=st.integers(min_value=0, max_value=19),
)
def test_model_prefix_never_contains_future_steps(horizon: int, prefix: int) -> None:
    prefix = min(prefix, horizon - 1)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    events_list = []
    for step in range(horizon):
        for offset, event_type, suffix in (
            (0, EventType.OBSERVATION, "observation"),
            (1, EventType.ACTION, "action"),
            (2, EventType.USER_RESPONSE, "response"),
        ):
            events_list.append(
                Event(
                    trajectory_id="trajectory",
                    event_id=f"event-{step:03d}-{suffix}",
                    event_time=start + timedelta(seconds=3 * step + offset),
                    step_index=step,
                    event_type=event_type,
                    source="property-test",
                )
            )
    events = (
        *events_list,
        Event(
            trajectory_id="trajectory",
            event_id="outcome",
            event_time=start + timedelta(seconds=3 * horizon),
            step_index=horizon,
            event_type=EventType.OUTCOME,
            source="property-test",
        ),
    )
    trajectory = Trajectory(
        trajectory_id="trajectory",
        events=events,
        start_time=start,
        end_time=start + timedelta(seconds=3 * horizon),
    )

    model_events = trajectory.model_input_events(prefix)

    assert all(event.step_index <= prefix for event in model_events)
    assert all(event.event_type is not EventType.OUTCOME for event in model_events)
    assert all(
        event.step_index < prefix or event.event_type is not EventType.USER_RESPONSE
        for event in model_events
    )
