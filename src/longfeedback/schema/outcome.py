"""Learning examples that keep outcomes, utility, and credit distinct."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import AwareDatetime, Field, JsonValue, field_validator, model_validator

from longfeedback.schema.base import FrozenRecord


class PropensityQuality(StrEnum):
    EXACT = "exact"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"
    CONFOUNDED = "confounded"


class CreditContinuation(StrEnum):
    FROZEN = "frozen"
    POLICY_REACTIVE = "policy_reactive"


class DelayedOutcomeExample(FrozenRecord):
    """A frozen prediction example; future evidence is label-only provenance."""

    trajectory_id: str = Field(min_length=1)
    prefix_end_step: int = Field(ge=0)
    observations: tuple[dict[str, JsonValue], ...]
    actions: tuple[dict[str, JsonValue], ...]
    responses: tuple[dict[str, JsonValue], ...]
    terminal_outcome: float | int | None
    outcome_type: str = Field(min_length=1)
    outcome_observed_at: AwareDatetime | None = None
    censored: bool = False
    behavior_logprobs: tuple[float, ...] | None = None
    propensity_quality: PropensityQuality = PropensityQuality.UNKNOWN
    sample_weight: float = Field(default=1.0, gt=0.0)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    schema_version: str = "0.1"

    @field_validator("outcome_observed_at")
    @classmethod
    def normalize_outcome_time(cls, value: datetime | None) -> datetime | None:
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def validate_outcome_state(self) -> DelayedOutcomeExample:
        if self.censored and self.terminal_outcome is not None:
            raise ValueError("a censored example cannot carry a finalized terminal outcome")
        if self.behavior_logprobs is not None and len(self.behavior_logprobs) != len(self.actions):
            raise ValueError("behavior_logprobs must align with actions")
        return self


class OracleCreditExample(FrozenRecord):
    """Intervention-grounded credit with explicit estimand semantics."""

    trajectory_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    action: int = Field(ge=0)
    reference_action: int = Field(ge=0)
    future_policy_id: str | None = None
    continuation_id: str = Field(min_length=1)
    continuation_mode: CreditContinuation
    credit_utility: float
    credit_proxy: float
    monte_carlo_se: float = Field(ge=0.0)
    schema_version: str = "0.1"

    @model_validator(mode="after")
    def validate_continuation_provenance(self) -> OracleCreditExample:
        if (
            self.continuation_mode is CreditContinuation.POLICY_REACTIVE
            and self.future_policy_id is None
        ):
            raise ValueError("policy-reactive credit requires future_policy_id")
        return self
