"""Shared validation behavior for canonical serialized records."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenRecord(BaseModel):
    """Strict, immutable boundary record with stable serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)
