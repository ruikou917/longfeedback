"""Auxiliary semantic hindsight label schema (design section 9.8).

Labels are representation-shaping auxiliaries only. They never replace branch
Q or direct V targets, never touch locked-reference selection, and are only
produced for training trajectories. This module defines the schema and its
validation; no teacher is implemented in the CPU milestone.
"""

from __future__ import annotations

from dataclasses import dataclass

PROGRESS_LABELS = ("progress", "recoverable_detour", "persistent_error")
LABEL_SCHEMA_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class HindsightLabel:
    game_id: str
    episode_id: str
    step_index: int
    state_hash: str
    action: str
    teacher_id: str
    prompt_hash: str
    label_schema_version: str
    progress_label: str
    error_type: str
    confidence: float
    rationale_hash: str
    target_role: str

    def __post_init__(self) -> None:
        if self.progress_label not in PROGRESS_LABELS:
            raise ValueError(f"progress_label must be one of {PROGRESS_LABELS}")
        if self.label_schema_version != LABEL_SCHEMA_VERSION:
            raise ValueError(f"unsupported label schema {self.label_schema_version!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.target_role != "train":
            raise ValueError(
                "hindsight labels may only exist for training trajectories; "
                f"got target_role={self.target_role!r}"
            )
