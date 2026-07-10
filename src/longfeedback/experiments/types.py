"""Shared experiment result types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentResult:
    """Summary and artifact locations for a completed experiment."""

    metrics: dict[str, Any]
    output_dir: Path
    artifacts: dict[str, Path]
