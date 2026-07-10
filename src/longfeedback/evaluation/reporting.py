"""Deterministic metric serialization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_metrics_json(metrics: Mapping[str, Any], path: str | Path) -> Path:
    """Write strict, key-sorted JSON and return the output path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        _jsonable(metrics),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    output.write_text(serialized + "\n", encoding="utf-8")
    return output
