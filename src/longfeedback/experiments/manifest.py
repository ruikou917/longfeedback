"""Reproducibility metadata for experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from longfeedback import __version__


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data using its canonical representation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _git_value(repository: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def build_run_manifest(
    *,
    repository: Path,
    resolved_config: dict[str, Any],
    artifacts: dict[str, str],
) -> dict[str, Any]:
    """Build an auditable manifest without introducing nondeterministic metrics."""

    status = _git_value(repository, "status", "--porcelain")
    return {
        "project": "longfeedback",
        "project_version": __version__,
        "config": resolved_config,
        "config_sha256": sha256_json(resolved_config),
        "git_commit": _git_value(repository, "rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "artifacts": artifacts,
    }
