"""Tests for validated E0 configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from longfeedback.config import E0Config, ReportSettings, load_e0_config


def test_default_config_is_valid() -> None:
    config = E0Config()

    assert config.experiment.name == "e0"
    assert config.experiment.horizon == 8
    assert config.oracle.continuation_mode == "frozen"


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("experiment:\n  typo: 1\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_e0_config(path)


def test_report_filenames_cannot_escape_output_directory() -> None:
    with pytest.raises(ValidationError):
        ReportSettings(metrics_filename="../metrics.json")
