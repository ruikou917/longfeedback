"""CLI smoke tests."""

from __future__ import annotations

from typer.testing import CliRunner

from longfeedback import __version__
from longfeedback.cli import app

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_unknown_experiment_is_rejected_before_execution() -> None:
    result = runner.invoke(app, ["experiment", "run", "unknown"])

    assert result.exit_code != 0
    assert "unknown experiment" in result.output
