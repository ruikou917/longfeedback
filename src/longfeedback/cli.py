"""Command-line interface for LongFeedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from longfeedback import __version__
from longfeedback.config import E0Config, GateAConfig, load_e0_config, load_gate_a_config

app = typer.Typer(
    name="longfeedback",
    help="Research tools for learning from delayed behavioral outcomes.",
    no_args_is_help=True,
)
experiment_app = typer.Typer(help="Run reproducible research experiments.")
app.add_typer(experiment_app, name="experiment")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed LongFeedback version.",
    ),
) -> None:
    """Run LongFeedback commands."""

    del version


@experiment_app.command("run")
def run_experiment(
    name: Annotated[
        str,
        typer.Argument(help="Experiment name: 'e0' or 'gate_a'."),
    ],
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Validated YAML experiment configuration.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="Override the configured artifact directory.",
        ),
    ] = None,
) -> None:
    """Run a named experiment and print its metrics as JSON."""

    experiment_name = name.lower()
    if experiment_name == "e0":
        from longfeedback.experiments.e0 import run_e0

        e0_config = E0Config() if config_path is None else load_e0_config(config_path)
        result = run_e0(e0_config, output_dir=output_dir)
    elif experiment_name == "gate_a":
        try:
            from longfeedback.experiments.gate_a import run_gate_a
        except ImportError as error:
            raise typer.BadParameter(
                "the gate_a experiment needs the research extra; "
                "install it with `uv sync --extra research`",
                param_hint="name",
            ) from error

        gate_a_config = GateAConfig() if config_path is None else load_gate_a_config(config_path)
        result = run_gate_a(gate_a_config, output_dir=output_dir)
    else:
        raise typer.BadParameter("unknown experiment; expected 'e0' or 'gate_a'", param_hint="name")
    typer.echo(json.dumps(result.metrics, indent=2, sort_keys=True))
