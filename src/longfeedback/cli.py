"""Command-line interface for LongFeedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from longfeedback import __version__
from longfeedback.config import (
    E0Config,
    E1Config,
    E5Config,
    E6Config,
    GateAConfig,
    GateBConfig,
    KuaiRandDataConfig,
    LmsysDataConfig,
    WildChatDataConfig,
    load_e0_config,
    load_e1_config,
    load_e5_config,
    load_e6_config,
    load_gate_a_config,
    load_gate_b_config,
    load_kuairand_data_config,
    load_lmsys_data_config,
    load_wildchat_data_config,
)

app = typer.Typer(
    name="longfeedback",
    help="Research tools for learning from delayed behavioral outcomes.",
    no_args_is_help=True,
)
experiment_app = typer.Typer(help="Run reproducible research experiments.")
app.add_typer(experiment_app, name="experiment")
data_app = typer.Typer(help="Prepare local source datasets into canonical trajectories.")
app.add_typer(data_app, name="data")
report_app = typer.Typer(help="Generate paper-style reports from experiment artifacts.")
app.add_typer(report_app, name="report")


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
        typer.Argument(help="Experiment name: 'e0', 'e1', 'e5', 'e6', 'gate_a', or 'gate_b'."),
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
    elif experiment_name == "e1":
        try:
            from longfeedback.experiments.e1 import run_e1
        except ImportError as error:
            raise typer.BadParameter(
                "the e1 experiment needs the research extra; "
                "install it with `uv sync --extra research`",
                param_hint="name",
            ) from error

        e1_config = E1Config() if config_path is None else load_e1_config(config_path)
        result = run_e1(e1_config, output_dir=output_dir)
    elif experiment_name == "gate_b":
        try:
            from longfeedback.experiments.gate_b import run_gate_b
        except ImportError as error:
            raise typer.BadParameter(
                "the gate_b experiment needs the research extra; "
                "install it with `uv sync --extra research`",
                param_hint="name",
            ) from error

        gate_b_config = GateBConfig() if config_path is None else load_gate_b_config(config_path)
        result = run_gate_b(gate_b_config, output_dir=output_dir)
    elif experiment_name == "e5":
        try:
            from longfeedback.experiments.e5 import run_e5
        except ImportError as error:
            raise typer.BadParameter(
                "the e5 experiment needs the research extra; "
                "install it with `uv sync --extra research`",
                param_hint="name",
            ) from error

        e5_config = E5Config() if config_path is None else load_e5_config(config_path)
        result = run_e5(e5_config, output_dir=output_dir)
    elif experiment_name == "e6":
        try:
            from longfeedback.experiments.e6 import run_e6
        except ImportError as error:
            raise typer.BadParameter(
                "the e6 experiment needs the research extra; "
                "install it with `uv sync --extra research`",
                param_hint="name",
            ) from error

        e6_config = E6Config() if config_path is None else load_e6_config(config_path)
        result = run_e6(e6_config, output_dir=output_dir)
    else:
        raise typer.BadParameter(
            "unknown experiment; expected 'e0', 'e1', 'e5', 'e6', 'gate_a', or 'gate_b'",
            param_hint="name",
        )
    typer.echo(json.dumps(result.metrics, indent=2, sort_keys=True))


@data_app.command("prepare")
def prepare_data(
    source: Annotated[
        str,
        typer.Argument(help="Source dataset name; 'lmsys', 'wildchat', or 'kuairand'."),
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
            help="Validated YAML data-preparation configuration.",
        ),
    ] = None,
    input_dir: Annotated[
        Path | None,
        typer.Option(
            "--input-dir",
            file_okay=False,
            help="Override the configured source snapshot directory.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="Override the configured processed-data directory.",
        ),
    ] = None,
) -> None:
    """Prepare a local dataset snapshot and print its statistics as JSON."""

    source_name = source.lower()
    if source_name not in ("lmsys", "wildchat", "kuairand"):
        raise typer.BadParameter(
            "unknown source; expected 'lmsys', 'wildchat', or 'kuairand'", param_hint="source"
        )
    try:
        stats: dict[str, Any]
        if source_name == "lmsys":
            from longfeedback.data.lmsys import prepare_lmsys

            lmsys_config = (
                LmsysDataConfig() if config_path is None else load_lmsys_data_config(config_path)
            )
            stats = prepare_lmsys(lmsys_config, input_dir=input_dir, output_dir=output_dir).stats
        elif source_name == "wildchat":
            from longfeedback.data.wildchat import prepare_wildchat

            wildchat_config = (
                WildChatDataConfig()
                if config_path is None
                else load_wildchat_data_config(config_path)
            )
            stats = prepare_wildchat(
                wildchat_config, input_dir=input_dir, output_dir=output_dir
            ).stats
        else:
            from longfeedback.data.kuairand import prepare_kuairand

            kuairand_config = (
                KuaiRandDataConfig()
                if config_path is None
                else load_kuairand_data_config(config_path)
            )
            stats = prepare_kuairand(
                kuairand_config, input_dir=input_dir, output_dir=output_dir
            ).stats
    except ImportError as error:
        raise typer.BadParameter(
            "data preparation needs the research extra; install it with `uv sync --extra research`",
            param_hint="source",
        ) from error
    typer.echo(json.dumps(stats, indent=2, sort_keys=True))


@report_app.command("build")
def build_report(
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            dir_okay=False,
            help="Report destination; defaults to reports/v0_2_report.md.",
        ),
    ] = None,
) -> None:
    """Build the v0.2 report from experiment artifacts."""

    from longfeedback.experiments.report import write_v02_report

    repository = Path.cwd().resolve()
    for candidate in (repository, *repository.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "longfeedback"
        ).is_dir():
            repository = candidate
            break
    try:
        target = write_v02_report(repository, output_path)
    except FileNotFoundError as error:
        raise typer.BadParameter(str(error), param_hint="artifacts") from error
    typer.echo(f"wrote {target}")
