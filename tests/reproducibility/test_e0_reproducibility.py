"""E0 scientific outputs must be stable under repeated seeded runs."""

from __future__ import annotations

from pathlib import Path

from longfeedback.config import E0Config
from longfeedback.experiments.e0 import run_e0


def _config(output_dir: Path) -> E0Config:
    return E0Config.model_validate(
        {
            "experiment": {
                "seed": 23,
                "episodes": 32,
                "horizon": 4,
                "train_fraction": 0.75,
                "output_dir": str(output_dir),
            },
            "oracle": {"mc_rollouts": 2, "max_examples": 128},
        }
    )


def test_repeated_runs_have_identical_scientific_outputs(tmp_path: Path) -> None:
    first = run_e0(_config(tmp_path / "first"))
    second = run_e0(_config(tmp_path / "second"))

    assert first.metrics["scientific_metrics_sha256"] == second.metrics["scientific_metrics_sha256"]
    assert (
        first.artifacts["predictions"].read_bytes() == second.artifacts["predictions"].read_bytes()
    )
    assert first.artifacts["plot"].read_bytes() == second.artifacts["plot"].read_bytes()
