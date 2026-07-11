"""E5 scientific outputs must be stable under repeated seeded runs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from longfeedback.config import E5Config
from longfeedback.experiments.e5 import run_e5


def _config(output_dir: Path) -> E5Config:
    return E5Config.model_validate(
        {
            "seed": 17,
            "output_dir": str(output_dir),
            "logging_regimes": [{"name": "tiny", "episodes": 32, "behavior_epsilon": 0.3}],
            "evaluation_episodes": 16,
            "reward_model": {"d_model": 16, "epochs": 3, "ensemble_members": 2},
            "optimization": {
                "updates": 6,
                "checkpoint_every": 3,
                "batch_episodes": 8,
                "behavior_clone_epochs": 20,
            },
        }
    )


def test_repeated_runs_have_identical_scientific_outputs(tmp_path: Path) -> None:
    first = run_e5(_config(tmp_path / "first"))
    second = run_e5(_config(tmp_path / "second"))

    assert first.metrics["scientific_metrics_sha256"] == second.metrics["scientific_metrics_sha256"]
    assert (
        first.artifacts["predictions"].read_bytes() == second.artifacts["predictions"].read_bytes()
    )
