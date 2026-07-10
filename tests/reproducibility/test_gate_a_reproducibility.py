"""Gate A scientific outputs must be stable under repeated seeded runs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from longfeedback.config import GateAConfig
from longfeedback.experiments.gate_a import run_gate_a


def _config(output_dir: Path) -> GateAConfig:
    return GateAConfig.model_validate(
        {
            "experiment": {"seed": 29, "train_fraction": 0.75, "output_dir": str(output_dir)},
            "world_a": {"episodes": 24, "horizon": 5, "observabilities": ["partial"]},
            "world_b": {
                "episodes": 24,
                "horizon": 5,
                "regimes": ["clean"],
                "proxy_threshold": 2.5,
            },
            "oracle": {
                "initial_rollouts": 4,
                "max_rollouts": 8,
                "se_threshold": 0.5,
                "label_train_episodes": 10,
                "stability_examples": 12,
            },
            "model": {"d_model": 16, "n_layers": 1, "n_heads": 2},
            "training": {"epochs": 3, "batch_size": 16},
            "policy_check": {
                "regime": "world_a_partial",
                "evaluation_episodes": 16,
                "bc_epochs": 20,
            },
        }
    )


def test_repeated_runs_have_identical_scientific_outputs(tmp_path: Path) -> None:
    first = run_gate_a(_config(tmp_path / "first"))
    second = run_gate_a(_config(tmp_path / "second"))

    assert first.metrics["scientific_metrics_sha256"] == second.metrics["scientific_metrics_sha256"]
    assert (
        first.artifacts["predictions"].read_bytes() == second.artifacts["predictions"].read_bytes()
    )
