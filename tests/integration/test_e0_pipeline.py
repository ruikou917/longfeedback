"""End-to-end E0 pipeline test."""

from __future__ import annotations

import json
from pathlib import Path

from longfeedback.config import E0Config
from longfeedback.experiments.e0 import run_e0


def _small_config(output_dir: Path, *, seed: int = 11) -> E0Config:
    return E0Config.model_validate(
        {
            "experiment": {
                "seed": seed,
                "episodes": 32,
                "horizon": 4,
                "train_fraction": 0.75,
                "output_dir": str(output_dir),
            },
            "oracle": {"mc_rollouts": 2, "max_examples": 128},
        }
    )


def test_e0_pipeline_passes_and_writes_auditable_artifacts(tmp_path: Path) -> None:
    result = run_e0(_small_config(tmp_path / "e0"))

    assert result.metrics["status"] == "pass"
    assert all(result.metrics["acceptance"].values())
    assert set(result.artifacts) == {"metrics", "predictions", "plot", "manifest"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in result.artifacts.values())

    persisted = json.loads(result.artifacts["metrics"].read_text(encoding="utf-8"))
    assert persisted["scientific_metrics_sha256"] == result.metrics["scientific_metrics_sha256"]
    manifest = json.loads(result.artifacts["manifest"].read_text(encoding="utf-8"))
    assert manifest["config_sha256"]
    assert manifest["config"]["experiment"]["seed"] == 11
