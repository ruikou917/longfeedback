"""End-to-end multi-seed protocol test at wiring scale (needs the research extra)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("torch")

from longfeedback.config import MultiSeedConfig


def _small_config(output_dir: Path) -> MultiSeedConfig:
    return MultiSeedConfig.model_validate(
        {
            "seeds": [11, 12],
            "min_seeds": 5,
            "experiments": ["gate_b", "e5"],
            "output_dir": str(output_dir),
            "bootstrap": {"resamples": 200, "confidence": 0.9, "seed": 0},
            "gate_b": {
                "experiment": {"train_fraction": 0.75},
                "families": {"episodes": 20, "label_train_episodes": 8, "shift_episodes": 16},
                "oracle": {
                    "initial_rollouts": 4,
                    "max_rollouts": 8,
                    "se_threshold": 0.5,
                    "stability_examples": 8,
                },
                "model": {"d_model": 16, "n_layers": 1, "n_heads": 2},
                "training": {"epochs": 2, "batch_size": 16},
                "ensemble_members": 2,
                "decision": {"require_real_log": False},
            },
            "e5": {
                "logging_regimes": [{"name": "tiny", "episodes": 32, "behavior_epsilon": 0.3}],
                "evaluation_episodes": 16,
                "reward_model": {"d_model": 16, "epochs": 3, "ensemble_members": 2},
                "optimization": {
                    "updates": 8,
                    "checkpoint_every": 4,
                    "batch_episodes": 8,
                    "behavior_clone_epochs": 20,
                },
            },
        }
    )


def test_multiseed_pipeline_aggregates_both_experiments(tmp_path: Path) -> None:
    from longfeedback.experiments.multiseed import run_multiseed

    output_dir = tmp_path / "multiseed"
    result = run_multiseed(_small_config(output_dir))
    metrics = result.metrics

    # Per-seed runs land in their own auditable subdirectories.
    for experiment in ("gate_b", "e5"):
        for seed in (11, 12):
            assert (output_dir / experiment / f"seed_{seed}" / "metrics.json").is_file()
        assert set(metrics["per_seed_runs"][experiment]) == {"11", "12"}
        for run in metrics["per_seed_runs"][experiment].values():
            assert run["scientific_metrics_sha256"]

    # Aggregated tables carry per-seed values and bootstrap CIs.
    gate_b_table = metrics["tables"]["gate_b"]
    assert set(gate_b_table) == {"world_a", "world_b", "world_c", "world_d"}
    margin = gate_b_table["world_a"]["credit_margin"]
    assert len(margin["per_seed"]) == 2
    assert margin["ci_low"] <= margin["mean"] <= margin["ci_high"]
    e5_table = metrics["tables"]["e5"]
    assert set(e5_table) == {"tiny"}
    assert "kl_hacking_gap_reduction" in e5_table["tiny"]

    # Two seeds run fine but the >=5-seed protocol is recorded as unmet.
    decision = metrics["multiseed_decision"]
    assert decision["seed_count"] == 2
    assert decision["protocol_seed_count_met"] is False
    assert decision["pass"] is False
    assert set(decision["gate_b"]) >= {"per_family", "credit_recovery_robust"}
    assert set(decision["e5"]) >= {"per_regime", "goodhart_robust"}

    # The long-format seed table covers every (experiment, group, metric, seed).
    with result.artifacts["seed_table"].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["experiment"] for row in rows} == {"gate_b", "e5"}
    assert {row["seed"] for row in rows} == {"11", "12"}
    gate_b_rows = [row for row in rows if row["experiment"] == "gate_b"]
    assert len(gate_b_rows) == 2 * 4 * len(gate_b_table["world_a"])

    assert all(path.is_file() and path.stat().st_size > 0 for path in result.artifacts.values())
