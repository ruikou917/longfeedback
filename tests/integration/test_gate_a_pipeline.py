"""End-to-end Gate A pipeline test at wiring scale (needs the research extra)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from longfeedback.config import GateAConfig
from longfeedback.experiments.gate_a import run_gate_a


def small_gate_a_config(output_dir: Path, *, seed: int = 7) -> GateAConfig:
    """A tiny configuration that exercises every Gate A component."""

    return GateAConfig.model_validate(
        {
            "experiment": {
                "seed": seed,
                "train_fraction": 0.75,
                "output_dir": str(output_dir),
            },
            "world_a": {"episodes": 24, "horizon": 5, "observabilities": ["partial"]},
            "world_b": {
                "episodes": 24,
                "horizon": 5,
                "regimes": ["clean", "confounded"],
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


def test_gate_a_pipeline_emits_auditable_artifacts(tmp_path: Path) -> None:
    result = run_gate_a(small_gate_a_config(tmp_path / "gate_a"))
    metrics: dict[str, Any] = result.metrics

    assert set(result.artifacts) == {"metrics", "predictions", "plot", "manifest"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in result.artifacts.values())

    decision = metrics["gate_a_decision"]
    assert set(decision) == {
        "outcome_credit_gap",
        "oracle_stability",
        "policy_improvement",
        "gap_details",
        "pass",
    }
    assert metrics["status"] == ("pass" if decision["pass"] else "fail")

    # Every configured regime is generated, labeled, and evaluated with all
    # three capacity-matched variants.
    expected_regimes = {"world_a_partial", "world_b_clean", "world_b_confounded"}
    assert set(metrics["regimes"]) == expected_regimes
    assert set(decision["gap_details"]) == expected_regimes
    assert metrics["model"]["capacity_matched"] is True
    for regime in metrics["regimes"].values():
        assert set(regime) == {"docm_outcome", "docm_prefix", "docm_credit"}
        for variant in regime.values():
            assert {"auroc", "brier", "ece"} <= set(variant["outcome"])
            assert {"pearson", "spearman", "kendall_tau", "sign_accuracy"} <= set(variant["credit"])

    for name, data in metrics["data"].items():
        assert data["oracle"]["labeled_steps"] > 0
        assert data["oracle"]["max_monte_carlo_se"] >= 0.0
        if name == "world_b_confounded":
            assert data["propensity_quality"] == "confounded"
            assert data["observation_regime"] == "hidden_confounding"

    persisted = json.loads(result.artifacts["metrics"].read_text(encoding="utf-8"))
    assert persisted["scientific_metrics_sha256"] == metrics["scientific_metrics_sha256"]
    manifest = json.loads(result.artifacts["manifest"].read_text(encoding="utf-8"))
    assert manifest["config"]["experiment"]["seed"] == 7

    policy_check = metrics["policy_check"]
    assert policy_check["regime"] == "world_a_partial"
    for key in ("behavior", "behavior_clone", "greedy_q"):
        assert set(policy_check[key]) == {"utility", "proxy"}


def test_gate_a_rejects_unknown_policy_check_regime(tmp_path: Path) -> None:
    config = small_gate_a_config(tmp_path / "gate_a")
    broken = config.model_copy(
        update={
            "policy_check": config.policy_check.model_copy(update={"regime": "world_c"}),
        }
    )
    with pytest.raises(ValueError, match="policy_check regime"):
        run_gate_a(broken)
