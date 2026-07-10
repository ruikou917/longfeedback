"""End-to-end Gate B pipeline test at wiring scale (needs the research extra)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from longfeedback.config import GateBConfig
from longfeedback.experiments.gate_b import FAMILY_NAMES, run_gate_b


def _small_config(output_dir: Path) -> GateBConfig:
    return GateBConfig.model_validate(
        {
            "experiment": {"seed": 5, "train_fraction": 0.75, "output_dir": str(output_dir)},
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
        }
    )


def test_gate_b_pipeline_emits_decision_for_all_four_families(tmp_path: Path) -> None:
    result = run_gate_b(_small_config(tmp_path / "gate_b"))
    metrics = result.metrics

    assert set(metrics["families"]) == set(FAMILY_NAMES)
    decision = metrics["gate_b_decision"]
    assert set(decision) >= {
        "credit_recovery_across_families",
        "capacity_matched",
        "uncertainty_under_shift",
        "real_log_learnable",
        "pass",
    }
    # Within each family the three variants must be capacity-matched.
    assert decision["capacity_matched"] is True
    # Real-log criterion is skipped gracefully when E1 artifacts are absent.
    assert decision["real_log"]["available"] in (True, False)

    for family in metrics["families"].values():
        assert set(family["variants"]) == {"docm_outcome", "docm_prefix", "docm_credit"}
        ensemble = family["ensemble"]
        for block in ("in_distribution", "under_shift"):
            assert {"outcome_auroc", "credit_spearman", "uncertainty_error_spearman"} <= set(
                ensemble[block]
            )
    assert all(path.is_file() for path in result.artifacts.values())
