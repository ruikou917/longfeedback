"""E11 smoke pipeline on the fake environment (research extra required)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pyarrow")

from longfeedback.experiments.e11_alfworld_credit import load_config, run_e11

_CONFIG = Path(__file__).resolve().parents[2] / (
    "configs/experiments/e11_alfworld_credit_smoke.yaml"
)

_OVERRIDES = {
    "collection": {
        "base_episodes": 8,
        "calibration_episodes": 4,
        "reference_episodes": 4,
        "train_rollouts_per_action": 2,
        "unforced_rollouts": 3,
        "calibration_rollouts_per_action": 4,
        "reference_rollouts_per_action": 6,
        "replay_audit_prefixes": 24,
    },
    "training": {"seeds": [0], "epochs": 3},
    "uncertainty": {"members": 2},
    "decision": {"minimum_reference_states": 2, "minimum_positive_seeds": 1},
}


def _test_config():
    config = load_config(_CONFIG)
    raw = config.model_dump(mode="json")
    for section, values in _OVERRIDES.items():
        raw[section].update(values)
    return type(config).model_validate(raw)


def test_e11_smoke_pipeline(tmp_path: Path) -> None:
    result = run_e11(_test_config(), output_dir=tmp_path / "run")
    metrics = result.metrics

    signal = metrics["signal_gate"]
    assert signal["replay_match_rate"] == 1.0
    assert signal["split_overlap_free"] is True
    assert metrics["model"]["capacity_matched"] is True

    per_seed = metrics["per_seed"]["0"]["variants"]
    for variant in ("docm_dueling_credit", "docm_dueling_no_tree"):
        assert per_seed[variant]["policy_centering_residual"] <= 1.0e-5
    for name in (
        "episodes",
        "steps",
        "branch_targets",
        "state_policy_distributions",
        "predictions",
        "c3_direct_predictions",
        "gigpo_credit_diagnostics",
        "replay_audit",
        "budget_ledger",
        "resolved_config",
        "metrics",
        "manifest",
        "credit_calibration",
        "uncertainty_calibration",
    ):
        assert result.artifacts[name].is_file(), name

    coverage = metrics["uncertainty"]["coverage"]
    assert 0.0 <= coverage["empirical_coverage"] <= 1.0
    assert metrics["baselines"]["c3_states"] > 0


def test_e11_smoke_is_reproducible(tmp_path: Path) -> None:
    first = run_e11(_test_config(), output_dir=tmp_path / "a")
    second = run_e11(_test_config(), output_dir=tmp_path / "b")
    assert first.metrics["scientific_metrics_sha256"] == second.metrics["scientific_metrics_sha256"]
