"""E12 smoke pipeline: two synchronous iterations on the fake environment."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from longfeedback.experiments.e12_alfworld_online import load_config, run_e12

_CONFIG = Path(__file__).resolve().parents[2] / (
    "configs/experiments/e12_alfworld_online_smoke.yaml"
)


def _test_config():
    config = load_config(_CONFIG)
    raw = config.model_dump(mode="json")
    raw["online"].update({"iterations": 2, "base_episodes_per_iteration": 4})
    raw["critic"].update({"epochs": 2})
    raw["environment"]["fake"].update(
        {"train_games": 4, "valid_seen_games": 2, "valid_unseen_games": 2}
    )
    return type(config).model_validate(raw)


def test_e12_smoke_pipeline(tmp_path: Path) -> None:
    result = run_e12(_test_config(), output_dir=tmp_path / "run")
    metrics = result.metrics
    decision = metrics["e12_decision"]

    assert decision["no_budget_violation"] is True
    assert decision["paper_ready_baselines_present"] is True
    per_seed = metrics["per_seed"]["0"]
    assert set(per_seed) == {
        "frozen_actor",
        "terminal_grpo",
        "prefix_group",
        "c3_group",
        "gigpo",
        "longfeedback_group",
    }

    frozen = per_seed["frozen_actor"]
    assert frozen["policy_ids"]["distinct"] == 1
    for method in ("terminal_grpo", "prefix_group", "gigpo", "longfeedback_group"):
        assert per_seed[method]["policy_ids"]["distinct"] > 1, method
    lf = per_seed["longfeedback_group"]
    assert lf["diagnostics"]["policy_centering_residual"] <= 1.0e-5
    assert lf["diagnostics"]["centering_aborts"] == 0
    assert lf["diagnostics"]["stale_rows_excluded"] == 0
    assert "gigpo_anchor" in per_seed["gigpo"]["diagnostics"]
    for method, seed_result in per_seed.items():
        assert 0.0 <= seed_result["locked_valid_unseen_success"] <= 1.0, method
        assert seed_result["ledger"]["replay_integrity_failures"] == 0

    for name in ("metrics", "learning_curves", "resolved_config", "manifest"):
        assert result.artifacts[name].is_file(), name


def test_e12_smoke_is_reproducible(tmp_path: Path) -> None:
    first = run_e12(_test_config(), output_dir=tmp_path / "a")
    second = run_e12(_test_config(), output_dir=tmp_path / "b")
    assert first.metrics["scientific_metrics_sha256"] == second.metrics["scientific_metrics_sha256"]
