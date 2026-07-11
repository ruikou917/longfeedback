"""End-to-end E5 pipeline test at wiring scale (needs the research extra)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from longfeedback.config import E5Config
from longfeedback.experiments.e5 import REWARD_VARIANTS, run_e5


def small_e5_config(output_dir: Path, *, seed: int = 3) -> E5Config:
    return E5Config.model_validate(
        {
            "seed": seed,
            "output_dir": str(output_dir),
            "logging_regimes": [{"name": "tiny", "episodes": 32, "behavior_epsilon": 0.3}],
            "evaluation_episodes": 16,
            "reward_model": {"d_model": 16, "epochs": 3, "ensemble_members": 2},
            "optimization": {
                "updates": 8,
                "checkpoint_every": 4,
                "batch_episodes": 8,
                "behavior_clone_epochs": 20,
            },
        }
    )


def test_e5_pipeline_emits_two_channel_decision_and_curves(tmp_path: Path) -> None:
    result = run_e5(small_e5_config(tmp_path / "e5"))
    metrics = result.metrics

    decision = metrics["e5_decision"]
    assert set(decision) >= {
        "goodhart_observed",
        "rm_error_channel_active",
        "lcb_mitigates",
        "kl_mitigates",
        "hypothesis_h5_lcb",
        "per_regime",
        "pass",
    }
    assert decision["hypothesis_h5_lcb"] in {
        "supported",
        "not_testable_rm_error_channel_inactive",
        "refuted_in_this_environment",
    }
    assert metrics["status"] == ("pass" if decision["pass"] else "fail")

    regime = metrics["regimes"]["tiny"]
    assert set(regime["variants"]) == set(REWARD_VARIANTS)
    for variant in regime["variants"].values():
        summary = variant["summary"]
        # Both failure channels are reported separately for every variant.
        assert "hacking_gap" in summary
        assert "rm_exploitation_gap" in summary
        # Checkpoint 0 (before any update) is always evaluated.
        assert variant["checkpoints"][0]["update"] == 0.0
        for point in variant["checkpoints"]:
            assert {
                "learned_reward",
                "observed_proxy",
                "true_utility",
                "ensemble_uncertainty",
                "kl_to_behavior_clone",
                "bait_action_fraction",
            } <= set(point)

    assert result.artifacts["plot_tiny"].is_file()
    assert all(path.is_file() and path.stat().st_size > 0 for path in result.artifacts.values())


def test_e5_checkpoint_zero_matches_across_variants(tmp_path: Path) -> None:
    """All variants start from the same behavior clone on paired eval seeds."""

    result = run_e5(small_e5_config(tmp_path / "e5"))
    starts = {
        variant: values["checkpoints"][0]["true_utility"]
        for variant, values in result.metrics["regimes"]["tiny"]["variants"].items()
    }
    assert len(set(starts.values())) == 1
