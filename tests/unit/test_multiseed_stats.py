"""Unit tests for the multi-seed aggregation layer (torch-free)."""

from __future__ import annotations

from typing import Any

import numpy as np

from longfeedback.config import (
    E5DecisionSettings,
    GateBDecisionSettings,
    MultiSeedBootstrapSettings,
)
from longfeedback.experiments.multiseed import (
    aggregate_seed_tables,
    bootstrap_ci,
    e5_robustness,
    e5_seed_table,
    gate_b_robustness,
    gate_b_seed_table,
    summarize_metric,
)

_SETTINGS = MultiSeedBootstrapSettings(resamples=2000, confidence=0.9, seed=0)


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean() -> None:
    values = [0.1, 0.3, 0.2, 0.4, 0.25]
    first = bootstrap_ci(values, resamples=2000, confidence=0.9, rng=np.random.default_rng(7))
    second = bootstrap_ci(values, resamples=2000, confidence=0.9, rng=np.random.default_rng(7))
    assert first == second
    low, high = first
    assert low <= float(np.mean(values)) <= high
    assert min(values) <= low and high <= max(values)


def test_bootstrap_ci_degenerates_to_a_point_for_constant_values() -> None:
    low, high = bootstrap_ci(
        [0.5, 0.5, 0.5], resamples=500, confidence=0.95, rng=np.random.default_rng(0)
    )
    assert low == high == 0.5


def test_summarize_metric_reports_moments_and_per_seed_values() -> None:
    values = [1.0, 2.0, 3.0]
    summary = summarize_metric(values, _SETTINGS, np.random.default_rng(0))
    assert summary["per_seed"] == values
    assert summary["mean"] == 2.0
    assert summary["min"] == 1.0 and summary["max"] == 3.0
    assert summary["std"] == 1.0
    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]


def _gate_b_metrics(margin: float) -> dict[str, Any]:
    """Minimal Gate B metrics with one family and a chosen credit margin."""

    return {
        "families": {
            "world_a": {
                "variants": {
                    "docm_credit": {
                        "credit": {"spearman": 0.5 + margin},
                        "outcome": {"auroc": 0.8},
                    },
                    "docm_outcome": {
                        "credit": {"spearman": 0.5},
                        "outcome": {"auroc": 0.82},
                    },
                    "docm_prefix": {
                        "credit": {"spearman": 0.45},
                        "outcome": {"auroc": 0.79},
                    },
                },
                "ensemble": {
                    "under_shift": {
                        "uncertainty_error_spearman": 0.4,
                        "error_detection_auroc": 0.7,
                    }
                },
            }
        },
        "gate_b_decision": {
            "transfer": {"per_family": {"world_a": {"transfer_balanced_accuracy": 0.8}}}
        },
    }


def test_gate_b_seed_table_extracts_the_paired_credit_margin() -> None:
    table = gate_b_seed_table(_gate_b_metrics(0.12))
    row = table["world_a"]
    assert row["credit_margin"] == 0.12
    assert row["credit_spearman_docm_credit"] == 0.62
    assert row["shift_error_detection_auroc"] == 0.7
    assert row["transfer_balanced_accuracy"] == 0.8


def _e5_metrics(single_gap: float, kl_gap: float) -> dict[str, Any]:
    """Minimal E5 metrics with one regime and chosen hacking gaps."""

    def variant(gap: float, utility: float) -> dict[str, Any]:
        return {
            "summary": {
                "hacking_gap": gap,
                "utility_at_proxy_optimal": utility,
                "rm_exploitation_gap": 0.1,
            }
        }

    return {
        "regimes": {
            "broad": {
                "variants": {
                    "single": variant(single_gap, 1.0),
                    "ensemble_mean": variant(single_gap, 1.1),
                    "ensemble_lcb": variant(single_gap, 1.0),
                    "single_kl": variant(kl_gap, 2.0),
                }
            }
        }
    }


def test_e5_seed_table_derives_paired_mitigation_deltas() -> None:
    table = e5_seed_table(_e5_metrics(3.0, 0.5))
    row = table["broad"]
    assert row["hacking_gap_single"] == 3.0
    assert row["kl_hacking_gap_reduction"] == 2.5
    assert row["kl_utility_gain"] == 1.0
    # LCB shares the single gap here, so its paired deltas are exactly zero.
    assert row["lcb_hacking_gap_reduction"] == 0.0
    assert row["lcb_utility_gain"] == 0.0


def test_gate_b_robustness_requires_a_ci_excluding_zero() -> None:
    consistent = [gate_b_seed_table(_gate_b_metrics(m)) for m in (0.10, 0.12, 0.11, 0.13, 0.09)]
    aggregated = aggregate_seed_tables(consistent, _SETTINGS)
    verdicts = gate_b_robustness(aggregated, GateBDecisionSettings(min_winning_families=1))
    assert verdicts["per_family"]["world_a"]["credit_margin_robust"] is True
    assert verdicts["credit_recovery_robust"] is True
    assert verdicts["uncertainty_under_shift_robust"] is True

    sign_flipping = [gate_b_seed_table(_gate_b_metrics(m)) for m in (0.2, -0.2, 0.2, -0.2, 0.02)]
    aggregated = aggregate_seed_tables(sign_flipping, _SETTINGS)
    verdicts = gate_b_robustness(aggregated, GateBDecisionSettings(min_winning_families=1))
    assert verdicts["per_family"]["world_a"]["credit_margin_robust"] is False
    assert verdicts["credit_recovery_robust"] is False


def test_e5_robustness_separates_goodhart_from_mitigation_verdicts() -> None:
    tables = [e5_seed_table(_e5_metrics(g, 0.5)) for g in (2.8, 3.0, 3.2, 2.9, 3.1)]
    aggregated = aggregate_seed_tables(tables, _SETTINGS)
    verdicts = e5_robustness(aggregated, E5DecisionSettings())
    assert verdicts["goodhart_robust"] is True
    assert verdicts["kl_mitigation_robust"] is True
    # LCB never separates from single in these fixtures.
    assert verdicts["lcb_mitigation_robust"] is False

    inconsistent = [e5_seed_table(_e5_metrics(g, 0.5)) for g in (3.0, -3.0, 3.0, -3.0, 0.1)]
    aggregated = aggregate_seed_tables(inconsistent, _SETTINGS)
    verdicts = e5_robustness(aggregated, E5DecisionSettings())
    assert verdicts["goodhart_robust"] is False
