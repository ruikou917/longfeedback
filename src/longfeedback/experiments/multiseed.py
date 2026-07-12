"""Multi-seed statistical protocol for the primary Gate B and E5 tables.

Design doc 13.5 requires at least five seeds for small-model experiments,
paired evaluation across policies, bootstrap confidence intervals, a
predeclared primary metric, and effect sizes rather than bare p-values. This
experiment reruns the unmodified Gate B and E5 pipelines once per seed and
aggregates their tables accordingly:

- **Predeclared primary metrics.** Gate B: the per-family credit margin
  (``docm_credit`` credit Spearman minus the best of ``docm_outcome`` /
  ``docm_prefix``). E5: the per-regime hacking gap of the ``single`` reward
  (Goodhart effect size) plus the paired mitigation deltas for KL and LCB
  (hacking-gap reduction and utility gain at the proxy-optimal checkpoint).
  Everything else in the tables is secondary and reported without a
  robustness verdict of its own.
- **Pairing.** Within one seed, all variants of a run already share
  evaluation seeds, so differences between variants are paired; the bootstrap
  resamples per-seed paired differences, never unpaired marginals.
- **Bootstrap.** Percentile bootstrap over seeds with a fixed resampling
  seed, so the reported intervals are themselves reproducible. Effect sizes
  stay in native units (Spearman points, utility points).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from longfeedback.config import (
    E5Config,
    E5DecisionSettings,
    GateBConfig,
    GateBDecisionSettings,
    MultiSeedBootstrapSettings,
    MultiSeedConfig,
    dump_resolved_config,
)
from longfeedback.evaluation import write_metrics_json
from longfeedback.experiments.manifest import build_run_manifest, sha256_json
from longfeedback.experiments.types import ExperimentResult

# family/regime -> metric name -> value, extracted from one seed's run.
SeedTable = dict[str, dict[str, float]]
# family/regime -> metric name -> summary statistics across seeds.
AggregateTable = dict[str, dict[str, dict[str, Any]]]

E5_VARIANTS: tuple[str, ...] = ("single", "ensemble_mean", "ensemble_lcb", "single_kl")
GATE_B_VARIANTS: tuple[str, ...] = ("docm_credit", "docm_outcome", "docm_prefix")


def bootstrap_ci(
    values: list[float], *, resamples: int, confidence: float, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for the mean of ``values``."""

    sample = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(sample), size=(resamples, len(sample)))
    means = sample[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def summarize_metric(
    values: list[float], settings: MultiSeedBootstrapSettings, rng: np.random.Generator
) -> dict[str, Any]:
    """Across-seed summary: per-seed values, moments, and a bootstrap CI."""

    sample = np.asarray(values, dtype=np.float64)
    ci_low, ci_high = bootstrap_ci(
        values, resamples=settings.resamples, confidence=settings.confidence, rng=rng
    )
    return {
        "per_seed": [float(value) for value in sample],
        "mean": float(np.mean(sample)),
        "std": float(np.std(sample, ddof=1)) if len(sample) > 1 else 0.0,
        "min": float(np.min(sample)),
        "max": float(np.max(sample)),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def gate_b_seed_table(metrics: dict[str, Any]) -> SeedTable:
    """Extract the primary-table rows from one Gate B run's metrics."""

    table: SeedTable = {}
    transfer = metrics["gate_b_decision"]["transfer"]["per_family"]
    for family, block in metrics["families"].items():
        row: dict[str, float] = {}
        for variant in GATE_B_VARIANTS:
            values = block["variants"][variant]
            row[f"credit_spearman_{variant}"] = float(values["credit"]["spearman"])
            row[f"outcome_auroc_{variant}"] = float(values["outcome"]["auroc"])
        row["credit_margin"] = row["credit_spearman_docm_credit"] - max(
            row["credit_spearman_docm_outcome"], row["credit_spearman_docm_prefix"]
        )
        under_shift = block["ensemble"]["under_shift"]
        row["shift_uncertainty_error_spearman"] = float(under_shift["uncertainty_error_spearman"])
        row["shift_error_detection_auroc"] = float(under_shift["error_detection_auroc"])
        row["transfer_balanced_accuracy"] = float(transfer[family]["transfer_balanced_accuracy"])
        table[family] = row
    return table


def e5_seed_table(metrics: dict[str, Any]) -> SeedTable:
    """Extract the primary-table rows from one E5 run's metrics."""

    table: SeedTable = {}
    for regime, block in metrics["regimes"].items():
        row: dict[str, float] = {}
        for variant in E5_VARIANTS:
            summary = block["variants"][variant]["summary"]
            row[f"hacking_gap_{variant}"] = float(summary["hacking_gap"])
            row[f"utility_at_proxy_optimal_{variant}"] = float(summary["utility_at_proxy_optimal"])
            row[f"rm_exploitation_gap_{variant}"] = float(summary["rm_exploitation_gap"])
        for label, variant in (("kl", "single_kl"), ("lcb", "ensemble_lcb")):
            row[f"{label}_hacking_gap_reduction"] = (
                row["hacking_gap_single"] - row[f"hacking_gap_{variant}"]
            )
            row[f"{label}_utility_gain"] = (
                row[f"utility_at_proxy_optimal_{variant}"] - row["utility_at_proxy_optimal_single"]
            )
        table[regime] = row
    return table


def aggregate_seed_tables(
    tables: list[SeedTable], settings: MultiSeedBootstrapSettings
) -> AggregateTable:
    """Combine per-seed tables into across-seed summaries with bootstrap CIs.

    Groups and metrics are iterated in sorted order with one fixed-seed
    generator, so the reported intervals are deterministic for a given
    configuration.
    """

    rng = np.random.default_rng(settings.seed)
    first = tables[0]
    aggregated: AggregateTable = {}
    for group in sorted(first):
        aggregated[group] = {}
        for metric in sorted(first[group]):
            values = [table[group][metric] for table in tables]
            aggregated[group][metric] = summarize_metric(values, settings, rng)
    return aggregated


def _robust_positive(summary: dict[str, Any], *, mean_at_least: float) -> bool:
    """CI excludes zero in the beneficial direction and the mean meets the
    single-seed decision threshold."""

    return bool(summary["ci_low"] > 0.0 and summary["mean"] >= mean_at_least)


def gate_b_robustness(
    aggregated: AggregateTable, decision: GateBDecisionSettings
) -> dict[str, Any]:
    """Multi-seed verdicts for Gate B's primary comparisons."""

    per_family: dict[str, Any] = {}
    robust_credit_families = 0
    robust_uncertainty_families: list[str] = []
    for family, row in aggregated.items():
        credit_robust = _robust_positive(
            row["credit_margin"], mean_at_least=decision.credit_spearman_margin
        )
        uncertainty_robust = bool(
            row["shift_uncertainty_error_spearman"]["ci_low"] > 0.0
            and row["shift_uncertainty_error_spearman"]["mean"]
            >= decision.uncertainty_spearman_threshold
            and row["shift_error_detection_auroc"]["ci_low"] > 0.5
            and row["shift_error_detection_auroc"]["mean"]
            >= decision.error_detection_auroc_threshold
        )
        robust_credit_families += int(credit_robust)
        if uncertainty_robust:
            robust_uncertainty_families.append(family)
        per_family[family] = {
            "credit_margin_robust": credit_robust,
            "uncertainty_under_shift_robust": uncertainty_robust,
        }
    return {
        "per_family": per_family,
        "robust_credit_families": robust_credit_families,
        "credit_recovery_robust": bool(robust_credit_families >= decision.min_winning_families),
        "uncertainty_robust_families": robust_uncertainty_families,
        "uncertainty_under_shift_robust": bool(robust_uncertainty_families),
    }


def e5_robustness(aggregated: AggregateTable, decision: E5DecisionSettings) -> dict[str, Any]:
    """Multi-seed verdicts for E5's primary comparisons."""

    per_regime: dict[str, Any] = {}
    for regime, row in aggregated.items():
        goodhart_robust = _robust_positive(
            row["hacking_gap_single"], mean_at_least=decision.hacking_gap_threshold
        )
        mitigation: dict[str, bool] = {}
        for label in ("kl", "lcb"):
            mitigation[f"{label}_mitigation_robust"] = bool(
                _robust_positive(
                    row[f"{label}_hacking_gap_reduction"],
                    mean_at_least=decision.mitigation_margin,
                )
                and _robust_positive(
                    row[f"{label}_utility_gain"],
                    mean_at_least=decision.mitigation_margin,
                )
            )
        per_regime[regime] = {"goodhart_robust": goodhart_robust, **mitigation}
    return {
        "per_regime": per_regime,
        "goodhart_robust": any(block["goodhart_robust"] for block in per_regime.values()),
        "kl_mitigation_robust": any(block["kl_mitigation_robust"] for block in per_regime.values()),
        "lcb_mitigation_robust": any(
            block["lcb_mitigation_robust"] for block in per_regime.values()
        ),
    }


def _repository_root() -> Path:
    working_directory = Path.cwd().resolve()
    for candidate in (working_directory, *working_directory.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and (candidate / "src" / "longfeedback").is_dir():
            return candidate
    return working_directory


def _seeded_gate_b(config: GateBConfig, seed: int) -> GateBConfig:
    return config.model_copy(
        update={"experiment": config.experiment.model_copy(update={"seed": seed})}
    )


def _seeded_e5(config: E5Config, seed: int) -> E5Config:
    return config.model_copy(update={"seed": seed})


def _write_seed_table_csv(
    path: Path, per_experiment_tables: dict[str, list[SeedTable]], seeds: tuple[int, ...]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "group", "metric", "seed", "value"])
        for experiment, tables in per_experiment_tables.items():
            for seed, table in zip(seeds, tables, strict=True):
                for group in sorted(table):
                    for metric in sorted(table[group]):
                        writer.writerow([experiment, group, metric, seed, table[group][metric]])


def run_multiseed(config: MultiSeedConfig, *, output_dir: Path | None = None) -> ExperimentResult:
    """Run the multi-seed protocol and persist an auditable set of artifacts."""

    # Imported lazily: the aggregation half of this module stays usable
    # without the research extra (torch).
    from longfeedback.experiments.e5 import run_e5
    from longfeedback.experiments.gate_b import run_gate_b

    started = time.perf_counter()
    repository = _repository_root()
    resolved_output = output_dir or config.output_dir
    if not resolved_output.is_absolute():
        resolved_output = repository / resolved_output
    resolved_output.mkdir(parents=True, exist_ok=True)

    per_experiment_tables: dict[str, list[SeedTable]] = {}
    per_seed_runs: dict[str, dict[str, Any]] = {}
    for experiment in config.experiments:
        tables: list[SeedTable] = []
        runs: dict[str, Any] = {}
        for seed in config.seeds:
            seed_dir = resolved_output / experiment / f"seed_{seed}"
            if experiment == "gate_b":
                result = run_gate_b(_seeded_gate_b(config.gate_b, seed), output_dir=seed_dir)
                tables.append(gate_b_seed_table(result.metrics))
            else:
                result = run_e5(_seeded_e5(config.e5, seed), output_dir=seed_dir)
                tables.append(e5_seed_table(result.metrics))
            runs[str(seed)] = {
                "status": result.metrics["status"],
                "scientific_metrics_sha256": result.metrics["scientific_metrics_sha256"],
                "output_dir": str(seed_dir.relative_to(resolved_output)),
            }
        per_experiment_tables[experiment] = tables
        per_seed_runs[experiment] = runs

    aggregated: dict[str, AggregateTable] = {
        experiment: aggregate_seed_tables(tables, config.bootstrap)
        for experiment, tables in per_experiment_tables.items()
    }

    seed_count_met = len(config.seeds) >= config.min_seeds
    decision: dict[str, Any] = {
        "seed_count": len(config.seeds),
        "min_seeds": config.min_seeds,
        "protocol_seed_count_met": bool(seed_count_met),
    }
    robust_components: list[bool] = [bool(seed_count_met)]
    if "gate_b" in aggregated:
        gate_b_verdicts = gate_b_robustness(aggregated["gate_b"], config.gate_b.decision)
        decision["gate_b"] = gate_b_verdicts
        robust_components.append(
            gate_b_verdicts["credit_recovery_robust"]
            and gate_b_verdicts["uncertainty_under_shift_robust"]
        )
    if "e5" in aggregated:
        e5_verdicts = e5_robustness(aggregated["e5"], config.e5.decision)
        decision["e5"] = e5_verdicts
        robust_components.append(
            e5_verdicts["goodhart_robust"]
            and (e5_verdicts["kl_mitigation_robust"] or e5_verdicts["lcb_mitigation_robust"])
        )
    decision["pass"] = all(robust_components)

    metrics: dict[str, Any] = {
        "experiment": "multiseed",
        "status": "pass" if decision["pass"] else "fail",
        "seeds": list(config.seeds),
        "runtime_seconds": 0.0,
        "multiseed_decision": decision,
        "bootstrap": {
            "resamples": config.bootstrap.resamples,
            "confidence": config.bootstrap.confidence,
            "seed": config.bootstrap.seed,
        },
        "per_seed_runs": per_seed_runs,
        "tables": aggregated,
        "metric_conventions": {
            "predeclared_primary_metrics": {
                "gate_b": (
                    "credit_margin: docm_credit credit Spearman minus the best of "
                    "docm_outcome/docm_prefix, per family"
                ),
                "e5": (
                    "hacking_gap_single (Goodhart effect size) plus paired kl/lcb "
                    "hacking-gap reduction and utility gain at proxy-optimal, per regime"
                ),
            },
            "pairing": (
                "variants share evaluation seeds within a run, so per-seed "
                "differences are paired; the bootstrap resamples those paired "
                "differences over seeds"
            ),
            "multiple_comparisons": (
                "per-family / per-regime rows are reported in full; the robustness "
                "verdicts aggregate them with the same min_winning_families / "
                "any-regime rules as the single-seed decisions rather than testing "
                "each row in isolation"
            ),
            "effect_sizes": "native units (Spearman points, utility points), never bare p-values",
        },
    }

    metrics_path = resolved_output / config.metrics_filename
    seed_table_path = resolved_output / config.seed_table_filename
    manifest_path = resolved_output / config.manifest_filename
    _write_seed_table_csv(seed_table_path, per_experiment_tables, config.seeds)

    metrics["runtime_seconds"] = time.perf_counter() - started
    scientific_metrics = {key: value for key, value in metrics.items() if key != "runtime_seconds"}
    metrics["scientific_metrics_sha256"] = sha256_json(scientific_metrics)
    write_metrics_json(metrics, metrics_path)
    manifest = build_run_manifest(
        repository=repository,
        resolved_config=dump_resolved_config(config),
        artifacts={
            "metrics": metrics_path.name,
            "seed_table": seed_table_path.name,
            "manifest": manifest_path.name,
        },
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ExperimentResult(
        metrics=metrics,
        output_dir=resolved_output,
        artifacts={
            "metrics": metrics_path,
            "seed_table": seed_table_path,
            "manifest": manifest_path,
        },
    )
