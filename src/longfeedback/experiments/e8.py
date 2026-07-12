"""E8 phase 1: real delayed effects at randomized KuaiRand session steps."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from longfeedback.config import E8Config, dump_resolved_config
from longfeedback.evaluation import write_metrics_json
from longfeedback.experiments.manifest import build_run_manifest, sha256_json
from longfeedback.experiments.types import ExperimentResult

FloatArray = npt.NDArray[np.float64]


def _repository_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/longfeedback").is_dir():
            return candidate
    return cwd


def ols_slope(x: FloatArray, y: FloatArray) -> float:
    """OLS slope with an intercept."""

    centered = x - np.mean(x)
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        raise ValueError("treatment attribute has zero variance")
    return float(np.dot(centered, y - np.mean(y)) / denominator)


def cluster_bootstrap_slopes(
    x: FloatArray,
    y: FloatArray,
    users: npt.NDArray[np.str_],
    *,
    resamples: int,
    seed: int,
) -> FloatArray:
    """Bootstrap users and compute slopes from per-user sufficient statistics."""

    unique, inverse = np.unique(users, return_inverse=True)
    groups = len(unique)
    sufficient = np.zeros((groups, 5), dtype=np.float64)
    np.add.at(sufficient[:, 0], inverse, 1.0)
    np.add.at(sufficient[:, 1], inverse, x)
    np.add.at(sufficient[:, 2], inverse, y)
    np.add.at(sufficient[:, 3], inverse, x * x)
    np.add.at(sufficient[:, 4], inverse, x * y)
    rng = np.random.default_rng(seed)
    slopes = np.empty(resamples, dtype=np.float64)
    batch_size = 64
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        draws = rng.integers(0, groups, size=(stop - start, groups))
        totals = sufficient[draws].sum(axis=1)
        n, sum_x, sum_y, sum_xx, sum_xy = totals.T
        denominator = sum_xx - sum_x * sum_x / n
        slopes[start:stop] = (sum_xy - sum_x * sum_y / n) / denominator
    return slopes


def _choose_horizon(config: E8Config, base_rates: dict[int, float]) -> int:
    eligible = [
        horizon
        for horizon, rate in base_rates.items()
        if config.base_rate_low <= rate <= config.base_rate_high
    ]
    if not eligible:
        return config.primary_horizon
    return min(eligible, key=lambda value: (abs(value - config.primary_horizon), value))


def _summary(
    x: FloatArray,
    y: FloatArray,
    users: npt.NDArray[np.str_],
    config: E8Config,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    slope = ols_slope(x, y)
    draws = cluster_bootstrap_slopes(
        x,
        y,
        users,
        resamples=config.bootstrap.resamples,
        seed=config.bootstrap.seed + seed_offset,
    )
    alpha = (1.0 - config.bootstrap.confidence) / 2.0
    return {
        "slope": slope,
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "steps": len(y),
        "users": len(np.unique(users)),
        "survival_rate": float(np.mean(y)),
    }


def run_e8(config: E8Config, *, output_dir: Path | None = None) -> ExperimentResult:
    """Run the frozen E8 phase-1 power gate."""

    started = time.perf_counter()
    repository = _repository_root()
    resolved_output = (output_dir or config.output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    sessions_path = config.processed_dir / config.sessions_filename
    if not sessions_path.is_file():
        raise FileNotFoundError(f"prepared KuaiRand sessions not found at {sessions_path}")

    columns = [
        "user_id",
        "duration_ms",
        "video_type",
        "is_rand",
        *[f"survival_{horizon}" for horizon in config.survival_horizons],
    ]
    frame = pq.read_table(sessions_path, columns=columns).to_pydict()
    mask = np.asarray(frame["is_rand"], dtype=np.int8) == 1
    duration = np.asarray(frame["duration_ms"], dtype=np.float64)[mask]
    finite = np.isfinite(duration) & (duration >= 0.0)
    users = np.asarray(frame["user_id"], dtype=np.str_)[mask][finite]
    video_types = np.asarray(
        ["unknown" if value is None else str(value) for value in frame["video_type"]],
        dtype=np.str_,
    )[mask][finite]
    log_duration = np.log1p(duration[finite])
    x = np.asarray((log_duration - np.mean(log_duration)) / np.std(log_duration), dtype=np.float64)
    outcomes = {
        horizon: np.asarray(frame[f"survival_{horizon}"], dtype=np.float64)[mask][finite]
        for horizon in config.survival_horizons
    }
    base_rates = {horizon: float(np.mean(values)) for horizon, values in outcomes.items()}
    primary_horizon = _choose_horizon(config, base_rates)
    horizon_results = {
        str(horizon): _summary(x, outcomes[horizon], users, config, seed_offset=horizon)
        for horizon in config.survival_horizons
    }
    primary = horizon_results[str(primary_horizon)]
    ci_excludes_zero = bool(primary["ci_low"] > 0.0 or primary["ci_high"] < 0.0)
    power_gate_pass = bool(
        ci_excludes_zero and abs(primary["slope"]) >= config.decision.min_abs_slope
    )
    primary["minimum_detectable_effect_80pct"] = float(
        config.decision.mde_z * primary["bootstrap_se"]
    )

    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, config.duration_quantile_bins + 1)))
    bins = np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)
    quintiles: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        selected = bins == index
        quintiles.append(
            {
                "bin": index + 1,
                "x_low": float(edges[index]),
                "x_high": float(edges[index + 1]),
                "steps": int(np.sum(selected)),
                "survival_rate": float(np.mean(outcomes[primary_horizon][selected])),
            }
        )
    type_rows = []
    for video_type in sorted(np.unique(video_types)):
        selected = video_types == video_type
        type_rows.append(
            {
                "video_type": str(video_type),
                "steps": int(np.sum(selected)),
                "survival_rate": float(np.mean(outcomes[primary_horizon][selected])),
                "mean_z_log_duration": float(np.mean(x[selected])),
            }
        )

    metrics: dict[str, Any] = {
        "experiment": "e8",
        "status": "pass" if power_gate_pass else "completed_null",
        "estimand": (
            "group-level intent-to-treat slope of within-session survival on randomized "
            "video log-duration; not individual counterfactual credit"
        ),
        "data": {
            "randomized_steps_analyzed": len(x),
            "users": len(np.unique(users)),
            "source_sessions_path": str(sessions_path),
        },
        "randomized_survival_base_rates": {str(k): v for k, v in base_rates.items()},
        "primary_horizon": primary_horizon,
        "horizon_effects": horizon_results,
        "duration_quantile_table": quintiles,
        "video_type_table": type_rows,
        "e8_decision": {
            "power_gate_pass": power_gate_pass,
            "ci_excludes_zero": ci_excludes_zero,
            "minimum_abs_slope_met": abs(primary["slope"]) >= config.decision.min_abs_slope,
            "hypothesis_h8_delayed_step_effect": (
                "supported" if power_gate_pass else "refuted_at_this_granularity"
            ),
            "phase_2_authorized": power_gate_pass,
        },
    }
    predictions_path = resolved_output / config.predictions_filename
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quintiles[0]))
        writer.writeheader()
        writer.writerows(quintiles)
    metrics_path = resolved_output / config.metrics_filename
    manifest_path = resolved_output / config.manifest_filename
    metrics["runtime_seconds"] = time.perf_counter() - started
    scientific = {key: value for key, value in metrics.items() if key != "runtime_seconds"}
    metrics["scientific_metrics_sha256"] = sha256_json(scientific)
    write_metrics_json(metrics, metrics_path)
    manifest = build_run_manifest(
        repository=repository,
        resolved_config=dump_resolved_config(config),
        artifacts={"metrics": metrics_path.name, "predictions": predictions_path.name},
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return ExperimentResult(
        metrics=metrics,
        output_dir=resolved_output,
        artifacts={
            "metrics": metrics_path,
            "predictions": predictions_path,
            "manifest": manifest_path,
        },
    )
