"""E9: randomized proximal and distal causal effects in HeartSteps V1."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from longfeedback.config import E9Config, dump_resolved_config
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


def _fold(user: str, folds: int) -> int:
    return int.from_bytes(hashlib.sha256(user.encode()).digest()[:8], "big") % folds


def _ridge_fit(x: FloatArray, y: FloatArray, alpha: float) -> FloatArray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.asarray(np.linalg.solve(x.T @ x + penalty, x.T @ y), dtype=np.float64)


def crossfit_arm_predictions(
    x: FloatArray,
    y: FloatArray,
    action: npt.NDArray[np.bool_],
    users: npt.NDArray[np.str_],
    *,
    folds: int,
    alpha: float,
) -> tuple[FloatArray, FloatArray]:
    """Participant-level cross-fitted outcome regressions for each action arm."""

    assignments = np.asarray([_fold(user, folds) for user in users], dtype=np.int64)
    mu0 = np.empty(len(y), dtype=np.float64)
    mu1 = np.empty(len(y), dtype=np.float64)
    for fold in range(folds):
        holdout = assignments == fold
        train = ~holdout
        for arm, output in ((False, mu0), (True, mu1)):
            selected = train & (action == arm)
            if np.sum(selected) < x.shape[1]:
                raise ValueError(f"insufficient cross-fit examples for action={int(arm)}")
            coefficients = _ridge_fit(x[selected], y[selected], alpha)
            output[holdout] = x[holdout] @ coefficients
    return mu0, mu1


def distal_scores(
    outcome: FloatArray,
    action: npt.NDArray[np.bool_],
    probability: FloatArray,
    mu0: FloatArray,
    mu1: FloatArray,
) -> FloatArray:
    """DCEE orthogonal score from the published HeartSteps estimator."""

    p_action = np.where(action, probability, 1.0 - probability)
    sign = np.where(action, 1.0, -1.0)
    residual = outcome - (1.0 - probability) * mu1 - probability * mu0
    return np.asarray(sign / p_action * residual, dtype=np.float64)


def _cluster_mean_ci(
    values: FloatArray,
    users: npt.NDArray[np.str_],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    unique, inverse = np.unique(users, return_inverse=True)
    totals = np.zeros((len(unique), 2), dtype=np.float64)
    np.add.at(totals[:, 0], inverse, values)
    np.add.at(totals[:, 1], inverse, 1.0)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.integers(0, len(unique), size=len(unique))
        aggregate = totals[selected].sum(axis=0)
        draws[index] = aggregate[0] / aggregate[1]
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(np.mean(values)),
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
    }


def _proximal_effect(
    outcome: FloatArray,
    action: npt.NDArray[np.bool_],
    users: npt.NDArray[np.str_],
    config: E9Config,
) -> dict[str, float]:
    treated = outcome[action]
    reference = outcome[~action]
    estimate = float(np.mean(treated) - np.mean(reference))
    unique = np.unique(users)
    per_user = np.zeros((len(unique), 4), dtype=np.float64)
    lookup = {user: index for index, user in enumerate(unique)}
    for value, arm, user in zip(outcome, action, users, strict=True):
        index = lookup[user]
        offset = 0 if arm else 2
        per_user[index, offset] += value
        per_user[index, offset + 1] += 1.0
    rng = np.random.default_rng(config.bootstrap.seed + 101)
    draws = np.empty(config.bootstrap.resamples, dtype=np.float64)
    valid = np.all(per_user[:, [1, 3]] > 0.0, axis=1)
    clusters = per_user[valid]
    for index in range(config.bootstrap.resamples):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        total = clusters[selected].sum(axis=0)
        draws[index] = total[0] / total[1] - total[2] / total[3]
    alpha = (1.0 - config.bootstrap.confidence) / 2.0
    return {
        "log_step_effect": estimate,
        "multiplicative_effect": float(np.expm1(estimate)),
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
    }


def run_e9(config: E9Config, *, output_dir: Path | None = None) -> ExperimentResult:
    """Run the HeartSteps positive-control and distal causal-effect benchmark."""

    started = time.perf_counter()
    repository = _repository_root()
    resolved_output = (output_dir or config.output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    decisions_path = config.processed_dir / config.decisions_filename
    if not decisions_path.is_file():
        raise FileNotFoundError(f"prepared HeartSteps decisions not found at {decisions_path}")
    columns = [
        "user_id",
        "available",
        "randomized",
        "action",
        "action_probability",
        "proximal_steps30",
        "prior_steps30",
        "home_work",
        "study_day",
        "analysis_period",
        "distal_week_daily_steps",
    ]
    frame = pq.read_table(decisions_path, columns=columns).to_pydict()
    eligible = (
        np.asarray(frame["available"], dtype=np.bool_)
        & np.asarray(frame["randomized"], dtype=np.bool_)
        & np.asarray(frame["analysis_period"], dtype=np.bool_)
    )
    users = np.asarray(frame["user_id"], dtype=np.str_)[eligible]
    action = np.asarray(frame["action"], dtype=np.bool_)[eligible]
    probability = np.asarray(frame["action_probability"], dtype=np.float64)[eligible]
    proximal = np.log1p(np.asarray(frame["proximal_steps30"], dtype=np.float64)[eligible])
    prior = np.log1p(np.asarray(frame["prior_steps30"], dtype=np.float64)[eligible])
    home_work = np.asarray(frame["home_work"], dtype=np.float64)[eligible]
    day = np.asarray(frame["study_day"], dtype=np.float64)[eligible]
    distal = np.asarray(frame["distal_week_daily_steps"], dtype=np.float64)[eligible]
    day_scaled = (day - np.mean(day)) / np.std(day)
    prior_scaled = (prior - np.mean(prior)) / np.std(prior)
    features = np.column_stack(
        [np.ones(len(day)), prior_scaled, home_work, day_scaled, day_scaled * day_scaled]
    )

    proximal_result = _proximal_effect(proximal, action, users, config)
    mu0, mu1 = crossfit_arm_predictions(
        features,
        distal,
        action,
        users,
        folds=config.crossfit_folds,
        alpha=config.ridge_alpha,
    )
    scores = distal_scores(distal, action, probability, mu0, mu1)
    distal_result = _cluster_mean_ci(
        scores,
        users,
        resamples=config.bootstrap.resamples,
        confidence=config.bootstrap.confidence,
        seed=config.bootstrap.seed + 202,
    )
    edges = np.quantile(day, [0.0, 0.25, 0.5, 0.75, 1.0])
    day_bin = np.clip(np.digitize(day, edges[1:-1], right=True), 0, 3)
    time_effects = []
    for index in range(4):
        selected = day_bin == index
        summary = _cluster_mean_ci(
            scores[selected],
            users[selected],
            resamples=config.bootstrap.resamples,
            confidence=config.bootstrap.confidence,
            seed=config.bootstrap.seed + 300 + index,
        )
        time_effects.append(
            {
                "period": index + 1,
                "day_low": float(edges[index]),
                "day_high": float(edges[index + 1]),
                "rows": int(np.sum(selected)),
                **summary,
            }
        )

    positive_control_direction = proximal_result["log_step_effect"] > 0.0
    metrics: dict[str, Any] = {
        "experiment": "e9",
        "status": "pass" if positive_control_direction else "warning",
        "data": {
            "participants": len(np.unique(users)),
            "eligible_analysis_decisions": len(users),
            "observed_action_rate": float(np.mean(action)),
        },
        "proximal_positive_control": proximal_result,
        "distal_average_excursion_effect": distal_result,
        "distal_time_effects": time_effects,
        "e9_decision": {
            "proximal_effect_positive": positive_control_direction,
            "published_direction_reproduced": positive_control_direction,
            "distal_ci_excludes_zero": bool(
                distal_result["ci_low"] > 0.0 or distal_result["ci_high"] < 0.0
            ),
            "model_grading_authorized": bool(
                distal_result["ci_low"] > 0.0 or distal_result["ci_high"] < 0.0
            ),
        },
        "claim_scope": (
            "randomization-identified group-level excursion effects; not individual "
            "counterfactual credit and not a conversational-domain result"
        ),
    }
    predictions_path = resolved_output / config.predictions_filename
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(time_effects[0]))
        writer.writeheader()
        writer.writerows(time_effects)
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
