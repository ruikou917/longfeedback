"""E9b: proximal-horizon capacity-matched model grading on HeartSteps V1.

The contract is predeclared in docs/real_credit_protocol.md (E9b section,
2026-07-12). Stage 1 estimates the randomized proximal gate with the same
orthogonal-score machinery E9 validated; the DOCM ladder is trained only if
the gate authorizes grading. Stage 2 builds participant pseudo-day episodes,
supervises the credit variant with doubly-robust pseudo-outcomes computed
inside training folds only, and grades held-out per-decision credit against
randomization-derived pseudo-outcomes at suggestion-sent decisions.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from longfeedback.config import E9bConfig, dump_resolved_config
from longfeedback.evaluation import auroc, pearson_correlation, write_metrics_json
from longfeedback.evaluation.metrics import spearman_correlation
from longfeedback.experiments.e8 import cluster_bootstrap_slopes, ols_slope
from longfeedback.experiments.e9 import (
    _cluster_mean_ci,
    _fold,
    _repository_root,
    _ridge_fit,
    crossfit_arm_predictions,
    distal_scores,
)
from longfeedback.experiments.manifest import build_run_manifest, sha256_json
from longfeedback.experiments.types import ExperimentResult

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

VARIANT_NAMES: tuple[str, ...] = ("docm_outcome", "docm_prefix", "docm_credit")


@dataclass(frozen=True)
class DecisionTable:
    """Column view of the prepared HeartSteps decisions used by E9b."""

    users: npt.NDArray[np.str_]
    study_day: FloatArray
    eligible: BoolArray
    action: BoolArray
    probability: FloatArray
    proximal_log_steps: FloatArray
    prior_log_steps: FloatArray
    home_work: FloatArray
    interval_steps: FloatArray
    decision_number: IntArray


def load_decisions(path: Path) -> DecisionTable:
    columns = [
        "user_id",
        "study_day",
        "available",
        "randomized",
        "action",
        "action_probability",
        "proximal_steps30",
        "prior_steps30",
        "home_work",
        "steps_until_next_decision",
        "decision_number",
    ]
    frame = pq.read_table(path, columns=columns).to_pydict()
    return DecisionTable(
        users=np.asarray(frame["user_id"], dtype=np.str_),
        study_day=np.asarray(frame["study_day"], dtype=np.float64),
        eligible=(
            np.asarray(frame["available"], dtype=np.bool_)
            & np.asarray(frame["randomized"], dtype=np.bool_)
        ),
        action=np.asarray(frame["action"], dtype=np.bool_),
        probability=np.asarray(frame["action_probability"], dtype=np.float64),
        proximal_log_steps=np.log1p(np.asarray(frame["proximal_steps30"], dtype=np.float64)),
        prior_log_steps=np.log1p(np.asarray(frame["prior_steps30"], dtype=np.float64)),
        # home_work is NULL exactly on never-randomized decisions (no logged
        # location); those slots only appear as placeholder observations.
        home_work=np.asarray(
            [0.0 if value is None else float(value) for value in frame["home_work"]],
            dtype=np.float64,
        ),
        interval_steps=np.asarray(frame["steps_until_next_decision"], dtype=np.float64),
        decision_number=np.asarray(frame["decision_number"], dtype=np.int64),
    )


def _ridge_features(table: DecisionTable, rows: BoolArray) -> FloatArray:
    """E9's adjustment basis evaluated on selected rows with global scaling."""

    day = table.study_day[table.eligible]
    prior = table.prior_log_steps[table.eligible]
    day_scaled = (table.study_day[rows] - np.mean(day)) / np.std(day)
    prior_scaled = (table.prior_log_steps[rows] - np.mean(prior)) / np.std(prior)
    return np.column_stack(
        [
            np.ones(int(np.sum(rows))),
            prior_scaled,
            table.home_work[rows],
            day_scaled,
            day_scaled * day_scaled,
        ]
    )


def proximal_gate(table: DecisionTable, config: E9bConfig) -> dict[str, Any]:
    """Predeclared grading-authorization gate on the proximal excursion effect."""

    rows = table.eligible
    users = table.users[rows]
    action = table.action[rows]
    probability = table.probability[rows]
    outcome = table.proximal_log_steps[rows]
    day = table.study_day[rows]
    features = _ridge_features(table, rows)
    mu0, mu1 = crossfit_arm_predictions(
        features,
        outcome,
        action,
        users,
        folds=config.crossfit_folds,
        alpha=config.ridge_alpha,
    )
    scores = distal_scores(outcome, action, probability, mu0, mu1)
    average = _cluster_mean_ci(
        scores,
        users,
        resamples=config.bootstrap.resamples,
        confidence=config.bootstrap.confidence,
        seed=config.bootstrap.seed + 400,
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
            seed=config.bootstrap.seed + 410 + index,
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

    def _excludes_zero(summary: dict[str, float]) -> bool:
        return bool(summary["ci_low"] > 0.0 or summary["ci_high"] < 0.0)

    average_pass = _excludes_zero(average)
    quartile1_pass = _excludes_zero(time_effects[0])
    return {
        "eligible_decisions": int(np.sum(rows)),
        "participants": len(np.unique(users)),
        "average_proximal_effect": average,
        "proximal_time_effects": time_effects,
        "day_quartile_edges": [float(edge) for edge in edges],
        "average_ci_excludes_zero": average_pass,
        "quartile1_ci_excludes_zero": quartile1_pass,
        "grading_authorized": bool(average_pass or quartile1_pass),
    }


@dataclass(frozen=True)
class EpisodeTable:
    """Fixed-slot pseudo-day episodes with a row index back to decisions."""

    users: npt.NDArray[np.str_]
    observations: FloatArray
    actions: IntArray
    responses: FloatArray
    day_steps: FloatArray
    study_day: FloatArray
    slot_eligible: BoolArray
    slot_action: BoolArray
    slot_probability: FloatArray
    slot_outcome: FloatArray
    slot_day: FloatArray
    slot_rows: IntArray

    @property
    def episodes(self) -> int:
        return int(self.observations.shape[0])


def build_episodes(table: DecisionTable, config: E9bConfig) -> EpisodeTable:
    """Group decisions into participant pseudo-days of exactly `episode_slots`."""

    slots = config.episode_slots
    keys = list(zip(table.users.tolist(), table.study_day.tolist(), strict=True))
    groups: dict[tuple[str, float], list[int]] = {}
    for row, key in enumerate(keys):
        groups.setdefault(key, []).append(row)
    episode_rows = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: int(table.decision_number[row]))
        if len(rows) == slots:
            episode_rows.append((key, rows))

    episodes = len(episode_rows)
    user_list: list[str] = []
    observations = np.zeros((episodes, slots, 5), dtype=np.float64)
    actions = np.zeros((episodes, slots), dtype=np.int64)
    responses = np.zeros((episodes, slots), dtype=np.float64)
    day_steps = np.zeros(episodes, dtype=np.float64)
    study_day = np.zeros(episodes, dtype=np.float64)
    slot_eligible = np.zeros((episodes, slots), dtype=np.bool_)
    slot_action = np.zeros((episodes, slots), dtype=np.bool_)
    slot_probability = np.zeros((episodes, slots), dtype=np.float64)
    slot_outcome = np.zeros((episodes, slots), dtype=np.float64)
    slot_day = np.zeros((episodes, slots), dtype=np.float64)
    slot_rows = np.zeros((episodes, slots), dtype=np.int64)
    for episode, ((user, day), rows) in enumerate(episode_rows):
        user_list.append(str(user))
        study_day[episode] = day
        for slot, row in enumerate(rows):
            eligible = bool(table.eligible[row])
            acted = bool(table.action[row]) and eligible
            observations[episode, slot] = (
                float(eligible),
                table.prior_log_steps[row],
                table.home_work[row],
                table.study_day[row],
                float(slot),
            )
            actions[episode, slot] = int(acted)
            responses[episode, slot] = table.proximal_log_steps[row]
            slot_eligible[episode, slot] = eligible
            slot_action[episode, slot] = acted
            slot_probability[episode, slot] = table.probability[row]
            slot_outcome[episode, slot] = table.proximal_log_steps[row]
            slot_day[episode, slot] = table.study_day[row]
            slot_rows[episode, slot] = row
        day_steps[episode] = float(np.sum(table.interval_steps[rows]))
    users = np.asarray(user_list, dtype=np.str_)
    return EpisodeTable(
        users=users,
        observations=observations,
        actions=actions,
        responses=responses,
        day_steps=day_steps,
        study_day=study_day,
        slot_eligible=slot_eligible,
        slot_action=slot_action,
        slot_probability=slot_probability,
        slot_outcome=slot_outcome,
        slot_day=slot_day,
        slot_rows=slot_rows,
    )


def _arm_nuisances(
    table: DecisionTable,
    train_rows: BoolArray,
    predict_rows: BoolArray,
    *,
    alpha: float,
) -> tuple[FloatArray, FloatArray]:
    """Fit per-arm ridge regressions on training rows, evaluate elsewhere."""

    train_features = _ridge_features(table, train_rows)
    predict_features = _ridge_features(table, predict_rows)
    outcome = table.proximal_log_steps[train_rows]
    action = table.action[train_rows]
    mu = []
    for arm in (False, True):
        selected = action == arm
        coefficients = _ridge_fit(train_features[selected], outcome[selected], alpha)
        mu.append(predict_features @ coefficients)
    return mu[0], mu[1]


def _sent_pseudo_outcomes(
    outcome: FloatArray,
    probability: FloatArray,
    mu0: FloatArray,
    mu1: FloatArray,
) -> FloatArray:
    """DR pseudo-outcome for the logged suggestion-sent action versus reference."""

    return (outcome - (1.0 - probability) * mu1 - probability * mu0) / probability


def _affine_calibration(predictions: FloatArray, targets: FloatArray) -> tuple[float, float]:
    design = np.column_stack([predictions, np.ones_like(predictions)])
    solution, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return float(solution[0]), float(solution[1])


@dataclass(frozen=True)
class FoldLabels:
    """Seed-independent per-fold supervision and held-out grading targets."""

    train_episode_indices: IntArray
    test_episode_indices: IntArray
    day_threshold: float
    train_outcomes: FloatArray
    test_outcomes: FloatArray
    credit_targets: FloatArray
    credit_mask: BoolArray
    credit_se: FloatArray
    train_grading_mask: BoolArray
    train_xi: FloatArray
    test_grading_mask: BoolArray
    test_xi: FloatArray
    test_scores: FloatArray
    test_score_rows: IntArray


def prepare_fold(
    table: DecisionTable,
    episodes: EpisodeTable,
    fold: int,
    config: E9bConfig,
) -> FoldLabels:
    episode_folds = np.asarray(
        [_fold(user, config.evaluation_folds) for user in episodes.users], dtype=np.int64
    )
    train_eps = np.flatnonzero(episode_folds != fold).astype(np.int64)
    test_eps = np.flatnonzero(episode_folds == fold).astype(np.int64)

    threshold = float(np.median(episodes.day_steps[train_eps]))
    train_outcomes = (episodes.day_steps[train_eps] > threshold).astype(np.float64)
    test_outcomes = (episodes.day_steps[test_eps] > threshold).astype(np.float64)

    row_folds = np.asarray(
        [_fold(user, config.evaluation_folds) for user in table.users], dtype=np.int64
    )
    train_rows = table.eligible & (row_folds != fold)
    test_rows = table.eligible & (row_folds == fold)

    # Training supervision: participant-cross-fitted nuisances inside the
    # training fold only, pseudo-outcomes at suggestion-sent decisions.
    train_features = _ridge_features(table, train_rows)
    mu0_train, mu1_train = crossfit_arm_predictions(
        train_features,
        table.proximal_log_steps[train_rows],
        table.action[train_rows],
        table.users[train_rows],
        folds=config.crossfit_folds,
        alpha=config.ridge_alpha,
    )
    xi_by_row = np.zeros(len(table.users), dtype=np.float64)
    xi_by_row[train_rows] = _sent_pseudo_outcomes(
        table.proximal_log_steps[train_rows],
        table.probability[train_rows],
        mu0_train,
        mu1_train,
    )

    credit_targets = np.zeros((episodes.episodes, config.episode_slots), dtype=np.float64)
    credit_mask = np.zeros((episodes.episodes, config.episode_slots), dtype=np.bool_)
    credit_se = np.ones((episodes.episodes, config.episode_slots), dtype=np.float64)
    train_row_set = np.zeros(len(table.users), dtype=np.bool_)
    train_row_set[train_rows] = True
    for episode in train_eps:
        for slot in range(config.episode_slots):
            row = int(episodes.slot_rows[episode, slot])
            if episodes.slot_action[episode, slot] and train_row_set[row]:
                credit_targets[episode, slot] = xi_by_row[row]
                credit_mask[episode, slot] = True

    # Held-out grading targets: nuisances fitted on training participants,
    # never refitted on the held-out fold.
    mu0_test, mu1_test = _arm_nuisances(table, train_rows, test_rows, alpha=config.ridge_alpha)
    xi_test_by_row = np.zeros(len(table.users), dtype=np.float64)
    test_row_indices = np.flatnonzero(test_rows)
    xi_test_by_row[test_rows] = _sent_pseudo_outcomes(
        table.proximal_log_steps[test_rows],
        table.probability[test_rows],
        mu0_test,
        mu1_test,
    )
    test_scores = distal_scores(
        table.proximal_log_steps[test_rows],
        table.action[test_rows],
        table.probability[test_rows],
        mu0_test,
        mu1_test,
    )

    train_grading_mask = credit_mask[train_eps]
    train_xi = credit_targets[train_eps]
    test_grading_mask = episodes.slot_action[test_eps]
    test_xi = np.zeros_like(test_grading_mask, dtype=np.float64)
    for position, episode in enumerate(test_eps):
        for slot in range(config.episode_slots):
            if test_grading_mask[position, slot]:
                test_xi[position, slot] = xi_test_by_row[int(episodes.slot_rows[episode, slot])]
    return FoldLabels(
        train_episode_indices=train_eps,
        test_episode_indices=test_eps,
        day_threshold=threshold,
        train_outcomes=train_outcomes,
        test_outcomes=test_outcomes,
        credit_targets=credit_targets,
        credit_mask=credit_mask,
        credit_se=credit_se,
        train_grading_mask=train_grading_mask,
        train_xi=train_xi,
        test_grading_mask=test_grading_mask,
        test_xi=test_xi,
        test_scores=test_scores,
        test_score_rows=test_row_indices.astype(np.int64),
    )


def _train_seed(
    episodes: EpisodeTable,
    folds: list[FoldLabels],
    config: E9bConfig,
    seed: int,
) -> dict[str, Any]:
    """Train the ladder for one seed and pool held-out grading across folds."""

    from longfeedback.models import (
        DelayedOutcomeCreditModel,
        EncoderArchitecture,
        SequenceDataset,
        TrainingSettings,
        variant_loss_weights,
    )

    architecture = EncoderArchitecture(
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        dropout=config.model.dropout,
    )
    training = TrainingSettings(
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        grad_clip=config.training.grad_clip,
    )

    pooled: dict[str, dict[str, list[FloatArray]]] = {
        name: {"errors": [], "predictions": []} for name in VARIANT_NAMES
    }
    pooled_days: list[FloatArray] = []
    pooled_xi: list[FloatArray] = []
    auroc_by_variant: dict[str, list[tuple[float, int]]] = {name: [] for name in VARIANT_NAMES}
    parameter_counts: dict[str, int] = {}

    for labels in folds:
        train_eps = labels.train_episode_indices
        test_eps = labels.test_episode_indices
        train_dataset = SequenceDataset(
            observations=episodes.observations[train_eps],
            actions=episodes.actions[train_eps],
            responses=episodes.responses[train_eps],
            outcomes=labels.train_outcomes,
            credit_targets=labels.credit_targets[train_eps],
            credit_mask=labels.credit_mask[train_eps],
            credit_se=labels.credit_se[train_eps],
        )
        test_dataset = SequenceDataset(
            observations=episodes.observations[test_eps],
            actions=episodes.actions[test_eps],
            responses=episodes.responses[test_eps],
            outcomes=labels.test_outcomes,
        )
        train_mask = labels.train_grading_mask
        test_mask = labels.test_grading_mask
        for name in VARIANT_NAMES:
            model = DelayedOutcomeCreditModel(
                observation_dim=int(episodes.observations.shape[-1]),
                n_actions=2,
                horizon=config.episode_slots,
                reference_action=0,
                architecture=architecture,
                loss_weights=variant_loss_weights(name),
                seed=seed,
            )
            model.fit(train_dataset, training=training)
            parameter_counts[name] = model.parameter_count()
            outcome_predictions = model.predict_outcome_probability(test_dataset)
            auroc_by_variant[name].append(
                (auroc(labels.test_outcomes, outcome_predictions), len(test_eps))
            )
            train_predictions = _variant_credit_predictions(name, model, train_dataset)
            test_predictions = _variant_credit_predictions(name, model, test_dataset)
            scale, shift = _affine_calibration(
                train_predictions[train_mask], labels.train_xi[train_mask]
            )
            calibrated = scale * test_predictions[test_mask] + shift
            pooled[name]["errors"].append(np.square(calibrated - labels.test_xi[test_mask]))
            pooled[name]["predictions"].append(calibrated)
        pooled_xi.append(labels.test_xi[test_mask])
        pooled_days.append(episodes.slot_day[test_eps][test_mask])

    slot_users: list[npt.NDArray[np.str_]] = []
    for labels in folds:
        test_eps = labels.test_episode_indices
        mask = labels.test_grading_mask
        users_grid = np.repeat(episodes.users[test_eps][:, None], config.episode_slots, axis=1)
        slot_users.append(users_grid[mask])

    errors = {name: np.concatenate(pooled[name]["errors"]) for name in VARIANT_NAMES}
    predictions = {name: np.concatenate(pooled[name]["predictions"]) for name in VARIANT_NAMES}
    return {
        "errors": errors,
        "predictions": predictions,
        "xi": np.concatenate(pooled_xi),
        "days": np.concatenate(pooled_days),
        "users": np.concatenate(slot_users),
        "auroc": {
            name: float(
                np.average(
                    [value for value, _ in rows],
                    weights=[weight for _, weight in rows],
                )
            )
            for name, rows in auroc_by_variant.items()
        },
        "parameter_counts": parameter_counts,
    }


def _variant_credit_predictions(name: str, model: Any, dataset: Any) -> FloatArray:
    if name == "docm_credit":
        return np.asarray(model.predict_logged_credit(dataset), dtype=np.float64)
    if name == "docm_prefix":
        return np.asarray(np.diff(model.predict_prefix_values(dataset), axis=-1))
    return np.asarray(np.diff(model.predict_outcome_head_prefix_values(dataset), axis=-1))


def run_e9b(config: E9bConfig, *, output_dir: Path | None = None) -> ExperimentResult:
    """Run the predeclared E9b gate and, if authorized, ladder grading."""

    started = time.perf_counter()
    repository = _repository_root()
    resolved_output = (output_dir or config.output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    decisions_path = config.processed_dir / config.decisions_filename
    if not decisions_path.is_file():
        raise FileNotFoundError(f"prepared HeartSteps decisions not found at {decisions_path}")
    table = load_decisions(decisions_path)
    gate = proximal_gate(table, config)

    metrics: dict[str, Any] = {
        "experiment": "e9b",
        "contract": "docs/real_credit_protocol.md E9b section, predeclared 2026-07-12",
        "claim_scope": (
            "estimation-level credit grading against randomization-identified "
            "proximal excursion effects in a 37-participant mobile-health MRT; "
            "not a conversational-domain or decision-level result"
        ),
        "gate": gate,
    }

    if not gate["grading_authorized"]:
        metrics["status"] = "completed_null"
        metrics["e9b_decision"] = {
            "grading_authorized": False,
            "core_claim_supported_at_proximal_horizon": False,
        }
        return _finalize(metrics, config, resolved_output, repository, started, rows=[])

    episodes = build_episodes(table, config)
    folds = [prepare_fold(table, episodes, fold, config) for fold in range(config.evaluation_folds)]
    metrics["data"] = {
        "episodes": episodes.episodes,
        "participants": len(np.unique(episodes.users)),
        "graded_sent_decisions": int(
            sum(int(np.sum(labels.test_grading_mask)) for labels in folds)
        ),
        "day_thresholds": [labels.day_threshold for labels in folds],
    }

    seed_results: dict[int, dict[str, Any]] = {}
    for seed in config.decision.seeds:
        seed_results[seed] = _train_seed(episodes, folds, config, seed)

    per_seed: dict[str, Any] = {}
    positive_seeds = 0
    negative_ci_seeds = 0
    for seed, result in seed_results.items():
        gap = result["errors"]["docm_outcome"] - result["errors"]["docm_credit"]
        summary = _cluster_mean_ci(
            gap,
            result["users"],
            resamples=config.bootstrap.resamples,
            confidence=config.bootstrap.confidence,
            seed=config.bootstrap.seed + 500 + seed,
        )
        prefix_gap = result["errors"]["docm_prefix"] - result["errors"]["docm_credit"]
        prefix_summary = _cluster_mean_ci(
            prefix_gap,
            result["users"],
            resamples=config.bootstrap.resamples,
            confidence=config.bootstrap.confidence,
            seed=config.bootstrap.seed + 600 + seed,
        )
        positive_seeds += int(summary["estimate"] > 0.0)
        negative_ci_seeds += int(summary["ci_high"] < 0.0)
        per_seed[str(seed)] = {
            "primary_gap_outcome_minus_credit": summary,
            "prefix_gap_prefix_minus_credit": prefix_summary,
            "mean_squared_error": {
                name: float(np.mean(result["errors"][name])) for name in VARIANT_NAMES
            },
            "outcome_auroc": result["auroc"],
        }

    seed0 = seed_results[config.decision.seeds[0]]
    primary = per_seed[str(config.decision.seeds[0])]["primary_gap_outcome_minus_credit"]
    auroc_gap = abs(seed0["auroc"]["docm_credit"] - seed0["auroc"]["docm_outcome"])
    matched_quality = bool(auroc_gap <= config.decision.auroc_tolerance)
    primary_pass = bool(primary["estimate"] > 0.0 and primary["ci_low"] > 0.0)
    robust = bool(positive_seeds >= config.decision.min_positive_seeds and negative_ci_seeds == 0)

    # Secondary descriptive diagnostics on the seed-0 pooled held-out slots.
    day_edges = np.asarray(gate["day_quartile_edges"], dtype=np.float64)
    pooled_score_days = np.concatenate(
        [table.study_day[labels.test_score_rows] for labels in folds]
    )
    pooled_scores = np.concatenate([labels.test_scores for labels in folds])
    pooled_score_users = np.concatenate([table.users[labels.test_score_rows] for labels in folds])
    score_bins = np.clip(np.digitize(pooled_score_days, day_edges[1:-1], right=True), 0, 3)
    prediction_bins = np.clip(np.digitize(seed0["days"], day_edges[1:-1], right=True), 0, 3)
    bin_truth = []
    for index in range(4):
        selected = score_bins == index
        bin_truth.append(
            _cluster_mean_ci(
                pooled_scores[selected],
                pooled_score_users[selected],
                resamples=config.bootstrap.resamples,
                confidence=config.bootstrap.confidence,
                seed=config.bootstrap.seed + 700 + index,
            )
        )
    bin_profiles: dict[str, Any] = {"held_out_truth": bin_truth}
    for name in VARIANT_NAMES:
        bin_means = [
            float(np.mean(seed0["predictions"][name][prediction_bins == index]))
            for index in range(4)
        ]
        bin_profiles[name] = {
            "bin_mean_predicted_credit": bin_means,
            "pearson_vs_truth": pearson_correlation(
                np.asarray([entry["estimate"] for entry in bin_truth]),
                np.asarray(bin_means),
            ),
        }

    day_all = pooled_score_days
    day_z = (day_all - np.mean(day_all)) / np.std(day_all)
    trend_slope = ols_slope(day_z, pooled_scores)
    trend_draws = cluster_bootstrap_slopes(
        day_z,
        pooled_scores,
        pooled_score_users,
        resamples=config.bootstrap.resamples,
        seed=config.bootstrap.seed + 800,
    )
    alpha = (1.0 - config.bootstrap.confidence) / 2.0
    trend = {
        "slope_per_sd_day": trend_slope,
        "ci_low": float(np.quantile(trend_draws, alpha)),
        "ci_high": float(np.quantile(trend_draws, 1.0 - alpha)),
    }
    trend_detected = bool(trend["ci_low"] > 0.0 or trend["ci_high"] < 0.0)
    attenuation: dict[str, Any] = {
        "held_out_time_trend": trend,
        "trend_ci_excludes_zero": trend_detected,
    }
    if trend_detected:
        for name in VARIANT_NAMES:
            correlation = spearman_correlation(seed0["days"], seed0["predictions"][name])
            attenuation[name] = {
                "spearman_predicted_credit_vs_day": correlation,
                "sign_matches_randomized_trend": bool(np.sign(correlation) == np.sign(trend_slope)),
            }

    metrics["model"] = {
        "parameter_counts": seed0["parameter_counts"],
        "capacity_matched": len(set(seed0["parameter_counts"].values())) == 1,
    }
    metrics["per_seed"] = per_seed
    metrics["secondary"] = {"day_quartile_profile": bin_profiles, "attenuation": attenuation}
    metrics["e9b_decision"] = {
        "grading_authorized": True,
        "matched_outcome_quality": matched_quality,
        "seed0_auroc_gap": auroc_gap,
        "primary_ci_excludes_zero_positive": primary_pass,
        "positive_gap_seeds": positive_seeds,
        "negative_ci_seeds": negative_ci_seeds,
        "robust_across_seeds": robust,
        "core_claim_supported_at_proximal_horizon": bool(
            matched_quality and primary_pass and robust
        ),
    }
    metrics["status"] = (
        "pass" if metrics["e9b_decision"]["core_claim_supported_at_proximal_horizon"] else "fail"
    )

    rows = []
    for index in range(len(seed0["xi"])):
        rows.append(
            {
                "user_id": str(seed0["users"][index]),
                "study_day": float(seed0["days"][index]),
                "held_out_xi": float(seed0["xi"][index]),
                **{
                    f"calibrated_{name}": float(seed0["predictions"][name][index])
                    for name in VARIANT_NAMES
                },
            }
        )
    return _finalize(metrics, config, resolved_output, repository, started, rows=rows)


def _finalize(
    metrics: dict[str, Any],
    config: E9bConfig,
    resolved_output: Path,
    repository: Path,
    started: float,
    *,
    rows: list[dict[str, Any]],
) -> ExperimentResult:
    predictions_path = resolved_output / config.predictions_filename
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        else:
            handle.write("no grading rows: gate did not authorize model grading\n")
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
