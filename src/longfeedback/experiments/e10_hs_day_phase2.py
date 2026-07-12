"""E10-HS-Day Phase 2: capacity-matched grading on randomized credit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field

from longfeedback.evaluation import write_metrics_json
from longfeedback.experiments.e9 import (
    _cluster_mean_ci,
    _fold,
    _repository_root,
    _ridge_fit,
)
from longfeedback.experiments.manifest import build_run_manifest, sha256_json
from longfeedback.experiments.types import ExperimentResult

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
StringArray = npt.NDArray[np.str_]
VARIANTS = ("docm_outcome", "docm_prefix", "docm_credit")


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    d_model: int = Field(ge=8)
    n_layers: int = Field(ge=1)
    n_heads: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)


class TrainingSettingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    grad_clip: float = Field(gt=0.0)


class E10HSDayPhase2Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["e10_hs_day_phase2"]
    processed_dir: Path
    decisions_filename: str
    output_dir: Path
    evaluation_folds: int = Field(ge=2)
    nuisance_folds: int = Field(ge=2)
    ridge_alpha: float = Field(ge=0.0)
    episode_slots: int = Field(ge=2)
    bootstrap_resamples: int = Field(ge=100)
    bootstrap_confidence: float = Field(gt=0.5, lt=1.0)
    bootstrap_seed: int
    model: ModelSettings
    training: TrainingSettingsConfig
    seeds: tuple[int, ...]
    min_positive_seeds: int = Field(ge=1)
    outcome_rmse_tolerance: float = Field(ge=0.0)
    metrics_filename: str
    predictions_filename: str
    manifest_filename: str


def load_config(path: Path) -> E10HSDayPhase2Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("E10-HS-Day Phase-2 config must be a YAML mapping")
    return E10HSDayPhase2Config.model_validate(raw)


@dataclass(frozen=True)
class Episodes:
    users: StringArray
    observations: FloatArray
    actions: IntArray
    terminal_scores: FloatArray
    eligible: BoolArray
    probabilities: FloatArray

    @property
    def count(self) -> int:
        return len(self.users)


def build_episodes(path: Path, *, slots: int) -> Episodes:
    columns = [
        "user_id",
        "study_day",
        "decision_number",
        "available",
        "randomized",
        "action",
        "action_probability",
        "proximal_steps30",
        "prior_steps30",
        "home_work",
    ]
    frame = pq.read_table(path, columns=columns).to_pydict()
    groups: dict[tuple[str, float], list[int]] = {}
    for row, (user, day) in enumerate(zip(frame["user_id"], frame["study_day"], strict=True)):
        groups.setdefault((str(user), float(day)), []).append(row)

    complete = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: int(frame["decision_number"][row]))
        if len(rows) == slots:
            complete.append((key, rows))

    user_values: list[str] = []
    observations = np.zeros((len(complete), slots, 5), dtype=np.float64)
    actions = np.zeros((len(complete), slots), dtype=np.int64)
    terminal_scores = np.zeros(len(complete), dtype=np.float64)
    eligible = np.zeros((len(complete), slots), dtype=np.bool_)
    probabilities = np.zeros((len(complete), slots), dtype=np.float64)
    for episode, ((user, day), rows) in enumerate(complete):
        user_values.append(user)
        terminal_scores[episode] = float(
            np.mean(
                np.log1p(np.asarray([frame["proximal_steps30"][row] for row in rows], dtype=float))
            )
        )
        for position, row in enumerate(rows):
            available = bool(frame["available"][row]) and bool(frame["randomized"][row])
            sent = available and bool(frame["action"][row])
            prior = frame["prior_steps30"][row]
            context = frame["home_work"][row]
            observations[episode, position] = (
                float(available),
                float(np.log1p(0.0 if prior is None else prior)),
                float(0 if context is None else context),
                day,
                float(position + 1),
            )
            actions[episode, position] = int(sent)
            eligible[episode, position] = available
            probabilities[episode, position] = (
                float(frame["action_probability"][row]) if available else 0.6
            )
    return Episodes(
        users=np.asarray(user_values, dtype=np.str_),
        observations=observations,
        actions=actions,
        terminal_scores=terminal_scores,
        eligible=eligible,
        probabilities=probabilities,
    )


def _decision_arrays(episodes: Episodes) -> dict[str, npt.NDArray[Any]]:
    count, slots = episodes.actions.shape
    repeated_users = np.repeat(episodes.users[:, None], slots, axis=1)
    repeated_outcomes = np.repeat(episodes.terminal_scores[:, None], slots, axis=1)
    obs = episodes.observations
    day = obs[:, :, 3]
    day_z = day / 42.0
    position = obs[:, :, 4].astype(np.int64)
    position_indicators = np.stack([position == value for value in range(2, slots + 1)], axis=-1)
    features = np.concatenate(
        [
            np.ones((count, slots, 1)),
            obs[:, :, 1:3],
            day_z[:, :, None],
            np.square(day_z)[:, :, None],
            position_indicators.astype(np.float64),
            (position_indicators * obs[:, :, 2, None]).astype(np.float64),
        ],
        axis=-1,
    )
    return {
        "users": repeated_users,
        "outcomes": repeated_outcomes,
        "actions": episodes.actions.astype(np.bool_),
        "probabilities": episodes.probabilities,
        "features": features,
    }


def _sent_xi(
    outcome: FloatArray,
    probability: FloatArray,
    mu0: FloatArray,
    mu1: FloatArray,
) -> FloatArray:
    return (outcome - (1.0 - probability) * mu1 - probability * mu0) / probability


def _arm_predictions(
    train_features: FloatArray,
    train_outcomes: FloatArray,
    train_actions: BoolArray,
    test_features: FloatArray,
    *,
    alpha: float,
) -> tuple[FloatArray, FloatArray]:
    predictions = []
    for arm in (False, True):
        coefficients = _ridge_fit(
            train_features[train_actions == arm],
            train_outcomes[train_actions == arm],
            alpha,
        )
        predictions.append(test_features @ coefficients)
    return predictions[0], predictions[1]


def _crossfit_arm_predictions(
    features: FloatArray,
    outcomes: FloatArray,
    actions: BoolArray,
    users: StringArray,
    *,
    folds: int,
    alpha: float,
) -> tuple[FloatArray, FloatArray]:
    """Balanced deterministic participant folds independent of evaluation folds."""

    unique = np.unique(users)
    ordered = sorted(
        unique.tolist(),
        key=lambda user: hashlib.sha256(f"e10-nuisance:{user}".encode()).digest(),
    )
    assignment = {user: index % folds for index, user in enumerate(ordered)}
    row_folds = np.asarray([assignment[user] for user in users], dtype=np.int64)
    predictions = [np.empty(len(outcomes), dtype=np.float64) for _ in range(2)]
    for fold in range(folds):
        holdout = row_folds == fold
        for arm in (0, 1):
            selected = (~holdout) & (actions == bool(arm))
            if np.sum(selected) < features.shape[1]:
                raise ValueError(f"insufficient nuisance examples for action={arm}")
            coefficients = _ridge_fit(features[selected], outcomes[selected], alpha)
            predictions[arm][holdout] = features[holdout] @ coefficients
    return predictions[0], predictions[1]


@dataclass(frozen=True)
class FoldData:
    train_indices: IntArray
    test_indices: IntArray
    outcome_mean: float
    outcome_scale: float
    credit_targets: FloatArray
    credit_mask: BoolArray
    train_xi: FloatArray
    test_xi: FloatArray
    test_mask: BoolArray


def prepare_fold(
    episodes: Episodes,
    arrays: dict[str, npt.NDArray[Any]],
    fold: int,
    config: E10HSDayPhase2Config,
) -> FoldData:
    episode_folds = np.asarray(
        [_fold(user, config.evaluation_folds) for user in episodes.users], dtype=np.int64
    )
    train_indices = np.flatnonzero(episode_folds != fold).astype(np.int64)
    test_indices = np.flatnonzero(episode_folds == fold).astype(np.int64)
    outcome_mean = float(np.mean(episodes.terminal_scores[train_indices]))
    outcome_scale = float(np.std(episodes.terminal_scores[train_indices]))

    train_eligible = episodes.eligible[train_indices]
    train_users = np.asarray(arrays["users"])[train_indices][train_eligible].astype(np.str_)
    train_outcomes = np.asarray(arrays["outcomes"], dtype=np.float64)[train_indices][train_eligible]
    train_actions = np.asarray(arrays["actions"], dtype=np.bool_)[train_indices][train_eligible]
    train_probabilities = np.asarray(arrays["probabilities"], dtype=np.float64)[train_indices][
        train_eligible
    ]
    train_features = np.asarray(arrays["features"], dtype=np.float64)[train_indices][train_eligible]
    mu0_train, mu1_train = _crossfit_arm_predictions(
        train_features,
        train_outcomes,
        train_actions,
        train_users,
        folds=config.nuisance_folds,
        alpha=config.ridge_alpha,
    )
    train_xi_flat = _sent_xi(train_outcomes, train_probabilities, mu0_train, mu1_train)
    credit_targets = np.zeros_like(episodes.actions, dtype=np.float64)
    credit_mask = np.zeros_like(episodes.eligible)
    train_sent = train_eligible & (episodes.actions[train_indices] == 1)
    eligible_xi_grid = np.zeros_like(train_eligible, dtype=np.float64)
    eligible_xi_grid[train_eligible] = train_xi_flat
    credit_targets[train_indices] = eligible_xi_grid / outcome_scale
    credit_mask[train_indices] = train_sent

    test_eligible = episodes.eligible[test_indices]
    test_outcomes = np.asarray(arrays["outcomes"], dtype=np.float64)[test_indices][test_eligible]
    test_probabilities = np.asarray(arrays["probabilities"], dtype=np.float64)[test_indices][
        test_eligible
    ]
    test_features = np.asarray(arrays["features"], dtype=np.float64)[test_indices][test_eligible]
    mu0_test, mu1_test = _arm_predictions(
        train_features,
        train_outcomes,
        train_actions,
        test_features,
        alpha=config.ridge_alpha,
    )
    test_xi_flat = _sent_xi(test_outcomes, test_probabilities, mu0_test, mu1_test)
    test_xi_grid = np.zeros_like(test_eligible, dtype=np.float64)
    test_xi_grid[test_eligible] = test_xi_flat
    test_mask = test_eligible & (episodes.actions[test_indices] == 1)
    return FoldData(
        train_indices=train_indices,
        test_indices=test_indices,
        outcome_mean=outcome_mean,
        outcome_scale=outcome_scale,
        credit_targets=credit_targets,
        credit_mask=credit_mask,
        train_xi=eligible_xi_grid[train_sent],
        test_xi=test_xi_grid[test_mask],
        test_mask=test_mask,
    )


def _affine(predictions: FloatArray, targets: FloatArray) -> tuple[float, float]:
    design = np.column_stack([predictions, np.ones_like(predictions)])
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def _attribution(name: str, model: Any, dataset: Any) -> FloatArray:
    if name == "docm_credit":
        return np.asarray(model.predict_logged_credit(dataset), dtype=np.float64)
    if name == "docm_prefix":
        return np.asarray(np.diff(model.predict_prefix_values(dataset), axis=-1))
    return np.asarray(np.diff(model.predict_outcome_head_prefix_values(dataset), axis=-1))


def train_seed(
    episodes: Episodes,
    folds: list[FoldData],
    config: E10HSDayPhase2Config,
    seed: int,
) -> dict[str, Any]:
    from longfeedback.models import (
        DelayedOutcomeCreditModel,
        EncoderArchitecture,
        SequenceDataset,
        TrainingSettings,
        variant_loss_weights,
    )

    architecture = EncoderArchitecture(**config.model.model_dump())
    training = TrainingSettings(**config.training.model_dump())
    errors: dict[str, list[FloatArray]] = {name: [] for name in VARIANTS}
    predictions: dict[str, list[FloatArray]] = {name: [] for name in VARIANTS}
    outcome_predictions: dict[str, list[FloatArray]] = {name: [] for name in VARIANTS}
    xi_rows: list[FloatArray] = []
    decision_users: list[StringArray] = []
    outcome_rows: list[FloatArray] = []
    outcome_users: list[StringArray] = []
    parameter_counts: dict[str, int] = {}

    for labels in folds:
        train = labels.train_indices
        test = labels.test_indices
        standardized_train = (
            episodes.terminal_scores[train] - labels.outcome_mean
        ) / labels.outcome_scale
        standardized_test = (
            episodes.terminal_scores[test] - labels.outcome_mean
        ) / labels.outcome_scale
        train_dataset = SequenceDataset(
            observations=episodes.observations[train],
            actions=episodes.actions[train],
            responses=np.zeros_like(episodes.actions[train], dtype=np.float64),
            outcomes=standardized_train,
            credit_targets=labels.credit_targets[train],
            credit_mask=labels.credit_mask[train],
            credit_se=np.ones_like(labels.credit_targets[train]),
        )
        test_dataset = SequenceDataset(
            observations=episodes.observations[test],
            actions=episodes.actions[test],
            responses=np.zeros_like(episodes.actions[test], dtype=np.float64),
            outcomes=standardized_test,
        )
        for name in VARIANTS:
            model = DelayedOutcomeCreditModel(
                observation_dim=episodes.observations.shape[-1],
                n_actions=2,
                horizon=config.episode_slots,
                reference_action=0,
                architecture=architecture,
                loss_weights=variant_loss_weights(name),
                outcome_type="continuous",
                seed=seed,
            )
            model.fit(train_dataset, training=training)
            parameter_counts[name] = model.parameter_count()
            predicted_terminal = (
                model.predict_outcome(test_dataset) * labels.outcome_scale + labels.outcome_mean
            )
            outcome_predictions[name].append(predicted_terminal)
            raw_train = _attribution(name, model, train_dataset)
            raw_test = _attribution(name, model, test_dataset)
            scale, shift = _affine(raw_train[labels.credit_mask[train]], labels.train_xi)
            calibrated = scale * raw_test[labels.test_mask] + shift
            predictions[name].append(calibrated)
            errors[name].append(np.square(calibrated - labels.test_xi))
        xi_rows.append(labels.test_xi)
        users_grid = np.repeat(episodes.users[test][:, None], config.episode_slots, axis=1)
        decision_users.append(users_grid[labels.test_mask])
        outcome_rows.append(episodes.terminal_scores[test])
        outcome_users.append(episodes.users[test])

    return {
        "errors": {name: np.concatenate(rows) for name, rows in errors.items()},
        "predictions": {name: np.concatenate(rows) for name, rows in predictions.items()},
        "outcome_predictions": {
            name: np.concatenate(rows) for name, rows in outcome_predictions.items()
        },
        "xi": np.concatenate(xi_rows),
        "decision_users": np.concatenate(decision_users),
        "outcomes": np.concatenate(outcome_rows),
        "outcome_users": np.concatenate(outcome_users),
        "parameter_counts": parameter_counts,
    }


def run_phase2(config: E10HSDayPhase2Config, *, output_dir: Path | None = None) -> ExperimentResult:
    started = time.perf_counter()
    repository = _repository_root()
    resolved_output = (output_dir or config.output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    episodes = build_episodes(
        config.processed_dir / config.decisions_filename, slots=config.episode_slots
    )
    arrays = _decision_arrays(episodes)
    folds = [
        prepare_fold(episodes, arrays, fold, config) for fold in range(config.evaluation_folds)
    ]
    seed_results: dict[int, dict[str, Any]] = {}
    for seed in config.seeds:
        seed_results[seed] = train_seed(episodes, folds, config, seed)

    per_seed: dict[str, Any] = {}
    positive = {"outcome": 0, "prefix": 0}
    negative = {"outcome": 0, "prefix": 0}
    for seed, result in seed_results.items():
        comparisons: dict[str, Any] = {}
        for baseline in ("outcome", "prefix"):
            name = f"docm_{baseline}"
            gap = result["errors"][name] - result["errors"]["docm_credit"]
            summary = _cluster_mean_ci(
                gap,
                result["decision_users"],
                resamples=config.bootstrap_resamples,
                confidence=config.bootstrap_confidence,
                seed=config.bootstrap_seed + 1000 * seed + (1 if baseline == "outcome" else 2),
            )
            positive[baseline] += int(summary["estimate"] > 0.0)
            negative[baseline] += int(summary["ci_high"] < 0.0)
            comparisons[f"{baseline}_minus_credit"] = summary
        rmse = {
            name: float(
                np.sqrt(
                    np.mean(np.square(result["outcome_predictions"][name] - result["outcomes"]))
                )
            )
            for name in VARIANTS
        }
        per_seed[str(seed)] = {
            "paired_mse_gaps": comparisons,
            "credit_mse": {name: float(np.mean(result["errors"][name])) for name in VARIANTS},
            "terminal_rmse": rmse,
        }

    primary_seed = config.seeds[0]
    primary = per_seed[str(primary_seed)]
    primary_pass = {
        baseline: bool(
            primary["paired_mse_gaps"][f"{baseline}_minus_credit"]["estimate"] > 0.0
            and primary["paired_mse_gaps"][f"{baseline}_minus_credit"]["ci_low"] > 0.0
        )
        for baseline in ("outcome", "prefix")
    }
    primary_rmse = primary["terminal_rmse"]
    rmse_difference = abs(primary_rmse["docm_credit"] - primary_rmse["docm_outcome"])
    outcome_matched = rmse_difference <= config.outcome_rmse_tolerance
    robust = {
        baseline: bool(positive[baseline] >= config.min_positive_seeds and negative[baseline] == 0)
        for baseline in ("outcome", "prefix")
    }
    counts = seed_results[primary_seed]["parameter_counts"]
    capacity_matched = len(set(counts.values())) == 1
    claim_supported = bool(
        capacity_matched and outcome_matched and all(primary_pass.values()) and all(robust.values())
    )
    metrics: dict[str, Any] = {
        "experiment": "e10_hs_day_phase2",
        "status": "pass" if claim_supported else "fail",
        "data": {
            "complete_participant_days": episodes.count,
            "participants": len(np.unique(episodes.users)),
            "held_out_graded_sent_decisions": int(
                sum(np.sum(labels.test_mask) for labels in folds)
            ),
        },
        "model": {"parameter_counts": counts, "capacity_matched": capacity_matched},
        "per_seed": per_seed,
        "e10_hs_day_phase2_decision": {
            "outcome_rmse_difference": rmse_difference,
            "outcome_quality_matched": outcome_matched,
            "primary_seed": primary_seed,
            "primary_positive_ci": primary_pass,
            "positive_gap_seeds": positive,
            "negative_ci_seeds": negative,
            "robust": robust,
            "core_claim_supported_on_real_data": claim_supported,
        },
        "claim_scope": (
            "group-level randomized excursion-effect recovery from a delayed terminal "
            "outcome in a real behavioral sequence; not unit counterfactuals, policy "
            "improvement, conversational transfer, or long-term welfare"
        ),
    }
    seed0 = seed_results[primary_seed]
    rows = [
        {
            "user_id": str(seed0["decision_users"][index]),
            "held_out_xi": float(seed0["xi"][index]),
            **{f"calibrated_{name}": float(seed0["predictions"][name][index]) for name in VARIANTS},
        }
        for index in range(len(seed0["xi"]))
    ]
    predictions_path = resolved_output / config.predictions_filename
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics_path = resolved_output / config.metrics_filename
    manifest_path = resolved_output / config.manifest_filename
    metrics["runtime_seconds"] = time.perf_counter() - started
    scientific = {key: value for key, value in metrics.items() if key != "runtime_seconds"}
    metrics["scientific_metrics_sha256"] = sha256_json(scientific)
    write_metrics_json(metrics, metrics_path)
    manifest = build_run_manifest(
        repository=repository,
        resolved_config=config.model_dump(mode="json"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/e10_hs_day_phase2.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    result = run_phase2(load_config(arguments.config), output_dir=arguments.output_dir)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
