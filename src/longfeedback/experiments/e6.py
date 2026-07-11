"""E6: the randomized bridge -- confounded-log bias against real randomization.

``log_random_*`` rows in the prepared KuaiRand events carry a genuinely
uniform exposure policy; ``log_standard_*`` rows are production-confounded,
exactly like WildChat/LMSYS. This experiment asks two questions E1 cannot
license an answer to, because E1's real logs are never randomized:

1. **Bias measurement** (no model): does a per-video engagement rate
   estimated from the confounded log systematically differ from the video's
   *true* population engagement rate, measured directly on the randomized
   log? Two videos with the same confounded-log rate can have very different
   true rates once the production recommender's targeting is removed. Both
   rates are direct empirical averages over two disjoint, already-collected
   logging populations, so there is no train/eval leakage to guard against.

2. **Feature adjustment** (a model, evaluated honestly): does a model that
   conditions on real confounders (user and video content features, trained
   on the confounded log) predict *better* on genuinely random exposures than
   the naive per-video rate from (1)? This is a narrower, different claim
   than the project's core sequential credit-assignment thesis (Gate A/B) --
   KuaiRand impressions are single-step, so there is no delayed-credit
   problem here, only a selection-bias/confounding-adjustment problem. See
   docs/scientific_contract.md ("E6 acceptance contract") for the full scope
   statement.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from longfeedback.baselines import MeanOutcomeBaseline, RidgeBaseline
from longfeedback.config import E6Config, dump_resolved_config
from longfeedback.evaluation import (
    auroc,
    brier_score,
    pearson_correlation,
    rmse,
    spearman_correlation,
    write_metrics_json,
)
from longfeedback.experiments.manifest import build_run_manifest, sha256_json
from longfeedback.experiments.types import ExperimentResult

FloatArray = npt.NDArray[np.float64]

_HASH_MODULUS = 997


@dataclass
class _Impression:
    user_id: str | None = None
    video_id: str | None = None
    engaged: bool | None = None
    logging_policy: str | None = None


@dataclass(frozen=True)
class _Example:
    user_id: str
    video_id: str
    features: FloatArray
    label: float


def _repository_root() -> Path:
    working_directory = Path.cwd().resolve()
    for candidate in (working_directory, *working_directory.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and (candidate / "src" / "longfeedback").is_dir():
            return candidate
    return working_directory


def load_impressions(events_path: Path) -> dict[str, _Impression]:
    """Reconstruct one impression per trajectory from the prepared events table."""

    table = pq.read_table(
        events_path,
        columns=["trajectory_id", "event_type", "logging_policy", "payload_json"],
    )
    impressions: dict[str, _Impression] = defaultdict(_Impression)
    for trajectory_id, event_type, logging_policy, payload_json in zip(
        table.column("trajectory_id").to_pylist(),
        table.column("event_type").to_pylist(),
        table.column("logging_policy").to_pylist(),
        table.column("payload_json").to_pylist(),
        strict=True,
    ):
        impression = impressions[trajectory_id]
        impression.logging_policy = logging_policy
        if event_type == "observation":
            impression.user_id = str(json.loads(payload_json)["user_id"])
        elif event_type == "action":
            impression.video_id = str(json.loads(payload_json)["video_id"])
        elif event_type == "outcome":
            impression.engaged = bool(json.loads(payload_json)["engaged"])
    return impressions


def _encode_value(raw: str) -> float:
    """Coerce one CSV cell to a float, deterministically hashing categoricals.

    A stable (not Python's randomized ``hash()``) but crude stand-in for real
    categorical encoding -- sufficient to test whether *any* conditioning on
    observable confounders helps, not a tuned production feature set.
    """

    try:
        return float(raw)
    except ValueError:
        digest = hashlib.sha256(raw.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % _HASH_MODULUS / _HASH_MODULUS


def load_feature_table(path: Path, *, id_column: str) -> dict[str, FloatArray]:
    """Load a KuaiRand side-feature CSV into per-id numeric feature vectors."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if not header or id_column not in header:
            raise ValueError(f"{path} is missing the {id_column!r} column")
        feature_columns = [column for column in header if column != id_column]
        table: dict[str, FloatArray] = {}
        for row in reader:
            table[row[id_column]] = np.asarray(
                [_encode_value(row[column]) for column in feature_columns], dtype=np.float64
            )
    return table


def build_examples(
    impressions: dict[str, _Impression],
    *,
    logging_policy: str,
    user_features: dict[str, FloatArray],
    video_features: dict[str, FloatArray],
) -> list[_Example]:
    """Join impressions with side features; drop rows missing either side."""

    examples: list[_Example] = []
    for impression in impressions.values():
        if (
            impression.logging_policy != logging_policy
            or impression.user_id is None
            or impression.video_id is None
            or impression.engaged is None
        ):
            continue
        user_vector = user_features.get(impression.user_id)
        video_vector = video_features.get(impression.video_id)
        if user_vector is None or video_vector is None:
            continue
        examples.append(
            _Example(
                user_id=impression.user_id,
                video_id=impression.video_id,
                features=np.concatenate([user_vector, video_vector]),
                label=float(impression.engaged),
            )
        )
    return examples


def per_video_engagement_rate(
    impressions: dict[str, _Impression], *, logging_policy: str, min_exposures: int
) -> dict[str, tuple[float, int]]:
    """Return {video_id: (engagement_rate, exposure_count)} for one logging policy."""

    totals: dict[str, int] = defaultdict(int)
    engaged: dict[str, int] = defaultdict(int)
    for impression in impressions.values():
        if impression.logging_policy != logging_policy or impression.video_id is None:
            continue
        totals[impression.video_id] += 1
        engaged[impression.video_id] += int(bool(impression.engaged))
    return {
        video_id: (engaged[video_id] / count, count)
        for video_id, count in totals.items()
        if count >= min_exposures
    }


def run_e6(config: E6Config, *, output_dir: Path | None = None) -> ExperimentResult:
    """Run the randomized-bridge confounding-bias comparison."""

    started = time.perf_counter()
    repository = _repository_root()
    resolved_output = output_dir or config.output_dir
    if not resolved_output.is_absolute():
        resolved_output = repository / resolved_output
    resolved_output.mkdir(parents=True, exist_ok=True)

    events_path = config.processed_dir / config.events_filename
    if not events_path.is_absolute():
        events_path = repository / events_path
    impressions = load_impressions(events_path)

    confounded = per_video_engagement_rate(
        impressions, logging_policy="standard", min_exposures=config.min_exposures_per_video
    )
    randomized = per_video_engagement_rate(
        impressions, logging_policy="random", min_exposures=config.min_exposures_per_video
    )
    common_videos = sorted(set(confounded) & set(randomized))
    if not common_videos:
        raise ValueError(
            "no video appears with enough exposures in both logging policies; "
            "lower min_exposures_per_video or process more rows in data prepare kuairand"
        )

    true_rate = np.asarray([randomized[video][0] for video in common_videos])
    confounded_rate = np.asarray([confounded[video][0] for video in common_videos])
    trivial_rate = np.full_like(true_rate, float(np.mean(true_rate)))

    bias = confounded_rate - true_rate
    mean_bias = float(np.mean(bias))
    mean_absolute_bias = float(np.mean(np.abs(bias)))
    decision: dict[str, Any] = {
        "videos_compared": len(common_videos),
        "mean_bias": mean_bias,
        "mean_absolute_bias": mean_absolute_bias,
        "rmse_confounded": rmse(true_rate, confounded_rate),
        "rmse_trivial_constant": rmse(true_rate, trivial_rate),
        "pearson_correlation": pearson_correlation(true_rate, confounded_rate),
        "spearman_correlation": spearman_correlation(true_rate, confounded_rate),
        "confounding_bias_detected": bool(mean_absolute_bias >= config.bias_detection_threshold),
        "confounded_log_rank_useful": bool(
            spearman_correlation(true_rate, confounded_rate) >= config.rank_correlation_threshold
        ),
    }
    decision["confounded_beats_trivial_calibration"] = bool(
        decision["rmse_confounded"] < decision["rmse_trivial_constant"]
    )
    if decision["confounding_bias_detected"] and decision["confounded_log_rank_useful"]:
        h6_verdict = "biased_but_rank_useful"
    elif decision["confounding_bias_detected"]:
        h6_verdict = "biased_and_uninformative"
    else:
        h6_verdict = "no_material_bias_detected"
    decision["hypothesis_h6_confounded_log_bias"] = h6_verdict
    decision["pass"] = bool(decision["confounding_bias_detected"])

    raw_dir = config.raw_dir if config.raw_dir.is_absolute() else repository / config.raw_dir
    user_features = load_feature_table(
        raw_dir / "data" / config.user_features_filename, id_column="user_id"
    )
    video_features = load_feature_table(
        raw_dir / "data" / config.video_features_filename, id_column="video_id"
    )
    train_examples = build_examples(
        impressions,
        logging_policy="standard",
        user_features=user_features,
        video_features=video_features,
    )
    eval_examples = build_examples(
        impressions,
        logging_policy="random",
        user_features=user_features,
        video_features=video_features,
    )
    if not train_examples or not eval_examples:
        raise ValueError(
            "no impression had both user and video side features available; "
            "check that raw_dir points at the downloaded KuaiRand-Pure snapshot"
        )

    train_features = np.stack([example.features for example in train_examples])
    train_labels = np.asarray([example.label for example in train_examples])
    eval_features = np.stack([example.features for example in eval_examples])
    eval_labels = np.asarray([example.label for example in eval_examples])
    global_mean = float(np.mean(train_labels))

    trivial_model = MeanOutcomeBaseline().fit(train_features, train_labels)
    ridge_model = RidgeBaseline(alpha=config.ridge_alpha).fit(train_features, train_labels)
    trivial_pred = np.clip(trivial_model.predict(eval_features), 0.0, 1.0)
    ridge_pred = np.clip(ridge_model.predict(eval_features), 0.0, 1.0)
    naive_video_pred = np.asarray(
        [confounded.get(example.video_id, (global_mean, 0))[0] for example in eval_examples]
    )

    predictions_by_name = {
        "trivial": trivial_pred,
        "naive_video_rate": naive_video_pred,
        "ridge_features": ridge_pred,
    }
    feature_adjustment: dict[str, Any] = {
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "feature_dim": int(train_features.shape[1]),
        "auroc": {name: auroc(eval_labels, pred) for name, pred in predictions_by_name.items()},
        "brier": {
            name: brier_score(eval_labels, pred) for name, pred in predictions_by_name.items()
        },
        "rmse": {name: rmse(eval_labels, pred) for name, pred in predictions_by_name.items()},
    }
    margin = config.feature_adjustment_improvement_margin
    ridge_beats_naive = bool(
        feature_adjustment["brier"]["ridge_features"]
        <= feature_adjustment["brier"]["naive_video_rate"] - margin
    )
    ridge_beats_trivial = bool(
        feature_adjustment["brier"]["ridge_features"]
        <= feature_adjustment["brier"]["trivial"] - margin
    )
    feature_adjustment["ridge_beats_naive_video_rate"] = ridge_beats_naive
    feature_adjustment["ridge_beats_trivial"] = ridge_beats_trivial
    if ridge_beats_naive and ridge_beats_trivial:
        h6b_verdict = "supported"
    elif ridge_beats_trivial:
        h6b_verdict = "partially_supported_beats_trivial_not_naive_video_rate"
    else:
        h6b_verdict = "refuted_in_this_environment"
    feature_adjustment["hypothesis_h6b_feature_adjustment_helps"] = h6b_verdict

    elapsed = time.perf_counter() - started
    metrics: dict[str, Any] = {
        "experiment": "e6",
        "status": "pass" if decision["pass"] else "fail",
        "seed": config.seed,
        "runtime_seconds": elapsed,
        "data": {
            "impressions_total": len(impressions),
            "distinct_videos_standard": len(confounded),
            "distinct_videos_random": len(randomized),
            "min_exposures_per_video": config.min_exposures_per_video,
        },
        "e6_decision": decision,
        "e6_feature_adjustment": feature_adjustment,
        "claims": {
            "scope": (
                "causal for the stated comparison only: the 'random' rate is an unbiased "
                "estimate of each video's population engagement rate under uniform "
                "exposure; the 'standard' rate reflects the production policy's targeting "
                "and is not a general claim about any other confounded log"
            ),
            "feature_adjustment_scope": (
                "a narrower claim than the project's core sequential credit-assignment "
                "thesis (Gate A/B): KuaiRand impressions are single-step, so this tests "
                "whether conditioning on real confounders beats naive log-averaging, not "
                "delayed/sequential credit assignment"
            ),
        },
    }

    metrics_path = resolved_output / config.metrics_filename
    predictions_path = resolved_output / config.predictions_filename
    feature_predictions_path = resolved_output / "feature_adjustment_predictions.csv"
    manifest_path = resolved_output / config.manifest_filename
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "video_id",
                "standard_rate",
                "standard_exposures",
                "random_rate",
                "random_exposures",
                "bias",
            ]
        )
        for video in common_videos:
            standard_rate, standard_count = confounded[video]
            random_rate, random_count = randomized[video]
            writer.writerow(
                [
                    video,
                    standard_rate,
                    standard_count,
                    random_rate,
                    random_count,
                    standard_rate - random_rate,
                ]
            )
    with feature_predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "user_id",
                "video_id",
                "label",
                "trivial_pred",
                "naive_video_rate_pred",
                "ridge_features_pred",
            ]
        )
        for example, trivial_value, naive_value, ridge_value in zip(
            eval_examples, trivial_pred, naive_video_pred, ridge_pred, strict=True
        ):
            writer.writerow(
                [
                    example.user_id,
                    example.video_id,
                    example.label,
                    trivial_value,
                    naive_value,
                    ridge_value,
                ]
            )

    elapsed = time.perf_counter() - started
    metrics["runtime_seconds"] = elapsed
    scientific_metrics = {key: value for key, value in metrics.items() if key != "runtime_seconds"}
    metrics["scientific_metrics_sha256"] = sha256_json(scientific_metrics)
    write_metrics_json(metrics, metrics_path)
    manifest = build_run_manifest(
        repository=repository,
        resolved_config=dump_resolved_config(config),
        artifacts={
            "metrics": metrics_path.name,
            "predictions": predictions_path.name,
            "feature_adjustment_predictions": feature_predictions_path.name,
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
            "predictions": predictions_path,
            "feature_adjustment_predictions": feature_predictions_path,
            "manifest": manifest_path,
        },
    )
