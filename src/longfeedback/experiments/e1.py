"""E1: real-log delayed-outcome prediction on prepared conversation events.

For every assistant turn with at least one future user turn, rule-based
labelers (frozen before modeling) read the *future* user messages to decide
whether the turn failed — an explicit correction/negative signal or a
repetition of the original request. Model inputs are built from past turns
only; trivial length baselines are included to expose leakage or degeneracy
(design doc §6.7, §9.1).

Labels are behavioral proxies. This experiment supports predictive claims
only — never causal ones (see docs/scientific_contract.md).
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from longfeedback.baselines import MeanOutcomeBaseline, RidgeBaseline
from longfeedback.config import E1Config, dump_resolved_config
from longfeedback.evaluation import (
    auroc,
    average_precision,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    write_metrics_json,
)
from longfeedback.experiments.manifest import build_run_manifest, sha256_json
from longfeedback.experiments.types import ExperimentResult
from longfeedback.models import (
    DelayedOutcomeCreditModel,
    EncoderArchitecture,
    SequenceDataset,
    TrainingSettings,
    variant_loss_weights,
)
from longfeedback.outcomes.rules import (
    LABELER_VERSION,
    is_repetition,
    negative_signal,
    positive_signal,
)

FloatArray = npt.NDArray[np.float64]

_CODE_FENCE = "```"
_STEP_FEATURES = 11


@dataclass(frozen=True)
class ConversationSteps:
    trajectory_id: str
    split: str
    user_messages: tuple[str, ...]
    assistant_messages: tuple[str, ...]


@dataclass(frozen=True)
class RealLogExample:
    trajectory_id: str
    turn_index: int
    split: str
    fail_next: float
    fail_any: float
    window_features: FloatArray  # [window, features]
    trivial_features: FloatArray  # [3]


def load_conversations(events_path: Path) -> list[ConversationSteps]:
    """Group the flattened event table back into ordered conversations."""

    import pyarrow.parquet as pq

    table = pq.read_table(
        events_path,
        columns=["trajectory_id", "step_index", "event_type", "content", "split"],
    )
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"user": {}, "assistant": {}, "split": ""}
    )
    for row in table.to_pylist():
        entry = grouped[row["trajectory_id"]]
        entry["split"] = row["split"]
        role = "user" if row["event_type"] == "observation" else "assistant"
        entry[role][row["step_index"]] = row["content"]

    conversations = []
    for trajectory_id in sorted(grouped):
        entry = grouped[trajectory_id]
        steps = sorted(entry["user"])
        if steps != sorted(entry["assistant"]) or steps != list(range(len(steps))):
            raise ValueError(f"malformed step structure in {trajectory_id}")
        conversations.append(
            ConversationSteps(
                trajectory_id=trajectory_id,
                split=str(entry["split"]),
                user_messages=tuple(entry["user"][step] for step in steps),
                assistant_messages=tuple(entry["assistant"][step] for step in steps),
            )
        )
    return conversations


def _text_features(text: str) -> tuple[float, float, float, float]:
    words = text.split()
    return (
        math.log1p(len(text)) / 10.0,
        min(len(words), 500) / 100.0,
        min(text.count("?"), 5) / 5.0,
        1.0 if _CODE_FENCE in text else 0.0,
    )


def _step_features(
    user_message: str,
    assistant_message: str,
    previous_user_message: str | None,
    *,
    repetition_threshold: float,
) -> list[float]:
    user_stats = _text_features(user_message)
    assistant_stats = _text_features(assistant_message)
    repeated = previous_user_message is not None and is_repetition(
        user_message, previous_user_message, threshold=repetition_threshold
    )
    return [
        1.0,  # presence flag; zero rows are padding
        *user_stats,
        1.0 if negative_signal(user_message) else 0.0,
        1.0 if positive_signal(user_message) else 0.0,
        1.0 if repeated else 0.0,
        assistant_stats[0],
        assistant_stats[1],
        assistant_stats[3],
    ]


def build_examples(
    conversation: ConversationSteps,
    *,
    window: int,
    repetition_threshold: float,
) -> list[RealLogExample]:
    """Build one labeled example per assistant turn with future evidence.

    Labels read user messages strictly after turn ``t``; features read
    messages up to and including turn ``t``.
    """

    steps = len(conversation.user_messages)
    step_rows = [
        _step_features(
            conversation.user_messages[index],
            conversation.assistant_messages[index],
            conversation.user_messages[index - 1] if index > 0 else None,
            repetition_threshold=repetition_threshold,
        )
        for index in range(steps)
    ]
    examples = []
    for turn_index in range(steps - 1):
        failure_signals = [
            negative_signal(conversation.user_messages[future])
            or is_repetition(
                conversation.user_messages[future],
                conversation.user_messages[turn_index],
                threshold=repetition_threshold,
            )
            for future in range(turn_index + 1, steps)
        ]
        window_rows = step_rows[max(0, turn_index + 1 - window) : turn_index + 1]
        padded = [[0.0] * _STEP_FEATURES] * (window - len(window_rows)) + window_rows
        trivial = np.asarray(
            [
                (turn_index + 1) / 10.0,
                math.log1p(len(conversation.assistant_messages[turn_index])) / 10.0,
                math.log1p(len(conversation.user_messages[turn_index])) / 10.0,
            ],
            dtype=np.float64,
        )
        examples.append(
            RealLogExample(
                trajectory_id=conversation.trajectory_id,
                turn_index=turn_index,
                split=conversation.split,
                fail_next=float(failure_signals[0]),
                fail_any=float(any(failure_signals)),
                window_features=np.asarray(padded, dtype=np.float64),
                trivial_features=trivial,
            )
        )
    return examples


def _subsample(examples: list[RealLogExample], limit: int, seed: int) -> list[RealLogExample]:
    if len(examples) <= limit:
        return examples
    order = np.random.default_rng(seed).permutation(len(examples))[:limit]
    return [examples[int(index)] for index in sorted(order)]


def _sequence_dataset(examples: list[RealLogExample], label: str) -> SequenceDataset:
    features = np.stack([example.window_features for example in examples])
    episodes, window, _ = features.shape
    return SequenceDataset(
        observations=features,
        actions=np.zeros((episodes, window), dtype=np.int64),
        responses=np.zeros((episodes, window), dtype=np.float64),
        outcomes=np.asarray([getattr(example, label) for example in examples], dtype=np.float64),
    )


def _probability_metrics(labels: FloatArray, scores: FloatArray) -> dict[str, float]:
    clipped = np.clip(scores, 0.0, 1.0)
    return {
        "auroc": auroc(labels, scores),
        "auprc": average_precision(labels, scores),
        "brier": brier_score(labels, clipped),
        "ece": expected_calibration_error(labels, clipped),
        "nll": negative_log_likelihood(labels, clipped),
    }


def _repository_root() -> Path:
    working_directory = Path.cwd().resolve()
    for candidate in (working_directory, *working_directory.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and (candidate / "src" / "longfeedback").is_dir():
            return candidate
    return working_directory


def run_e1(config: E1Config, *, output_dir: Path | None = None) -> ExperimentResult:
    """Run the real-log delayed-outcome prediction experiment."""

    started = time.perf_counter()
    repository = _repository_root()
    resolved_output = output_dir or config.output_dir
    if not resolved_output.is_absolute():
        resolved_output = repository / resolved_output
    resolved_output.mkdir(parents=True, exist_ok=True)

    events_path = config.processed_dir / config.events_filename
    if not events_path.is_absolute():
        events_path = repository / events_path
    conversations = load_conversations(events_path)

    examples: list[RealLogExample] = []
    for conversation in conversations:
        examples.extend(
            build_examples(
                conversation,
                window=config.window,
                repetition_threshold=config.repetition_threshold,
            )
        )
    train = _subsample(
        [example for example in examples if example.split == "train"],
        config.max_train_examples,
        config.seed,
    )
    test = _subsample(
        [example for example in examples if example.split == "test"],
        config.max_eval_examples,
        config.seed + 1,
    )
    if not train or not test:
        raise ValueError("prepared events yielded no train or test examples")

    label_metrics: dict[str, Any] = {}
    predictions_rows: list[list[Any]] = []
    for label in ("fail_next", "fail_any"):
        train_labels = np.asarray([getattr(example, label) for example in train])
        test_labels = np.asarray([getattr(example, label) for example in test])

        flat_train = np.stack([example.window_features.reshape(-1) for example in train])
        flat_test = np.stack([example.window_features.reshape(-1) for example in test])
        trivial_train = np.stack([example.trivial_features for example in train])
        trivial_test = np.stack([example.trivial_features for example in test])

        base = MeanOutcomeBaseline().fit(trivial_train, train_labels)
        trivial_model = RidgeBaseline(alpha=config.ridge_alpha).fit(trivial_train, train_labels)
        full_model = RidgeBaseline(alpha=config.ridge_alpha).fit(flat_train, train_labels)

        sequence_model = DelayedOutcomeCreditModel(
            observation_dim=_STEP_FEATURES,
            n_actions=2,
            horizon=config.window,
            architecture=EncoderArchitecture(
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
            ),
            loss_weights=variant_loss_weights("docm_outcome"),
            seed=config.seed,
        )
        sequence_model.fit(
            _sequence_dataset(train, label),
            training=TrainingSettings(
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
            ),
        )
        model_scores = {
            "base_rate": base.predict(trivial_test),
            "trivial_length_ridge": trivial_model.predict(trivial_test),
            "full_feature_ridge": full_model.predict(flat_test),
            "sequence_transformer": sequence_model.predict_outcome_probability(
                _sequence_dataset(test, label)
            ),
        }
        label_metrics[label] = {
            "prevalence_train": float(np.mean(train_labels)),
            "prevalence_test": float(np.mean(test_labels)),
            "models": {
                name: _probability_metrics(test_labels, scores)
                for name, scores in model_scores.items()
            },
        }
        for row, example in enumerate(test):
            predictions_rows.append(
                [
                    label,
                    example.trajectory_id,
                    example.turn_index,
                    getattr(example, label),
                    model_scores["trivial_length_ridge"][row],
                    model_scores["full_feature_ridge"][row],
                    model_scores["sequence_transformer"][row],
                ]
            )

    primary = label_metrics["fail_next"]["models"]
    best_informed = max(
        primary["full_feature_ridge"]["auroc"], primary["sequence_transformer"]["auroc"]
    )
    trivial_auroc = primary["trivial_length_ridge"]["auroc"]
    decision = {
        "best_informed_auroc": best_informed,
        "trivial_length_auroc": trivial_auroc,
        "margin_over_trivial": best_informed - trivial_auroc,
        "learnable_above_trivial": bool(
            best_informed >= config.min_auroc
            and best_informed - trivial_auroc >= config.auroc_margin_over_trivial
        ),
    }

    elapsed = time.perf_counter() - started
    metrics: dict[str, Any] = {
        "experiment": "e1",
        "status": "pass" if decision["learnable_above_trivial"] else "fail",
        "seed": config.seed,
        "runtime_seconds": elapsed,
        "labeler_version": LABELER_VERSION,
        "data": {
            "conversations": len(conversations),
            "examples_total": len(examples),
            "examples_train": len(train),
            "examples_test": len(test),
            "window": config.window,
        },
        "labels": label_metrics,
        "e1_decision": decision,
        "claims": {
            "scope": (
                "predictive only: labels are rule-based behavioral proxies from future "
                "user turns; no causal or satisfaction claims"
            )
        },
    }

    metrics_path = resolved_output / config.metrics_filename
    predictions_path = resolved_output / config.predictions_filename
    manifest_path = resolved_output / config.manifest_filename
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "trajectory_id",
                "turn_index",
                "observed",
                "trivial_length_ridge",
                "full_feature_ridge",
                "sequence_transformer",
            ]
        )
        writer.writerows(predictions_rows)

    elapsed = time.perf_counter() - started
    metrics["runtime_seconds"] = elapsed
    scientific_metrics = {key: value for key, value in metrics.items() if key != "runtime_seconds"}
    metrics["scientific_metrics_sha256"] = sha256_json(scientific_metrics)
    write_metrics_json(metrics, metrics_path)
    manifest = build_run_manifest(
        repository=repository,
        resolved_config=dump_resolved_config(config),
        artifacts={"metrics": metrics_path.name, "predictions": predictions_path.name},
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
            "manifest": manifest_path,
        },
    )
