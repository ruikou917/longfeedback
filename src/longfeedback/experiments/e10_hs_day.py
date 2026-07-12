"""E10-HS-Day: terminal daily credit from randomized HeartSteps decisions."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field

from longfeedback.evaluation import write_metrics_json
from longfeedback.experiments.manifest import build_run_manifest, sha256_json
from longfeedback.experiments.types import ExperimentResult

FloatArray = npt.NDArray[np.float64]
StringArray = npt.NDArray[np.str_]
BoolArray = npt.NDArray[np.bool_]


class E10HSDayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["e10_hs_day"] = "e10_hs_day"
    processed_dir: Path = Path("data/processed/heartsteps")
    decisions_filename: str = "decisions.parquet"
    output_dir: Path = Path("artifacts/e10_hs_day")
    decisions_per_day: int = Field(default=5, ge=2)
    min_abs_effect: float = Field(default=0.01, gt=0.0)
    max_median_group_se: float = Field(default=0.15, gt=0.0)
    min_detected_groups: int = Field(default=2, ge=1)
    bootstrap_resamples: int = Field(default=2_000, ge=100)
    bootstrap_confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    bootstrap_seed: int = 0
    metrics_filename: str = "metrics.json"
    predictions_filename: str = "group_effects.csv"
    manifest_filename: str = "run_manifest.json"


def load_e10_hs_day_config(path: Path) -> E10HSDayConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("E10-HS-Day config must be a YAML mapping")
    return E10HSDayConfig.model_validate(raw)


def _repository_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/longfeedback").is_dir():
            return candidate
    return cwd


def _cluster_effect_draws(
    outcome: FloatArray,
    action: BoolArray,
    users: StringArray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, FloatArray]:
    unique, inverse = np.unique(users, return_inverse=True)
    sufficient = np.zeros((len(unique), 4), dtype=np.float64)
    treated = action.astype(np.float64)
    reference = 1.0 - treated
    np.add.at(sufficient[:, 0], inverse, outcome * treated)
    np.add.at(sufficient[:, 1], inverse, treated)
    np.add.at(sufficient[:, 2], inverse, outcome * reference)
    np.add.at(sufficient[:, 3], inverse, reference)
    total = sufficient.sum(axis=0)
    estimate = float(total[0] / total[1] - total[2] / total[3])
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.integers(0, len(unique), size=len(unique))
        aggregate = sufficient[selected].sum(axis=0)
        if aggregate[1] == 0.0 or aggregate[3] == 0.0:
            draws[index] = np.nan
        else:
            draws[index] = aggregate[0] / aggregate[1] - aggregate[2] / aggregate[3]
    draws = draws[np.isfinite(draws)]
    if len(draws) < resamples * 0.95:
        raise ValueError("too many cluster bootstrap draws lacked treatment overlap")
    return estimate, draws


def _effect_summary(
    outcome: FloatArray,
    action: BoolArray,
    users: StringArray,
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> dict[str, float | int]:
    estimate, draws = _cluster_effect_draws(outcome, action, users, resamples=resamples, seed=seed)
    alpha = (1.0 - confidence) / 2.0
    return {
        "effect": estimate,
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "decisions": len(outcome),
        "users": len(np.unique(users)),
    }


def build_terminal_day_examples(
    decisions_path: Path, *, decisions_per_day: int
) -> dict[str, npt.NDArray[Any]]:
    """Return one analysis row per available decision in complete pseudo-days."""

    columns = [
        "user_id",
        "study_day",
        "decision_number",
        "available",
        "randomized",
        "action",
        "home_work",
        "proximal_steps30",
    ]
    frame = pq.read_table(decisions_path, columns=columns).to_pydict()
    by_day: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (user, day) in enumerate(zip(frame["user_id"], frame["study_day"], strict=True)):
        by_day[(str(user), int(day))].append(index)

    rows: dict[str, list[Any]] = defaultdict(list)
    complete_days = 0
    for (user, day), indices in sorted(by_day.items()):
        if len(indices) != decisions_per_day:
            continue
        ordered = sorted(indices, key=lambda index: int(frame["decision_number"][index]))
        complete_days += 1
        terminal = float(
            np.mean(
                np.log1p(
                    np.asarray([frame["proximal_steps30"][index] for index in ordered], dtype=float)
                )
            )
        )
        for position, index in enumerate(ordered, start=1):
            if not bool(frame["available"][index]) or not bool(frame["randomized"][index]):
                continue
            rows["user_id"].append(user)
            rows["study_day"].append(day)
            rows["position"].append(position)
            rows["home_work"].append(int(frame["home_work"][index]))
            rows["action"].append(bool(frame["action"][index]))
            rows["terminal_score"].append(terminal)
    rows["complete_days"] = [complete_days]
    return {key: np.asarray(value) for key, value in rows.items()}


def run_e10_hs_day(config: E10HSDayConfig, *, output_dir: Path | None = None) -> ExperimentResult:
    started = time.perf_counter()
    repository = _repository_root()
    resolved_output = (output_dir or config.output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    decisions_path = config.processed_dir / config.decisions_filename
    if not decisions_path.is_file():
        raise FileNotFoundError(f"prepared HeartSteps decisions not found at {decisions_path}")
    examples = build_terminal_day_examples(
        decisions_path, decisions_per_day=config.decisions_per_day
    )
    outcome = np.asarray(examples["terminal_score"], dtype=np.float64)
    action = np.asarray(examples["action"], dtype=np.bool_)
    users = np.asarray(examples["user_id"], dtype=np.str_)
    positions = np.asarray(examples["position"], dtype=np.int64)
    home_work = np.asarray(examples["home_work"], dtype=np.int64)
    global_summary = _effect_summary(
        outcome,
        action,
        users,
        resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
        confidence=config.bootstrap_confidence,
    )
    group_count = config.decisions_per_day * 2
    simultaneous_confidence = 1.0 - (1.0 - config.bootstrap_confidence) / group_count
    groups: list[dict[str, Any]] = []
    for position in range(1, config.decisions_per_day + 1):
        for context in (0, 1):
            selected = (positions == position) & (home_work == context)
            summary = _effect_summary(
                outcome[selected],
                action[selected],
                users[selected],
                resamples=config.bootstrap_resamples,
                seed=config.bootstrap_seed + 100 * position + context,
                confidence=simultaneous_confidence,
            )
            detected = bool(
                abs(float(summary["effect"])) >= config.min_abs_effect
                and (float(summary["ci_low"]) > 0.0 or float(summary["ci_high"]) < 0.0)
            )
            groups.append(
                {
                    "position": position,
                    "home_work": context,
                    **summary,
                    "relevant_detected_effect": detected,
                }
            )
    global_detected = bool(
        abs(float(global_summary["effect"])) >= config.min_abs_effect
        and (float(global_summary["ci_low"]) > 0.0 or float(global_summary["ci_high"]) < 0.0)
    )
    detected_groups = sum(bool(group["relevant_detected_effect"]) for group in groups)
    median_group_se = float(np.median([group["bootstrap_se"] for group in groups]))
    precision_adequate = median_group_se <= config.max_median_group_se
    signal_gate_pass = bool(
        precision_adequate and (global_detected or detected_groups >= config.min_detected_groups)
    )
    metrics: dict[str, Any] = {
        "experiment": "e10_hs_day",
        "status": "phase_1_pass" if signal_gate_pass else "completed_null",
        "estimand": (
            "randomized send-vs-no-send effect on terminal five-decision daily "
            "activity score under the 0.6/0.4 continuation policy"
        ),
        "data": {
            "complete_participant_days": int(examples["complete_days"][0]),
            "eligible_randomized_decisions": len(outcome),
            "participants": len(np.unique(users)),
            "terminal_score_mean": float(np.mean(outcome)),
            "terminal_score_std": float(np.std(outcome, ddof=1)),
            "observed_action_rate": float(np.mean(action)),
        },
        "global_effect": global_summary,
        "group_effects": groups,
        "e10_hs_day_decision": {
            "signal_gate_pass": signal_gate_pass,
            "phase_2_authorized": signal_gate_pass,
            "global_relevant_detected_effect": global_detected,
            "detected_groups": detected_groups,
            "median_group_bootstrap_se": median_group_se,
            "precision_adequate": precision_adequate,
        },
        "claim_scope": (
            "real randomized multi-step behavioral benchmark; constructed daily "
            "terminal score, not conversational transfer or long-term welfare"
        ),
    }
    predictions_path = resolved_output / config.predictions_filename
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    metrics_path = resolved_output / config.metrics_filename
    manifest_path = resolved_output / config.manifest_filename
    metrics["runtime_seconds"] = time.perf_counter() - started
    scientific = {key: value for key, value in metrics.items() if key != "runtime_seconds"}
    metrics["scientific_metrics_sha256"] = sha256_json(scientific)
    write_metrics_json(metrics, metrics_path)
    manifest = build_run_manifest(
        repository=repository,
        resolved_config=config.model_dump(mode="json"),
        artifacts={"metrics": metrics_path.name, "group_effects": predictions_path.name},
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return ExperimentResult(
        metrics=metrics,
        output_dir=resolved_output,
        artifacts={
            "metrics": metrics_path,
            "group_effects": predictions_path,
            "manifest": manifest_path,
        },
    )


def main() -> None:
    """Run E10-HS-Day from its frozen YAML configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/e10_hs_day.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    result = run_e10_hs_day(
        load_e10_hs_day_config(arguments.config), output_dir=arguments.output_dir
    )
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
