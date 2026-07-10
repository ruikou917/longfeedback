"""E0: a deterministic end-to-end scientific vertical slice."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from longfeedback.baselines import MeanOutcomeBaseline, RidgeBaseline
from longfeedback.config import E0Config, dump_resolved_config
from longfeedback.credit.oracle import (
    ContinuationMode,
    estimate_oracle_credit,
    exact_deterministic_credit,
)
from longfeedback.credit.rudder import RudderRedistributor
from longfeedback.evaluation import (
    pearson_correlation,
    plot_outcome_vs_credit,
    rmse,
    sign_accuracy,
    spearman_correlation,
    telescoping_residual,
    write_metrics_json,
)
from longfeedback.experiments.features import (
    action_sequence_features,
    deterministic_split,
    prefix_action_features,
)
from longfeedback.experiments.manifest import build_run_manifest, sha256_json
from longfeedback.experiments.types import ExperimentResult
from longfeedback.schema import (
    CensoringStatus,
    CreditContinuation,
    DelayedOutcomeExample,
    Event,
    EventType,
    ObservationRegime,
    OracleCreditExample,
    PropensityQuality,
    Trajectory,
)
from longfeedback.worlds import (
    Episode,
    FatigueAction,
    FatigueHabitConfig,
    FatigueHabitObservation,
    FatigueHabitState,
    FatigueHabitStepNoise,
    FatigueHabitWorld,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
E0Episode = Episode[
    FatigueHabitState,
    FatigueAction,
    FatigueHabitObservation,
    FatigueHabitStepNoise,
]


@dataclass(frozen=True, slots=True)
class EpsilonHelpfulPolicy:
    """A transparent behavior policy with exact released propensities."""

    epsilon: float

    def probabilities(self, observation: FatigueHabitObservation) -> tuple[float, ...]:
        if observation.fatigue is None:
            raise ValueError("EpsilonHelpfulPolicy requires oracle or noisy observability")
        actions = tuple(FatigueAction)
        base_action = FatigueAction.NOOP if observation.fatigue > 0.9 else FatigueAction.HELPFUL
        probabilities = [self.epsilon / len(actions) for _ in actions]
        probabilities[actions.index(base_action)] += 1.0 - self.epsilon
        return tuple(probabilities)

    def select_action(
        self,
        observation: FatigueHabitObservation,
        *,
        step_index: int,
        random_value: float,
    ) -> FatigueAction:
        del step_index
        cumulative = 0.0
        for action, probability in zip(
            FatigueAction,
            self.probabilities(observation),
            strict=True,
        ):
            cumulative += probability
            if random_value < cumulative:
                return action
        return FatigueAction.URGENT

    def log_probability(
        self,
        observation: FatigueHabitObservation,
        action: FatigueAction,
    ) -> float:
        probability = self.probabilities(observation)[tuple(FatigueAction).index(action)]
        return math.log(probability)


@dataclass(frozen=True, slots=True)
class E0Data:
    episodes: tuple[E0Episode, ...]
    trajectories: tuple[Trajectory, ...]
    outcome_examples: tuple[DelayedOutcomeExample, ...]
    action_ids: IntArray
    proxies: FloatArray
    utilities: FloatArray
    behavior_logprobs: FloatArray


def _repository_root() -> Path:
    """Find the invoking checkout without relying on editable-install paths."""

    working_directory = Path.cwd().resolve()
    for candidate in (working_directory, *working_directory.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and (candidate / "src" / "longfeedback").is_dir():
            return candidate
    return working_directory


def _world_from_config(config: E0Config) -> FatigueHabitWorld:
    settings = config.world
    return FatigueHabitWorld(
        FatigueHabitConfig(
            horizon=config.experiment.horizon,
            habit_decay=settings.habit_decay,
            fatigue_decay=settings.fatigue_decay,
            fatigue_response_penalty=settings.fatigue_response_penalty,
            fatigue_utility_cost=settings.fatigue_utility_cost,
            action_utility_cost=settings.action_utility_cost,
            proxy_threshold=settings.proxy_threshold,
        )
    )


def hand_derived_one_step_credit_check(
    world: FatigueHabitWorld,
) -> tuple[float, float, float]:
    """Compare oracle credit with an independent closed-form one-step result."""

    fixture_world = FatigueHabitWorld(replace(world.config, horizon=1))
    exogenous = fixture_world.sample_exogenous(0)
    episode = fixture_world.rollout((FatigueAction.NOOP,), exogenous)
    oracle = exact_deterministic_credit(
        fixture_world,
        episode,
        step_index=0,
        action=FatigueAction.HELPFUL,
        reference_action=FatigueAction.NOOP,
    ).credit_utility

    config = fixture_world.config
    helpful_index = fixture_world.action_space.index(FatigueAction.HELPFUL)
    noop_index = fixture_world.action_space.index(FatigueAction.NOOP)

    def sigmoid(value: float) -> float:
        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    common_logit = (
        config.base_response_logit - config.fatigue_response_penalty * config.initial_fatigue
    )
    response_difference = sigmoid(common_logit + config.response_lifts[helpful_index]) - sigmoid(
        common_logit + config.response_lifts[noop_index]
    )
    intensity_difference = (
        config.action_intensities[helpful_index] - config.action_intensities[noop_index]
    )
    fatigue_difference = config.fatigue_sensitivity * intensity_difference
    analytic = (
        config.habit_gain * response_difference
        - config.fatigue_utility_cost * fatigue_difference
        - config.action_utility_cost * intensity_difference
    )
    return oracle, analytic, abs(oracle - analytic)


def _episode_to_schema(
    episode: E0Episode,
    *,
    episode_index: int,
    policy: EpsilonHelpfulPolicy,
) -> tuple[Trajectory, DelayedOutcomeExample, tuple[float, ...]]:
    trajectory_id = f"e0-{episode_index:06d}"
    base_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=episode_index)
    events: list[Event] = []
    observations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    behavior_logprobs: list[float] = []

    for transition in episode.transitions:
        step = transition.step_index
        habituation = transition.observation.habituation
        observation_payload: dict[str, Any] = {
            "habit": transition.observation.habit,
            "fatigue": transition.observation.fatigue,
            "habituation": None if habituation is None else list(habituation),
            "last_response": transition.observation.last_response,
        }
        action_payload: dict[str, Any] = {"action": transition.action.value}
        response_payload: dict[str, Any] = {"response": transition.info_value("response")}
        log_probability = policy.log_probability(
            transition.observation,
            transition.action,
        )
        action_payload["behavior_logprob"] = log_probability
        behavior_logprobs.append(log_probability)
        observations.append(observation_payload)
        actions.append(action_payload)
        responses.append(response_payload)

        for offset, event_type, payload, suffix in (
            (0, EventType.OBSERVATION, observation_payload, "observation"),
            (1, EventType.ACTION, action_payload, "action"),
            (2, EventType.USER_RESPONSE, response_payload, "response"),
        ):
            events.append(
                Event(
                    trajectory_id=trajectory_id,
                    event_id=f"{trajectory_id}:{step:03d}:{suffix}",
                    event_time=base_time + timedelta(seconds=3 * step + offset),
                    step_index=step,
                    event_type=event_type,
                    payload=payload,
                    source="structural/fatigue_habit",
                    source_row_id=f"seed:{episode.seed}:step:{step}",
                    policy_id="epsilon_helpful",
                    policy_version="v1",
                )
            )

    outcome_time = base_time + timedelta(seconds=3 * len(episode.transitions))
    events.append(
        Event(
            trajectory_id=trajectory_id,
            event_id=f"{trajectory_id}:outcome",
            event_time=outcome_time,
            step_index=len(episode.transitions),
            event_type=EventType.OUTCOME,
            payload={
                "behavioral_proxy": episode.terminal_proxy,
                "true_utility": episode.terminal_utility,
            },
            source="structural/fatigue_habit",
            source_row_id=f"seed:{episode.seed}:outcome",
        )
    )
    trajectory = Trajectory(
        trajectory_id=trajectory_id,
        events=tuple(events),
        start_time=base_time,
        end_time=outcome_time,
        behavior_policy_id="epsilon_helpful:v1",
        observation_regime=ObservationRegime.ORACLE,
        censoring_status=CensoringStatus.OBSERVED,
        metadata={
            "world": "fatigue_habit",
            "seed": episode.seed if episode.seed is not None else -1,
        },
    )
    outcome_example = DelayedOutcomeExample(
        trajectory_id=trajectory_id,
        prefix_end_step=len(episode.transitions) - 1,
        observations=tuple(observations),
        actions=tuple(actions),
        responses=tuple(responses),
        terminal_outcome=episode.terminal_proxy,
        outcome_type="behavioral_proxy/fatigue_habit",
        outcome_observed_at=outcome_time,
        censored=False,
        behavior_logprobs=tuple(behavior_logprobs),
        propensity_quality=PropensityQuality.EXACT,
        provenance={
            "source": "structural/fatigue_habit",
            "true_utility": episode.terminal_utility,
        },
    )
    return trajectory, outcome_example, tuple(behavior_logprobs)


def generate_e0_data(config: E0Config, world: FatigueHabitWorld) -> E0Data:
    """Generate deterministic structural episodes and canonical records."""

    policy = EpsilonHelpfulPolicy(config.world.behavior_epsilon)
    episodes: list[E0Episode] = []
    trajectories: list[Trajectory] = []
    outcome_examples: list[DelayedOutcomeExample] = []
    action_rows: list[list[int]] = []
    logprob_rows: list[tuple[float, ...]] = []

    for episode_index in range(config.experiment.episodes):
        seed = config.experiment.seed + episode_index
        episode = world.rollout_policy(policy, world.sample_exogenous(seed))
        trajectory, example, logprobs = _episode_to_schema(
            episode,
            episode_index=episode_index,
            policy=policy,
        )
        episodes.append(episode)
        trajectories.append(trajectory)
        outcome_examples.append(example)
        action_rows.append([world.action_space.index(action) for action in episode.actions])
        logprob_rows.append(logprobs)

    return E0Data(
        episodes=tuple(episodes),
        trajectories=tuple(trajectories),
        outcome_examples=tuple(outcome_examples),
        action_ids=np.asarray(action_rows, dtype=np.int64),
        proxies=np.asarray([episode.terminal_proxy for episode in episodes], dtype=np.float64),
        utilities=np.asarray([episode.terminal_utility for episode in episodes], dtype=np.float64),
        behavior_logprobs=np.asarray(logprob_rows, dtype=np.float64),
    )


def _credit_design_row(
    episode: E0Episode,
    *,
    step_index: int,
    action_id: int,
    action_sequence: npt.NDArray[np.int64],
    n_actions: int,
) -> FloatArray:
    horizon = len(episode.transitions)
    step_action = np.zeros(horizon * n_actions, dtype=np.float64)
    step_action[step_index * n_actions + action_id] = 1.0
    full_actions = action_sequence_features(
        [action_sequence.tolist()],
        horizon=horizon,
        n_actions=n_actions,
    )[0]
    state = episode.state_before(step_index)
    state_features = np.asarray(
        [
            step_index / max(horizon - 1, 1),
            state.habit,
            state.fatigue,
            *state.habituation,
            state.last_response,
            state.cumulative_fatigue,
            state.cumulative_action_intensity,
        ],
        dtype=np.float64,
    )
    action_indicator = np.zeros(n_actions, dtype=np.float64)
    action_indicator[action_id] = 1.0
    interactions = np.outer(action_indicator, state_features).reshape(-1)
    return np.concatenate((step_action, full_actions, state_features, interactions))


def _oracle_dataset(
    config: E0Config,
    world: FatigueHabitWorld,
    data: E0Data,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    IntArray,
    IntArray,
    IntArray,
    tuple[OracleCreditExample, ...],
    bool,
]:
    exact_values: list[float] = []
    estimated_values: list[float] = []
    standard_errors: list[float] = []
    trajectory_indices: list[int] = []
    steps: list[int] = []
    action_ids: list[int] = []
    examples: list[OracleCreditExample] = []
    all_pairs_reused_noise = True
    mode = ContinuationMode(config.oracle.continuation_mode)
    reference_action = world.action_space[config.oracle.reference_action]

    for trajectory_index, episode in enumerate(data.episodes):
        for transition in episode.transitions:
            if len(exact_values) >= config.oracle.max_examples:
                break
            step = transition.step_index
            action = transition.action
            exact = exact_deterministic_credit(
                world,
                episode,
                step_index=step,
                action=action,
                reference_action=reference_action,
                continuation_mode=mode,
            )
            estimated = estimate_oracle_credit(
                world,
                episode,
                step_index=step,
                action=action,
                reference_action=reference_action,
                continuation_mode=mode,
                num_rollouts=config.oracle.mc_rollouts,
                base_seed=config.experiment.seed + trajectory_index * 10_000 + step * 100,
            )
            action_id = world.action_space.index(action)
            exact_values.append(exact.credit_utility)
            estimated_values.append(estimated.credit_utility)
            standard_errors.append(estimated.monte_carlo_se)
            trajectory_indices.append(trajectory_index)
            steps.append(step)
            action_ids.append(action_id)
            all_pairs_reused_noise = all_pairs_reused_noise and exact.paired_noise_reused
            continuation_actions = [
                future_action.value for future_action in episode.actions[step + 1 :]
            ]
            examples.append(
                OracleCreditExample(
                    trajectory_id=data.trajectories[trajectory_index].trajectory_id,
                    step_index=step,
                    action=action_id,
                    reference_action=config.oracle.reference_action,
                    future_policy_id=None,
                    continuation_id=f"actions:sha256:{sha256_json(continuation_actions)}",
                    continuation_mode=CreditContinuation(mode.value),
                    credit_utility=estimated.credit_utility,
                    credit_proxy=estimated.credit_proxy,
                    monte_carlo_se=estimated.monte_carlo_se,
                )
            )
        if len(exact_values) >= config.oracle.max_examples:
            break

    return (
        np.asarray(exact_values, dtype=np.float64),
        np.asarray(estimated_values, dtype=np.float64),
        np.asarray(standard_errors, dtype=np.float64),
        np.asarray(trajectory_indices, dtype=np.int64),
        np.asarray(steps, dtype=np.int64),
        np.asarray(action_ids, dtype=np.int64),
        tuple(examples),
        all_pairs_reused_noise,
    )


def _write_predictions(
    path: Path,
    *,
    test_indices: IntArray,
    data: E0Data,
    mean_predictions: FloatArray,
    ridge_predictions: FloatArray,
    rudder_predictions: FloatArray,
    oracle_exact: FloatArray,
    oracle_estimated: FloatArray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "record_type",
                "row_id",
                "observed",
                "mean_prediction",
                "ridge_prediction",
                "rudder_prediction",
            ]
        )
        for row, trajectory_index in enumerate(test_indices):
            writer.writerow(
                [
                    "outcome_proxy",
                    data.trajectories[int(trajectory_index)].trajectory_id,
                    data.proxies[int(trajectory_index)],
                    mean_predictions[row],
                    ridge_predictions[row],
                    rudder_predictions[row],
                ]
            )
        for row, (exact, estimated) in enumerate(zip(oracle_exact, oracle_estimated, strict=True)):
            writer.writerow(["oracle_utility_credit", row, exact, "", estimated, ""])
    return path


def _resolved_output_config(config: E0Config, output_dir: Path | None) -> E0Config:
    if output_dir is None:
        return config
    experiment = config.experiment.model_copy(update={"output_dir": output_dir})
    return config.model_copy(update={"experiment": experiment})


def run_e0(config: E0Config, *, output_dir: Path | None = None) -> ExperimentResult:
    """Run E0 and persist an auditable set of artifacts."""

    started = time.perf_counter()
    config = _resolved_output_config(config, output_dir)
    repository = _repository_root()
    configured_output = config.experiment.output_dir
    resolved_output = (
        configured_output if configured_output.is_absolute() else repository / configured_output
    )
    resolved_output.mkdir(parents=True, exist_ok=True)

    world = _world_from_config(config)
    data = generate_e0_data(config, world)
    n_actions = len(world.action_space)
    horizon = config.experiment.horizon
    train_indices, test_indices = deterministic_split(
        len(data.episodes),
        train_fraction=config.experiment.train_fraction,
        seed=config.experiment.seed,
    )

    outcome_features = action_sequence_features(
        data.action_ids.tolist(),
        horizon=horizon,
        n_actions=n_actions,
    )
    prefix_features = np.stack(
        [
            prefix_action_features(
                actions.tolist(),
                horizon=horizon,
                n_actions=n_actions,
            )
            for actions in data.action_ids
        ]
    )
    mean_model = MeanOutcomeBaseline().fit(
        outcome_features[train_indices],
        data.proxies[train_indices],
    )
    ridge_outcome = RidgeBaseline(alpha=config.baselines.ridge_alpha).fit(
        outcome_features[train_indices],
        data.proxies[train_indices],
    )
    mean_predictions = mean_model.predict(outcome_features[test_indices])
    ridge_predictions = ridge_outcome.predict(outcome_features[test_indices])

    rudder = RudderRedistributor(alpha=config.baselines.ridge_alpha).fit(
        prefix_features[train_indices],
        data.proxies[train_indices],
    )
    rudder_values = rudder.predict_values(prefix_features[test_indices])
    rudder_rewards = rudder.redistribute(prefix_features[test_indices])
    rudder_predictions = rudder_values[:, -1]
    telescoping_error = telescoping_residual(
        rudder_rewards,
        rudder_values[:, 0],
        rudder_values[:, -1],
    )

    (
        exact_credit,
        estimated_credit,
        credit_standard_errors,
        credit_trajectory_indices,
        credit_steps,
        credit_action_ids,
        oracle_examples,
        all_pairs_reused_noise,
    ) = _oracle_dataset(config, world, data)

    credit_design = np.stack(
        [
            _credit_design_row(
                data.episodes[int(trajectory_index)],
                step_index=int(step),
                action_id=int(action_id),
                action_sequence=data.action_ids[int(trajectory_index)],
                n_actions=n_actions,
            )
            for trajectory_index, step, action_id in zip(
                credit_trajectory_indices,
                credit_steps,
                credit_action_ids,
                strict=True,
            )
        ]
    )
    train_trajectory_set = set(train_indices.tolist())
    test_trajectory_set = set(test_indices.tolist())
    credit_train_mask = np.asarray(
        [int(index) in train_trajectory_set for index in credit_trajectory_indices]
    )
    credit_test_mask = np.asarray(
        [int(index) in test_trajectory_set for index in credit_trajectory_indices]
    )
    ridge_credit = RidgeBaseline(alpha=config.baselines.ridge_alpha).fit(
        credit_design[credit_train_mask],
        exact_credit[credit_train_mask],
    )
    ridge_credit_predictions = ridge_credit.predict(credit_design[credit_test_mask])

    credit_matrix = np.full((len(data.episodes), horizon), np.nan, dtype=np.float64)
    for trajectory_index, step, value in zip(
        credit_trajectory_indices,
        credit_steps,
        exact_credit,
        strict=True,
    ):
        credit_matrix[int(trajectory_index), int(step)] = value
    rudder_credit_targets = credit_matrix[test_indices].reshape(-1)
    rudder_credit_predictions = rudder_rewards.reshape(-1)
    finite_credit = np.isfinite(rudder_credit_targets)

    outcome_scores = np.asarray(
        [
            pearson_correlation(data.proxies[test_indices], mean_predictions),
            pearson_correlation(data.proxies[test_indices], ridge_predictions),
            pearson_correlation(data.proxies[test_indices], rudder_predictions),
        ]
    )
    credit_association_scores = np.asarray(
        [
            0.0,
            pearson_correlation(
                exact_credit[credit_test_mask],
                ridge_credit_predictions,
            ),
            pearson_correlation(
                rudder_credit_targets[finite_credit],
                rudder_credit_predictions[finite_credit],
            ),
        ]
    )

    exact_mc_correlation = pearson_correlation(exact_credit, estimated_credit)
    max_mc_se = float(np.max(credit_standard_errors))
    fixture_oracle_credit, fixture_analytic_credit, fixture_error = (
        hand_derived_one_step_credit_check(world)
    )
    elapsed = time.perf_counter() - started
    acceptance = {
        "analytic_credit_fixture_error_below_1e_12": fixture_error <= 1.0e-12,
        "runtime_under_300_seconds": elapsed < 300.0,
        "oracle_credit_correlation_above_0_99": exact_mc_correlation > 0.99,
        "deterministic_mc_se_zero": max_mc_se <= 1.0e-12,
        "paired_noise_reused": all_pairs_reused_noise,
        "rudder_telescoping": telescoping_error <= 1.0e-12,
    }
    metrics: dict[str, Any] = {
        "experiment": "e0",
        "status": "pass" if all(acceptance.values()) else "fail",
        "seed": config.experiment.seed,
        "runtime_seconds": elapsed,
        "acceptance": acceptance,
        "metric_conventions": {
            "constant_input_correlation": (
                "reported as 0.0 for finite diagnostic plots; statistically undefined"
            )
        },
        "data": {
            "episodes": len(data.episodes),
            "horizon": horizon,
            "events": sum(len(trajectory.events) for trajectory in data.trajectories),
            "outcome_examples": len(data.outcome_examples),
            "oracle_examples": len(oracle_examples),
            "proxy_positive_rate": float(np.mean(data.proxies)),
            "utility_mean": float(np.mean(data.utilities)),
        },
        "outcome_proxy": {
            "mean_rmse": rmse(data.proxies[test_indices], mean_predictions),
            "ridge_rmse": rmse(data.proxies[test_indices], ridge_predictions),
            "ridge_pearson": outcome_scores[1],
            "rudder_final_pearson": outcome_scores[2],
        },
        "oracle_validation": {
            "one_step_oracle_credit": fixture_oracle_credit,
            "one_step_analytic_credit": fixture_analytic_credit,
            "one_step_absolute_error": fixture_error,
        },
        "utility_credit": {
            "exact_vs_mc_pearson": exact_mc_correlation,
            "exact_vs_mc_spearman": spearman_correlation(exact_credit, estimated_credit),
            "exact_vs_mc_sign_accuracy": sign_accuracy(exact_credit, estimated_credit),
            "max_monte_carlo_se": max_mc_se,
            "oracle_supervised_ridge_pearson": credit_association_scores[1],
        },
        "cross_estimand_diagnostic": {
            "rudder_proxy_increment_vs_oracle_utility_credit_pearson": (
                credit_association_scores[2]
            ),
            "interpretation": (
                "association only: RUDDER predicts behavioral proxy Y; the oracle target is "
                "interventional utility credit U"
            ),
        },
        "redistribution": {"telescoping_rms_residual": telescoping_error},
    }
    metrics_path = resolved_output / config.report.metrics_filename
    predictions_path = resolved_output / config.report.predictions_filename
    plot_path = resolved_output / config.report.plot_filename
    manifest_path = resolved_output / config.report.manifest_filename
    _write_predictions(
        predictions_path,
        test_indices=test_indices,
        data=data,
        mean_predictions=mean_predictions,
        ridge_predictions=ridge_predictions,
        rudder_predictions=rudder_predictions,
        oracle_exact=exact_credit,
        oracle_estimated=estimated_credit,
    )
    plot_outcome_vs_credit(
        outcome_scores,
        credit_association_scores,
        plot_path,
        labels=("mean (constant->0)", "oracle-supervised ridge", "RUDDER cross-estimand"),
        outcome_label="Behavioral-proxy prediction correlation",
        credit_label="Association with interventional utility credit",
        title="Outcome prediction vs. utility-credit association",
    )
    elapsed = time.perf_counter() - started
    acceptance["runtime_under_300_seconds"] = elapsed < 300.0
    metrics["runtime_seconds"] = elapsed
    metrics["status"] = "pass" if all(acceptance.values()) else "fail"
    scientific_metrics = {key: value for key, value in metrics.items() if key != "runtime_seconds"}
    metrics["scientific_metrics_sha256"] = sha256_json(scientific_metrics)
    write_metrics_json(metrics, metrics_path)
    manifest = build_run_manifest(
        repository=repository,
        resolved_config=dump_resolved_config(config),
        artifacts={
            "metrics": metrics_path.name,
            "predictions": predictions_path.name,
            "plot": plot_path.name,
        },
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "plot": plot_path,
        "manifest": manifest_path,
    }
    return ExperimentResult(metrics=metrics, output_dir=resolved_output, artifacts=artifacts)
