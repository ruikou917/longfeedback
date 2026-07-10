"""Gate A: outcome accuracy versus credit recovery on stochastic Worlds A/B.

Three capacity-matched DOCM variants (outcome-only, prefix/RUDDER, and
credit-supervised) are trained per logged-data regime. The gate passes when
(1) some regime shows similar outcome AUROC but a materially different oracle
credit recovery, (2) the paired Monte Carlo oracle is stable across seeds, and
(3) a Q-greedy policy from the credit head beats behavior cloning on true
utility without raising the behavioral proxy while lowering utility.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from longfeedback.config import (
    GateAConfig,
    GateAModelSettings,
    GateAOracleSettings,
    GateATrainingSettings,
    dump_resolved_config,
)
from longfeedback.credit.metrics import (
    credit_recovery_summary,
    spearman_by_temporal_distance,
)
from longfeedback.credit.oracle import (
    ContinuationMode,
    OracleCreditEstimate,
    estimate_oracle_credit,
)
from longfeedback.evaluation import (
    auroc,
    brier_score,
    expected_calibration_error,
    pearson_correlation,
    plot_outcome_vs_credit,
    write_metrics_json,
)
from longfeedback.experiments.features import (
    deterministic_split,
    observation_features_for,
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
from longfeedback.worlds import (
    FatigueAction,
    FatigueHabitConfig,
    FatigueHabitObservation,
    FatigueHabitWorld,
    HiddenIntentConfig,
    HiddenIntentWorld,
    PrivilegedIntentPolicy,
    RepeatSuccessPolicy,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

VARIANT_NAMES: tuple[str, ...] = ("docm_outcome", "docm_prefix", "docm_credit")

_REGIME_SEED_STRIDE = 1_000_000
_POLICY_EVAL_SEED_OFFSET = 9_000_000
_STABILITY_SUBSEED = 313


@dataclass(frozen=True, slots=True)
class ResponseSeekingPolicy:
    """World A behavior policy defined on released observation fields only."""

    epsilon: float
    threshold: float

    def probabilities(self, observation: FatigueHabitObservation) -> tuple[float, ...]:
        actions = tuple(FatigueAction)
        favored = (
            FatigueAction.HELPFUL
            if observation.last_response < self.threshold
            else FatigueAction.NOOP
        )
        probabilities = [self.epsilon / len(actions) for _ in actions]
        probabilities[actions.index(favored)] += 1.0 - self.epsilon
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
        for action, probability in zip(FatigueAction, self.probabilities(observation), strict=True):
            cumulative += probability
            if random_value < cumulative:
                return action
        return FatigueAction.URGENT

    def log_probability(
        self,
        observation: FatigueHabitObservation,
        action: FatigueAction,
    ) -> float:
        return math.log(self.probabilities(observation)[tuple(FatigueAction).index(action)])


@dataclass(frozen=True)
class RegimeData:
    """One logged-data regime: a world, a behavior policy, and arrays."""

    name: str
    world: Any
    behavior_policy: Any
    episodes: tuple[Any, ...]
    observations: FloatArray
    actions: IntArray
    responses: FloatArray
    proxies: FloatArray
    utilities: FloatArray
    observation_regime: str
    propensity_quality: str
    seed_base: int

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[1])

    @property
    def n_actions(self) -> int:
        return len(self.world.action_space)

    def observation_features(self, observation: Any) -> list[float]:
        return observation_features_for(self.world, observation, horizon=self.horizon)


def rollout_regime(
    *,
    name: str,
    world: Any,
    policy: Any,
    episodes: int,
    seed_base: int,
    observation_regime: str,
    propensity_quality: str,
) -> RegimeData:
    horizon = world.horizon
    episode_list = []
    feature_rows: list[list[list[float]]] = []
    action_rows: list[list[int]] = []
    response_rows: list[list[float]] = []
    for episode_index in range(episodes):
        exogenous = world.sample_exogenous(seed_base + episode_index)
        episode = world.rollout_policy(policy, exogenous)
        episode_list.append(episode)
        action_rows.append(
            [world.action_space.index(transition.action) for transition in episode.transitions]
        )
        response_rows.append(
            [transition.info_value("response") for transition in episode.transitions]
        )
        feature_rows.append([])
        for transition in episode.transitions:
            feature_rows[-1].append(
                observation_features_for(world, transition.observation, horizon=horizon)
            )

    return RegimeData(
        name=name,
        world=world,
        behavior_policy=policy,
        episodes=tuple(episode_list),
        observations=np.asarray(feature_rows, dtype=np.float64),
        actions=np.asarray(action_rows, dtype=np.int64),
        responses=np.asarray(response_rows, dtype=np.float64),
        proxies=np.asarray([episode.terminal_proxy for episode in episode_list], dtype=np.float64),
        utilities=np.asarray(
            [episode.terminal_utility for episode in episode_list], dtype=np.float64
        ),
        observation_regime=observation_regime,
        propensity_quality=propensity_quality,
        seed_base=seed_base,
    )


def generate_regimes(config: GateAConfig) -> dict[str, RegimeData]:
    """Generate every configured logged-data regime deterministically."""

    regimes: dict[str, RegimeData] = {}
    stride_index = 0
    settings_a = config.world_a
    for observability in settings_a.observabilities:
        world = FatigueHabitWorld(
            FatigueHabitConfig.stochastic(
                horizon=settings_a.horizon,
                observability=observability,
                response_noise_std=settings_a.response_noise_std,
                habit_noise_std=settings_a.habit_noise_std,
                fatigue_noise_std=settings_a.fatigue_noise_std,
                observation_noise_std=settings_a.observation_noise_std,
                proxy_threshold=settings_a.proxy_threshold,
            )
        )
        name = f"world_a_{observability}"
        regimes[name] = rollout_regime(
            name=name,
            world=world,
            policy=ResponseSeekingPolicy(
                settings_a.behavior_epsilon, settings_a.response_seek_threshold
            ),
            episodes=settings_a.episodes,
            seed_base=config.experiment.seed + stride_index * _REGIME_SEED_STRIDE,
            observation_regime="partial" if observability == "partial" else "clean",
            propensity_quality="exact",
        )
        stride_index += 1

    settings_b = config.world_b
    for regime in settings_b.regimes:
        world_b = HiddenIntentWorld(
            HiddenIntentConfig(
                horizon=settings_b.horizon,
                stay_probability=settings_b.stay_probability,
                match_logit=settings_b.match_logit,
                mismatch_logit=settings_b.mismatch_logit,
                progress_shock_scale=settings_b.progress_shock_scale,
                proxy_threshold=settings_b.proxy_threshold,
                signal_accuracy=settings_b.signal_accuracy,
            )
        )
        confounded = regime == "confounded"
        policy: RepeatSuccessPolicy | PrivilegedIntentPolicy
        if confounded:
            policy = PrivilegedIntentPolicy(settings_b.privileged_epsilon)
        else:
            policy = RepeatSuccessPolicy(settings_b.clean_epsilon)
        name = f"world_b_{regime}"
        regimes[name] = rollout_regime(
            name=name,
            world=world_b,
            policy=policy,
            episodes=settings_b.episodes,
            seed_base=config.experiment.seed + stride_index * _REGIME_SEED_STRIDE,
            observation_regime="hidden_confounding" if confounded else "clean",
            propensity_quality="confounded" if confounded else "exact",
        )
        stride_index += 1
    return regimes


def _adaptive_oracle_estimate(
    regime: RegimeData,
    *,
    episode_index: int,
    step_index: int,
    oracle: GateAOracleSettings,
    seed_shift: int = 0,
) -> tuple[OracleCreditEstimate[Any], int]:
    """Escalate paired rollouts until the Monte Carlo SE meets the threshold."""

    episode = regime.episodes[episode_index]
    action = episode.transitions[step_index].action
    reference_action = regime.world.action_space[oracle.reference_action]
    base_seed = (
        regime.seed_base
        + _STABILITY_SUBSEED
        + seed_shift
        + episode_index * 50_000
        + step_index * 200
    )
    rollouts = oracle.initial_rollouts
    while True:
        estimate = estimate_oracle_credit(
            regime.world,
            episode,
            step_index=step_index,
            action=action,
            reference_action=reference_action,
            continuation_mode=ContinuationMode(oracle.continuation_mode),
            num_rollouts=rollouts,
            base_seed=base_seed,
        )
        if estimate.monte_carlo_se <= oracle.se_threshold or rollouts >= oracle.max_rollouts:
            return estimate, rollouts
        rollouts = min(rollouts * 2, oracle.max_rollouts)


@dataclass(frozen=True)
class OracleLabels:
    targets: FloatArray
    mask: BoolArray
    standard_errors: FloatArray
    max_se: float
    mean_rollouts: float
    labeled_steps: int


def label_oracle_credit(
    regime: RegimeData,
    episode_indices: IntArray,
    oracle: GateAOracleSettings,
) -> OracleLabels:
    """Attach adaptive paired-CRN utility-credit labels to selected episodes."""

    episodes = len(regime.episodes)
    horizon = regime.horizon
    targets = np.zeros((episodes, horizon), dtype=np.float64)
    mask = np.zeros((episodes, horizon), dtype=np.bool_)
    standard_errors = np.zeros((episodes, horizon), dtype=np.float64)
    rollout_counts: list[int] = []
    for episode_index in episode_indices:
        for step_index in range(horizon):
            estimate, rollouts = _adaptive_oracle_estimate(
                regime,
                episode_index=int(episode_index),
                step_index=step_index,
                oracle=oracle,
            )
            targets[episode_index, step_index] = estimate.credit_utility
            standard_errors[episode_index, step_index] = estimate.monte_carlo_se
            mask[episode_index, step_index] = True
            rollout_counts.append(rollouts)
    labeled = int(np.sum(mask))
    return OracleLabels(
        targets=targets,
        mask=mask,
        standard_errors=standard_errors,
        max_se=float(np.max(standard_errors[mask])) if labeled else 0.0,
        mean_rollouts=float(np.mean(rollout_counts)) if rollout_counts else 0.0,
        labeled_steps=labeled,
    )


def oracle_stability_pearson(
    regime: RegimeData,
    labels: OracleLabels,
    oracle: GateAOracleSettings,
) -> float:
    """Re-estimate a labeled subsample under a shifted seed and correlate."""

    labeled_positions = np.argwhere(labels.mask)
    if labeled_positions.shape[0] == 0:
        raise ValueError("stability check requires labeled oracle examples")
    subset = labeled_positions[: oracle.stability_examples]
    first: list[float] = []
    second: list[float] = []
    for episode_index, step_index in subset:
        first.append(float(labels.targets[episode_index, step_index]))
        estimate, _ = _adaptive_oracle_estimate(
            regime,
            episode_index=int(episode_index),
            step_index=int(step_index),
            oracle=oracle,
            seed_shift=oracle.stability_seed_offset,
        )
        second.append(estimate.credit_utility)
    return pearson_correlation(np.asarray(first), np.asarray(second))


def _sequence_dataset(
    regime: RegimeData,
    indices: IntArray,
    labels: OracleLabels | None = None,
) -> SequenceDataset:
    return SequenceDataset(
        observations=regime.observations[indices],
        actions=regime.actions[indices],
        responses=regime.responses[indices],
        outcomes=regime.proxies[indices],
        credit_targets=None if labels is None else labels.targets[indices],
        credit_mask=None if labels is None else labels.mask[indices],
        credit_se=None if labels is None else labels.standard_errors[indices],
    )


@dataclass(frozen=True)
class VariantEvaluation:
    model: DelayedOutcomeCreditModel
    outcome_metrics: dict[str, float]
    credit_metrics: dict[str, float]
    credit_by_distance: dict[str, float]
    credit_predictions: FloatArray
    outcome_predictions: FloatArray
    fit_summary: dict[str, float]


def _variant_credit_predictions(
    name: str,
    model: DelayedOutcomeCreditModel,
    dataset: SequenceDataset,
) -> FloatArray:
    if name == "docm_credit":
        return model.predict_logged_credit(dataset)
    if name == "docm_prefix":
        return np.diff(model.predict_prefix_values(dataset), axis=-1)
    return np.diff(model.predict_outcome_head_prefix_values(dataset), axis=-1)


def train_and_evaluate_variants(
    regime: RegimeData,
    labels: OracleLabels,
    train_indices: IntArray,
    test_indices: IntArray,
    *,
    model_settings: GateAModelSettings,
    training_settings: GateATrainingSettings,
    reference_action: int,
    seed: int,
) -> dict[str, VariantEvaluation]:
    architecture = EncoderArchitecture(
        d_model=model_settings.d_model,
        n_layers=model_settings.n_layers,
        n_heads=model_settings.n_heads,
        dropout=model_settings.dropout,
    )
    training = TrainingSettings(
        epochs=training_settings.epochs,
        batch_size=training_settings.batch_size,
        learning_rate=training_settings.learning_rate,
        weight_decay=training_settings.weight_decay,
        grad_clip=training_settings.grad_clip,
    )
    train_dataset = _sequence_dataset(regime, train_indices, labels)
    test_dataset = _sequence_dataset(regime, test_indices)
    test_mask = labels.mask[test_indices]
    test_targets = labels.targets[test_indices]
    steps_to_outcome = regime.horizon - np.tile(np.arange(regime.horizon), (len(test_indices), 1))

    results: dict[str, VariantEvaluation] = {}
    for name in VARIANT_NAMES:
        model = DelayedOutcomeCreditModel(
            observation_dim=int(regime.observations.shape[-1]),
            n_actions=regime.n_actions,
            horizon=regime.horizon,
            reference_action=reference_action,
            architecture=architecture,
            loss_weights=variant_loss_weights(name),
            seed=seed,
        )
        fit_summary = model.fit(train_dataset, training=training)
        outcome_predictions = model.predict_outcome_probability(test_dataset)
        credit_predictions = _variant_credit_predictions(name, model, test_dataset)
        outcome_metrics = {
            "auroc": auroc(regime.proxies[test_indices], outcome_predictions),
            "brier": brier_score(regime.proxies[test_indices], outcome_predictions),
            "ece": expected_calibration_error(regime.proxies[test_indices], outcome_predictions),
        }
        credit_metrics = credit_recovery_summary(
            test_targets[test_mask], credit_predictions[test_mask]
        )
        credit_by_distance = spearman_by_temporal_distance(
            test_targets[test_mask],
            credit_predictions[test_mask],
            steps_to_outcome[test_mask],
        )
        results[name] = VariantEvaluation(
            model=model,
            outcome_metrics=outcome_metrics,
            credit_metrics=credit_metrics,
            credit_by_distance=credit_by_distance,
            credit_predictions=credit_predictions,
            outcome_predictions=outcome_predictions,
            fit_summary=fit_summary,
        )
    return results


def _fit_behavior_clone(
    features: FloatArray,
    actions: IntArray,
    *,
    n_actions: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> nn.Linear:
    torch.manual_seed(seed)
    model = nn.Linear(features.shape[-1], n_actions)
    inputs = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    targets = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.int64))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        loss = nn.functional.cross_entropy(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    model.eval()
    return model


def _rollout_value(
    regime: RegimeData,
    select_action: Any,
    *,
    episodes: int,
    seed_base: int,
) -> tuple[float, float]:
    """Mean true utility and behavioral proxy over fresh evaluation seeds."""

    world = regime.world
    utilities: list[float] = []
    proxies: list[float] = []
    for episode_index in range(episodes):
        exogenous = world.sample_exogenous(seed_base + episode_index)
        state = world.initial_state()
        history_features: list[list[float]] = []
        history_actions: list[int] = []
        history_responses: list[float] = []
        for step_index in range(world.horizon):
            observation = world.observe(state)
            history_features.append(regime.observation_features(observation))
            action_index = select_action(
                observation,
                history_features,
                history_actions,
                history_responses,
                step_index,
                world.policy_random_value(exogenous[step_index]),
            )
            action = world.action_space[action_index]
            transition = world.step(state, action, exogenous[step_index])
            history_actions.append(action_index)
            history_responses.append(transition.info_value("response"))
            state = transition.next_state
        utilities.append(world.terminal_utility(state))
        proxies.append(world.terminal_proxy(state))
    return float(np.mean(utilities)), float(np.mean(proxies))


def run_policy_check(
    config: GateAConfig,
    regimes: dict[str, RegimeData],
    evaluations: dict[str, dict[str, VariantEvaluation]],
    train_indices: dict[str, IntArray],
) -> dict[str, Any]:
    """Gate A criterion 3: Q-greedy vs behavior cloning on true utility."""

    regime_name = config.policy_check.regime
    if regime_name not in regimes:
        known = ", ".join(sorted(regimes))
        raise ValueError(f"policy_check regime {regime_name!r} is not generated ({known})")
    regime = regimes[regime_name]
    credit_model = evaluations[regime_name]["docm_credit"].model
    train_idx = train_indices[regime_name]
    clone = _fit_behavior_clone(
        regime.observations[train_idx].reshape(-1, regime.observations.shape[-1]),
        regime.actions[train_idx].reshape(-1),
        n_actions=regime.n_actions,
        epochs=config.policy_check.bc_epochs,
        learning_rate=config.policy_check.bc_learning_rate,
        seed=config.experiment.seed,
    )

    def behavior_action(
        observation: Any,
        history_features: list[list[float]],
        history_actions: list[int],
        history_responses: list[float],
        step_index: int,
        random_value: float,
    ) -> int:
        action = regime.behavior_policy.select_action(
            observation, step_index=step_index, random_value=random_value
        )
        return int(regime.world.action_space.index(action))

    def clone_action(
        observation: Any,
        history_features: list[list[float]],
        history_actions: list[int],
        history_responses: list[float],
        step_index: int,
        random_value: float,
    ) -> int:
        del observation, history_actions, history_responses, step_index, random_value
        features = torch.from_numpy(np.asarray(history_features[-1], dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(clone(features), dim=-1).item())

    def greedy_q_action(
        observation: Any,
        history_features: list[list[float]],
        history_actions: list[int],
        history_responses: list[float],
        step_index: int,
        random_value: float,
    ) -> int:
        del observation, random_value
        # Pad the current step with a placeholder action/response; causality
        # guarantees the act token at this step cannot influence Q_t.
        dataset = SequenceDataset(
            observations=np.asarray([history_features], dtype=np.float64),
            actions=np.asarray([[*history_actions, 0]], dtype=np.int64),
            responses=np.asarray([[*history_responses, 0.0]], dtype=np.float64),
            outcomes=np.zeros(1, dtype=np.float64),
        )
        action_values = credit_model.predict_action_values(dataset)
        return int(np.argmax(action_values[0, step_index]))

    seed_base = regime.seed_base + _POLICY_EVAL_SEED_OFFSET
    episodes = config.policy_check.evaluation_episodes
    behavior_utility, behavior_proxy = _rollout_value(
        regime, behavior_action, episodes=episodes, seed_base=seed_base
    )
    clone_utility, clone_proxy = _rollout_value(
        regime, clone_action, episodes=episodes, seed_base=seed_base
    )
    greedy_utility, greedy_proxy = _rollout_value(
        regime, greedy_q_action, episodes=episodes, seed_base=seed_base
    )

    margin = config.decision.policy_utility_margin
    beats_clone = greedy_utility > clone_utility + margin
    proxy_inversion = greedy_proxy > behavior_proxy and greedy_utility < behavior_utility
    return {
        "regime": regime_name,
        "evaluation_episodes": episodes,
        "behavior": {"utility": behavior_utility, "proxy": behavior_proxy},
        "behavior_clone": {"utility": clone_utility, "proxy": clone_proxy},
        "greedy_q": {"utility": greedy_utility, "proxy": greedy_proxy},
        "beats_behavior_clone": beats_clone,
        "proxy_inversion": proxy_inversion,
        "pass": bool(beats_clone and not proxy_inversion),
    }


def _gap_decision(
    evaluations: dict[str, dict[str, VariantEvaluation]],
    config: GateAConfig,
) -> dict[str, Any]:
    """Criterion 1: similar outcome AUROC, materially different credit."""

    per_regime: dict[str, Any] = {}
    overall = False
    for regime_name, variants in evaluations.items():
        outcome_auroc_by_variant = {
            name: evaluation.outcome_metrics["auroc"] for name, evaluation in variants.items()
        }
        credit_spearman_by_variant = {
            name: evaluation.credit_metrics["spearman"] for name, evaluation in variants.items()
        }
        auroc_difference = abs(
            outcome_auroc_by_variant["docm_credit"] - outcome_auroc_by_variant["docm_outcome"]
        )
        spearman_gap = (
            credit_spearman_by_variant["docm_credit"] - credit_spearman_by_variant["docm_outcome"]
        )
        passed = (
            auroc_difference <= config.decision.outcome_auroc_tolerance
            and spearman_gap >= config.decision.credit_spearman_gap
        )
        overall = overall or passed
        per_regime[regime_name] = {
            "outcome_auroc": outcome_auroc_by_variant,
            "credit_spearman": credit_spearman_by_variant,
            "auroc_difference": auroc_difference,
            "spearman_gap": spearman_gap,
            "pass": passed,
        }
    return {"per_regime": per_regime, "pass": overall}


def _write_predictions(
    path: Path,
    *,
    regimes: dict[str, RegimeData],
    evaluations: dict[str, dict[str, VariantEvaluation]],
    test_indices: dict[str, IntArray],
    labels: dict[str, OracleLabels],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "record_type",
                "regime",
                "variant",
                "episode_index",
                "step_index",
                "observed",
                "prediction",
            ]
        )
        for regime_name, variants in evaluations.items():
            regime = regimes[regime_name]
            indices = test_indices[regime_name]
            regime_labels = labels[regime_name]
            for variant_name, evaluation in variants.items():
                for row, episode_index in enumerate(indices):
                    writer.writerow(
                        [
                            "outcome_proxy",
                            regime_name,
                            variant_name,
                            int(episode_index),
                            "",
                            regime.proxies[int(episode_index)],
                            evaluation.outcome_predictions[row],
                        ]
                    )
                for row, episode_index in enumerate(indices):
                    for step_index in range(regime.horizon):
                        if not regime_labels.mask[int(episode_index), step_index]:
                            continue
                        writer.writerow(
                            [
                                "oracle_utility_credit",
                                regime_name,
                                variant_name,
                                int(episode_index),
                                step_index,
                                regime_labels.targets[int(episode_index), step_index],
                                evaluation.credit_predictions[row, step_index],
                            ]
                        )
    return path


def _repository_root() -> Path:
    working_directory = Path.cwd().resolve()
    for candidate in (working_directory, *working_directory.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and (candidate / "src" / "longfeedback").is_dir():
            return candidate
    return working_directory


def _resolved_output_config(config: GateAConfig, output_dir: Path | None) -> GateAConfig:
    if output_dir is None:
        return config
    experiment = config.experiment.model_copy(update={"output_dir": output_dir})
    return config.model_copy(update={"experiment": experiment})


def run_gate_a(config: GateAConfig, *, output_dir: Path | None = None) -> ExperimentResult:
    """Run the Gate A experiment and persist an auditable set of artifacts."""

    started = time.perf_counter()
    config = _resolved_output_config(config, output_dir)
    repository = _repository_root()
    configured_output = config.experiment.output_dir
    resolved_output = (
        configured_output if configured_output.is_absolute() else repository / configured_output
    )
    resolved_output.mkdir(parents=True, exist_ok=True)

    regimes = generate_regimes(config)
    labels: dict[str, OracleLabels] = {}
    stability: dict[str, float] = {}
    evaluations: dict[str, dict[str, VariantEvaluation]] = {}
    train_indices: dict[str, IntArray] = {}
    test_indices: dict[str, IntArray] = {}
    data_summary: dict[str, Any] = {}

    for name, regime in regimes.items():
        train_idx, test_idx = deterministic_split(
            len(regime.episodes),
            train_fraction=config.experiment.train_fraction,
            seed=config.experiment.seed,
        )
        train_indices[name] = train_idx
        test_indices[name] = test_idx
        labeled_train = train_idx[: config.oracle.label_train_episodes]
        labeled_episodes = np.concatenate((labeled_train, test_idx))
        regime_labels = label_oracle_credit(regime, labeled_episodes, config.oracle)
        labels[name] = regime_labels
        stability[name] = oracle_stability_pearson(regime, regime_labels, config.oracle)
        evaluations[name] = train_and_evaluate_variants(
            regime,
            regime_labels,
            train_idx,
            test_idx,
            model_settings=config.model,
            training_settings=config.training,
            reference_action=config.oracle.reference_action,
            seed=config.experiment.seed,
        )
        data_summary[name] = {
            "episodes": len(regime.episodes),
            "horizon": regime.horizon,
            "proxy_positive_rate": float(np.mean(regime.proxies)),
            "utility_mean": float(np.mean(regime.utilities)),
            "observation_regime": regime.observation_regime,
            "propensity_quality": regime.propensity_quality,
            "observation_features": int(regime.observations.shape[-1]),
            "oracle": {
                "labeled_steps": regime_labels.labeled_steps,
                "max_monte_carlo_se": regime_labels.max_se,
                "mean_rollouts": regime_labels.mean_rollouts,
                "stability_pearson": stability[name],
            },
        }

    policy_check = run_policy_check(config, regimes, evaluations, train_indices)
    gap = _gap_decision(evaluations, config)
    stability_pass = min(stability.values()) >= config.decision.stability_pearson_threshold

    decision = {
        "outcome_credit_gap": gap["pass"],
        "oracle_stability": bool(stability_pass),
        "policy_improvement": policy_check["pass"],
        "gap_details": gap["per_regime"],
    }
    decision["pass"] = bool(
        decision["outcome_credit_gap"]
        and decision["oracle_stability"]
        and decision["policy_improvement"]
    )

    parameter_counts = {
        name: evaluation.model.parameter_count()
        for name, evaluation in next(iter(evaluations.values())).items()
    }
    regime_metrics: dict[str, Any] = {}
    for regime_name, variants in evaluations.items():
        regime_metrics[regime_name] = {
            variant_name: {
                "outcome": evaluation.outcome_metrics,
                "credit": evaluation.credit_metrics,
                "credit_spearman_by_steps_to_outcome": evaluation.credit_by_distance,
                "fit": evaluation.fit_summary,
            }
            for variant_name, evaluation in variants.items()
        }

    elapsed = time.perf_counter() - started
    metrics: dict[str, Any] = {
        "experiment": "gate_a",
        "status": "pass" if decision["pass"] else "fail",
        "seed": config.experiment.seed,
        "runtime_seconds": elapsed,
        "gate_a_decision": decision,
        "metric_conventions": {
            "constant_input_correlation": (
                "reported as 0.0 for finite diagnostic plots; statistically undefined"
            ),
            "credit_target": (
                "oracle interventional utility credit with frozen continuation; "
                "predictive-variant scores are associations, not causal estimates"
            ),
        },
        "data": data_summary,
        "model": {
            "parameter_counts": parameter_counts,
            "capacity_matched": len(set(parameter_counts.values())) == 1,
        },
        "regimes": regime_metrics,
        "policy_check": policy_check,
    }

    metrics_path = resolved_output / config.report.metrics_filename
    predictions_path = resolved_output / config.report.predictions_filename
    plot_path = resolved_output / config.report.plot_filename
    manifest_path = resolved_output / config.report.manifest_filename

    plot_outcomes: list[float] = []
    plot_credits: list[float] = []
    plot_labels: list[str] = []
    for regime_name, variants in evaluations.items():
        for variant_name, evaluation in variants.items():
            plot_outcomes.append(evaluation.outcome_metrics["auroc"])
            plot_credits.append(evaluation.credit_metrics["spearman"])
            plot_labels.append(f"{regime_name}:{variant_name}")
    plot_outcome_vs_credit(
        np.asarray(plot_outcomes),
        np.asarray(plot_credits),
        plot_path,
        labels=plot_labels,
        outcome_label="Terminal outcome AUROC (behavioral proxy)",
        credit_label="Oracle utility-credit Spearman",
        title="Outcome accuracy vs. credit recovery (Gate A)",
    )
    _write_predictions(
        predictions_path,
        regimes=regimes,
        evaluations=evaluations,
        test_indices=test_indices,
        labels=labels,
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
            "plot": plot_path.name,
        },
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ExperimentResult(
        metrics=metrics,
        output_dir=resolved_output,
        artifacts={
            "metrics": metrics_path,
            "predictions": predictions_path,
            "plot": plot_path,
            "manifest": manifest_path,
        },
    )
