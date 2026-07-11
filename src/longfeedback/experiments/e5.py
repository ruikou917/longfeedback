"""E5: reward overoptimization and pessimistic mitigation in World D.

REINFORCE policies are optimized against learned proxy-reward models with an
increasing update budget; the world supplies real transitions and the true
utility that the reward models never see. Two failure channels are reported
separately and must not be conflated:

- **RM exploitation**: the learned reward rises above the observed behavioral
  proxy (the policy found reward-model errors off the logged distribution);
- **proxy misalignment**: the observed proxy rises while true utility falls
  (Goodhart on the proxy itself). Ensemble pessimism targets only the first
  channel.

Checkpoints are never selected by proxy reward alone; every checkpoint reports
learned reward, observed proxy, and true utility side by side (design doc
§11.4, §13.4).
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
import torch

from longfeedback.config import E5Config, E5LoggingRegime, dump_resolved_config
from longfeedback.evaluation import plot_optimization_curves, write_metrics_json
from longfeedback.experiments.features import world_d_observation_features
from longfeedback.experiments.gate_a import rollout_regime
from longfeedback.experiments.manifest import build_run_manifest, sha256_json
from longfeedback.experiments.types import ExperimentResult
from longfeedback.models import (
    BootstrapEnsemble,
    DelayedOutcomeCreditModel,
    EncoderArchitecture,
    SequenceDataset,
    TrainingSettings,
    variant_loss_weights,
)
from longfeedback.policies import (
    EpisodeBatch,
    SoftmaxPolicy,
    WorldPolicyAdapter,
    collect_episodes,
    mean_kl_divergence,
    reinforce_update,
)
from longfeedback.worlds import (
    InfluenceAction,
    MixedInfluencePolicy,
    ProxyUtilityConfig,
    ProxyUtilityWorld,
)

FloatArray = npt.NDArray[np.float64]

REWARD_VARIANTS: tuple[str, ...] = ("single", "ensemble_mean", "ensemble_lcb", "single_kl")

_TRAIN_SEED_STRIDE = 40_000_000
_EVAL_SEED_OFFSET = 90_000_000
_REGIME_SEED_STRIDE = 700_000_000
_BAIT_ACTIONS = (InfluenceAction.URGENT, InfluenceAction.FLATTER, InfluenceAction.FEAR)


@dataclass(frozen=True)
class RewardScorer:
    """Score whole trajectories with a learned reward variant."""

    variant: str
    single_model: DelayedOutcomeCreditModel
    ensemble: BootstrapEnsemble
    lcb_lambda: float

    def score(self, batch: EpisodeBatch) -> FloatArray:
        dataset = SequenceDataset(
            observations=batch.observations,
            actions=batch.actions,
            responses=batch.responses,
            outcomes=np.zeros(batch.episodes, dtype=np.float64),
        )
        if self.variant in ("single", "single_kl"):
            return self.single_model.predict_outcome_probability(dataset)
        mean, std = self.ensemble.predict_outcome_probability(dataset)
        if self.variant == "ensemble_mean":
            return np.asarray(mean, dtype=np.float64)
        return np.asarray(mean - self.lcb_lambda * std, dtype=np.float64)

    def ensemble_uncertainty(self, batch: EpisodeBatch) -> float:
        dataset = SequenceDataset(
            observations=batch.observations,
            actions=batch.actions,
            responses=batch.responses,
            outcomes=np.zeros(batch.episodes, dtype=np.float64),
        )
        _, std = self.ensemble.predict_outcome_probability(dataset)
        return float(np.mean(std))


def _bait_action_fraction(batch: EpisodeBatch, world: ProxyUtilityWorld) -> float:
    bait_indices = {world.action_space.index(action) for action in _BAIT_ACTIONS}
    return float(np.mean(np.isin(batch.actions, list(bait_indices))))


def _curve_summaries(
    checkpoints: list[dict[str, float]],
) -> dict[str, float]:
    learned = np.asarray([point["learned_reward"] for point in checkpoints])
    observed = np.asarray([point["observed_proxy"] for point in checkpoints])
    utility = np.asarray([point["true_utility"] for point in checkpoints])
    proxy_optimal = int(np.argmax(learned))
    normalized_learned = (learned - learned.min()) / max(learned.max() - learned.min(), 1e-9)
    normalized_utility = (utility - utility.min()) / max(utility.max() - utility.min(), 1e-9)
    return {
        "hacking_gap": float(np.max(utility) - utility[proxy_optimal]),
        "utility_at_proxy_optimal": float(utility[proxy_optimal]),
        "proxy_optimal_checkpoint": float(proxy_optimal),
        "peak_true_utility": float(np.max(utility)),
        "peak_learned_reward": float(np.max(learned)),
        "rm_exploitation_gap": float(np.max(learned - observed)),
        "proxy_utility_curve_area": float(np.mean(np.abs(normalized_learned - normalized_utility))),
        "final_true_utility": float(utility[-1]),
        "initial_true_utility": float(utility[0]),
    }


def _optimize_variant(
    variant: str,
    *,
    world: ProxyUtilityWorld,
    scorer: RewardScorer,
    reference: SoftmaxPolicy,
    config: E5Config,
    variant_index: int,
    eval_seed_base: int,
) -> tuple[list[dict[str, float]], list[int]]:
    """Run one REINFORCE budget sweep; return checkpoint metrics and budgets."""

    torch.manual_seed(config.seed + variant_index)
    policy = reference.clone_detached()
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.optimization.learning_rate)
    horizon = world.horizon

    def feature_fn(observation: Any) -> list[float]:
        return world_d_observation_features(
            observation, horizon=horizon, n_actions=len(world.action_space)
        )

    adapter = WorldPolicyAdapter(policy=policy, feature_fn=feature_fn)

    def evaluate(update_index: int) -> dict[str, float]:
        batch = collect_episodes(
            world, adapter, episodes=config.evaluation_episodes, seed_base=eval_seed_base
        )
        rewards = scorer.score(batch)
        return {
            "update": float(update_index),
            "learned_reward": float(np.mean(rewards)),
            "observed_proxy": float(np.mean(batch.proxies)),
            "true_utility": float(np.mean(batch.utilities)),
            "ensemble_uncertainty": scorer.ensemble_uncertainty(batch),
            "kl_to_behavior_clone": mean_kl_divergence(policy, reference, batch.observations),
            "bait_action_fraction": _bait_action_fraction(batch, world),
        }

    checkpoints = [evaluate(0)]
    budgets = [0]
    train_seed_base = config.seed + _TRAIN_SEED_STRIDE * (variant_index + 1)
    for update_index in range(1, config.optimization.updates + 1):
        batch = collect_episodes(
            world,
            adapter,
            episodes=config.optimization.batch_episodes,
            seed_base=train_seed_base + update_index * config.optimization.batch_episodes,
        )
        rewards = scorer.score(batch)
        reinforce_update(
            policy,
            optimizer,
            batch,
            rewards,
            entropy_coefficient=config.optimization.entropy_coefficient,
            kl_coefficient=(config.optimization.kl_beta if variant == "single_kl" else 0.0),
            reference=reference if variant == "single_kl" else None,
        )
        if (
            update_index % config.optimization.checkpoint_every == 0
            or update_index == config.optimization.updates
        ):
            checkpoints.append(evaluate(update_index))
            budgets.append(update_index)
    return checkpoints, budgets


def _repository_root() -> Path:
    working_directory = Path.cwd().resolve()
    for candidate in (working_directory, *working_directory.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and (candidate / "src" / "longfeedback").is_dir():
            return candidate
    return working_directory


def _run_logging_regime(
    regime_settings: E5LoggingRegime,
    *,
    world: ProxyUtilityWorld,
    config: E5Config,
    regime_index: int,
) -> dict[str, Any]:
    """Train reward models on one logging-support regime and sweep variants."""

    seed = config.seed + regime_index * _REGIME_SEED_STRIDE
    regime = rollout_regime(
        name=f"world_d_logged_{regime_settings.name}",
        world=world,
        policy=MixedInfluencePolicy(regime_settings.behavior_epsilon),
        episodes=regime_settings.episodes,
        seed_base=seed,
        observation_regime="partial",
        propensity_quality="exact",
    )
    logged_dataset = SequenceDataset(
        observations=regime.observations,
        actions=regime.actions,
        responses=regime.responses,
        outcomes=regime.proxies,
    )
    architecture = EncoderArchitecture(
        d_model=config.reward_model.d_model,
        n_layers=config.reward_model.n_layers,
        n_heads=config.reward_model.n_heads,
    )
    training = TrainingSettings(
        epochs=config.reward_model.epochs,
        batch_size=config.reward_model.batch_size,
        learning_rate=config.reward_model.learning_rate,
    )
    single_model = DelayedOutcomeCreditModel(
        observation_dim=int(regime.observations.shape[-1]),
        n_actions=regime.n_actions,
        horizon=regime.horizon,
        architecture=architecture,
        loss_weights=variant_loss_weights("docm_outcome"),
        seed=config.seed,
    )
    single_model.fit(logged_dataset, training=training)
    ensemble = BootstrapEnsemble(
        observation_dim=int(regime.observations.shape[-1]),
        n_actions=regime.n_actions,
        horizon=regime.horizon,
        architecture=architecture,
        loss_weights=variant_loss_weights("docm_outcome"),
        members=config.reward_model.ensemble_members,
        seed=config.seed + 1,
    )
    ensemble.fit(logged_dataset, training=training)

    reference = SoftmaxPolicy.behavior_clone(
        regime.observations.reshape(-1, regime.observations.shape[-1]),
        regime.actions.reshape(-1),
        n_actions=regime.n_actions,
        epochs=config.optimization.behavior_clone_epochs,
        learning_rate=config.optimization.behavior_clone_learning_rate,
        seed=config.seed,
    )

    variant_results: dict[str, Any] = {}
    curves_for_plot: dict[str, dict[str, list[float]]] = {}
    budgets: list[int] = []
    for variant_index, variant in enumerate(REWARD_VARIANTS):
        scorer = RewardScorer(
            variant=variant,
            single_model=single_model,
            ensemble=ensemble,
            lcb_lambda=config.reward_model.lcb_lambda,
        )
        checkpoints, budgets = _optimize_variant(
            variant,
            world=world,
            scorer=scorer,
            reference=reference,
            config=config,
            variant_index=regime_index * len(REWARD_VARIANTS) + variant_index,
            eval_seed_base=seed + _EVAL_SEED_OFFSET,
        )
        variant_results[variant] = {
            "checkpoints": checkpoints,
            "summary": _curve_summaries(checkpoints),
        }
        curves_for_plot[variant] = {
            "learned_reward": [point["learned_reward"] for point in checkpoints],
            "observed_proxy": [point["observed_proxy"] for point in checkpoints],
            "true_utility": [point["true_utility"] for point in checkpoints],
        }
    return {
        "logged": {
            "episodes": regime_settings.episodes,
            "behavior_epsilon": regime_settings.behavior_epsilon,
            "proxy_rate": float(np.mean(regime.proxies)),
            "utility_mean": float(np.mean(regime.utilities)),
        },
        "variants": variant_results,
        "curves": curves_for_plot,
        "budgets": budgets,
    }


def _mitigates(candidate: dict[str, float], single: dict[str, float], margin: float) -> bool:
    return bool(
        candidate["hacking_gap"] <= single["hacking_gap"] - margin
        and candidate["utility_at_proxy_optimal"] >= single["utility_at_proxy_optimal"] + margin
    )


def _regime_decision(variants: dict[str, Any], config: E5Config) -> dict[str, Any]:
    single_summary = variants["single"]["summary"]
    margin = config.decision.mitigation_margin
    goodhart_observed = single_summary["hacking_gap"] >= config.decision.hacking_gap_threshold
    rm_error_channel_active = (
        max(variants[variant]["summary"]["rm_exploitation_gap"] for variant in REWARD_VARIANTS)
        >= config.decision.rm_error_active_threshold
    )
    return {
        "goodhart_observed": bool(goodhart_observed),
        "rm_error_channel_active": bool(rm_error_channel_active),
        "lcb_mitigates": _mitigates(variants["ensemble_lcb"]["summary"], single_summary, margin),
        "kl_mitigates": _mitigates(variants["single_kl"]["summary"], single_summary, margin),
        "hacking_gaps": {
            variant: variants[variant]["summary"]["hacking_gap"] for variant in REWARD_VARIANTS
        },
        "utility_at_proxy_optimal": {
            variant: variants[variant]["summary"]["utility_at_proxy_optimal"]
            for variant in REWARD_VARIANTS
        },
        "rm_exploitation_gaps": {
            variant: variants[variant]["summary"]["rm_exploitation_gap"]
            for variant in REWARD_VARIANTS
        },
    }


def run_e5(config: E5Config, *, output_dir: Path | None = None) -> ExperimentResult:
    """Run the overoptimization experiment and persist auditable artifacts."""

    started = time.perf_counter()
    repository = _repository_root()
    resolved_output = output_dir or config.output_dir
    if not resolved_output.is_absolute():
        resolved_output = repository / resolved_output
    resolved_output.mkdir(parents=True, exist_ok=True)

    world = ProxyUtilityWorld(ProxyUtilityConfig())
    regime_metrics: dict[str, Any] = {}
    regime_curves: dict[str, dict[str, dict[str, list[float]]]] = {}
    regime_budgets: dict[str, list[int]] = {}
    for regime_index, regime_settings in enumerate(config.logging_regimes):
        outcome = _run_logging_regime(
            regime_settings, world=world, config=config, regime_index=regime_index
        )
        regime_metrics[regime_settings.name] = {
            "logged": outcome["logged"],
            "variants": outcome["variants"],
            "decision": _regime_decision(outcome["variants"], config),
        }
        regime_curves[regime_settings.name] = outcome["curves"]
        regime_budgets[regime_settings.name] = outcome["budgets"]

    decisions = {name: block["decision"] for name, block in regime_metrics.items()}
    goodhart_observed = any(block["goodhart_observed"] for block in decisions.values())
    lcb_mitigates = any(block["lcb_mitigates"] for block in decisions.values())
    kl_mitigates = any(block["kl_mitigates"] for block in decisions.values())
    rm_error_active = any(block["rm_error_channel_active"] for block in decisions.values())
    if lcb_mitigates:
        h5_verdict = "supported"
    elif not rm_error_active:
        # LCB pessimism targets reward-model error; when the learned reward
        # never overestimates the observed proxy there is nothing for it to
        # mitigate, so H5's precondition is absent rather than the claim
        # being refuted.
        h5_verdict = "not_testable_rm_error_channel_inactive"
    else:
        h5_verdict = "refuted_in_this_environment"
    decision = {
        "goodhart_observed": goodhart_observed,
        "rm_error_channel_active": rm_error_active,
        "lcb_mitigates": lcb_mitigates,
        "kl_mitigates": kl_mitigates,
        "hypothesis_h5_lcb": h5_verdict,
        "per_regime": decisions,
        # The experiment passes when the Goodhart effect is demonstrated and
        # at least one mitigation materially helps; the H5 verdict about LCB
        # specifically is recorded separately and may be negative.
        "pass": bool(goodhart_observed and (lcb_mitigates or kl_mitigates)),
    }

    elapsed = time.perf_counter() - started
    metrics: dict[str, Any] = {
        "experiment": "e5",
        "status": "pass" if decision["pass"] else "fail",
        "seed": config.seed,
        "runtime_seconds": elapsed,
        "e5_decision": decision,
        "data": {
            "horizon": world.horizon,
            "updates": config.optimization.updates,
            "evaluation_episodes": config.evaluation_episodes,
            "logging_regimes": [regime.name for regime in config.logging_regimes],
        },
        "regimes": regime_metrics,
        "metric_conventions": {
            "channels": (
                "rm_exploitation_gap = learned reward above observed proxy (RM error "
                "exploitation); a proxy rising while utility falls is proxy "
                "misalignment, which pessimism does not target"
            ),
            "checkpoint_selection": "never by proxy reward alone; all series reported",
        },
    }

    metrics_path = resolved_output / config.metrics_filename
    predictions_path = resolved_output / config.predictions_filename
    manifest_path = resolved_output / config.manifest_filename
    plot_base = Path(config.plot_filename)
    artifacts: dict[str, Path] = {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "manifest": manifest_path,
    }
    for regime_name, curves in regime_curves.items():
        plot_path = resolved_output / f"{plot_base.stem}_{regime_name}{plot_base.suffix}"
        plot_optimization_curves(
            curves,
            regime_budgets[regime_name],
            plot_path,
            series_labels={
                "learned_reward": "learned reward (J_R̂)",
                "observed_proxy": "observed proxy (J_Y)",
                "true_utility": "true utility (J_U)",
            },
            title=f"Reward overoptimization in World D ({regime_name} logging support)",
        )
        artifacts[f"plot_{regime_name}"] = plot_path

    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "regime",
                "variant",
                "update",
                "learned_reward",
                "observed_proxy",
                "true_utility",
                "ensemble_uncertainty",
                "kl_to_behavior_clone",
                "bait_action_fraction",
            ]
        )
        for regime_name, block in regime_metrics.items():
            for variant in REWARD_VARIANTS:
                for point in block["variants"][variant]["checkpoints"]:
                    writer.writerow(
                        [
                            regime_name,
                            variant,
                            int(point["update"]),
                            point["learned_reward"],
                            point["observed_proxy"],
                            point["true_utility"],
                            point["ensemble_uncertainty"],
                            point["kl_to_behavior_clone"],
                            point["bait_action_fraction"],
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
        artifacts={name: path.name for name, path in artifacts.items()},
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ExperimentResult(
        metrics=metrics,
        output_dir=resolved_output,
        artifacts=artifacts,
    )
