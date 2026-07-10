"""Gate B: robustness of credit recovery across four structural families.

Per family this experiment (1) reruns the capacity-matched variant comparison
(does credit supervision beat outcome-only and prefix/RUDDER training?),
(2) trains a bootstrap DOCM ensemble and evaluates its epistemic uncertainty
in-distribution and under a family-specific distribution shift (parameter
shift for Worlds A/C/D, logging-policy shift for World B), and (3) folds in
the real-log E1 result. The decision block encodes the design-doc Gate B
criteria.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from longfeedback.config import GateBConfig, dump_resolved_config
from longfeedback.credit.metrics import credit_recovery_summary
from longfeedback.evaluation import (
    auroc,
    error_detection_auroc,
    plot_outcome_vs_credit,
    spearman_correlation,
    write_metrics_json,
)
from longfeedback.experiments.features import deterministic_split
from longfeedback.experiments.gate_a import (
    OracleLabels,
    RegimeData,
    ResponseSeekingPolicy,
    VariantEvaluation,
    _sequence_dataset,
    label_oracle_credit,
    oracle_stability_pearson,
    rollout_regime,
    train_and_evaluate_variants,
)
from longfeedback.experiments.manifest import build_run_manifest, sha256_json
from longfeedback.experiments.types import ExperimentResult
from longfeedback.models import BootstrapEnsemble, variant_loss_weights
from longfeedback.models.docm import TrainingSettings
from longfeedback.models.encoders import EncoderArchitecture
from longfeedback.worlds import (
    DelayedConversionConfig,
    DelayedConversionWorld,
    FatigueHabitConfig,
    FatigueHabitWorld,
    HiddenIntentConfig,
    HiddenIntentWorld,
    MixedInfluencePolicy,
    PrivilegedIntentPolicy,
    ProxyUtilityConfig,
    ProxyUtilityWorld,
    RepeatSuccessPolicy,
    SpacedOutreachPolicy,
)

_FAMILY_SEED_STRIDE = 1_000_000
_SHIFT_SEED_OFFSET = 500_000

FAMILY_NAMES: tuple[str, ...] = ("world_a", "world_b", "world_c", "world_d")


@dataclass(frozen=True)
class FamilySpec:
    """A training regime plus a distribution-shifted evaluation regime."""

    name: str
    train_regime: RegimeData
    shifted_regime: RegimeData
    shift_kind: str


def build_family_specs(config: GateBConfig) -> dict[str, FamilySpec]:
    """Instantiate the four structural families and their shifts."""

    seed = config.experiment.seed
    episodes = config.families.episodes
    shift_episodes = config.families.shift_episodes
    specs: dict[str, FamilySpec] = {}

    # World A: stochastic, partial observability; shift = heavier noise.
    world_a = FatigueHabitWorld(FatigueHabitConfig.stochastic(observability="partial"))
    world_a_shifted = FatigueHabitWorld(
        FatigueHabitConfig.stochastic(
            observability="partial",
            response_noise_std=1.0,
            habit_noise_std=0.1,
            fatigue_noise_std=0.1,
            observation_noise_std=0.25,
        )
    )
    policy_a = ResponseSeekingPolicy(0.3, 0.55)
    specs["world_a"] = FamilySpec(
        name="world_a",
        train_regime=rollout_regime(
            name="world_a",
            world=world_a,
            policy=policy_a,
            episodes=episodes,
            seed_base=seed,
            observation_regime="partial",
            propensity_quality="exact",
        ),
        shifted_regime=rollout_regime(
            name="world_a_shifted",
            world=world_a_shifted,
            policy=policy_a,
            episodes=shift_episodes,
            seed_base=seed + _SHIFT_SEED_OFFSET,
            observation_regime="partial",
            propensity_quality="exact",
        ),
        shift_kind="parameter_shift(noise scales)",
    )

    # World B: hidden confounding; shift = logging-policy change to clean.
    world_b = HiddenIntentWorld(HiddenIntentConfig())
    specs["world_b"] = FamilySpec(
        name="world_b",
        train_regime=rollout_regime(
            name="world_b",
            world=world_b,
            policy=PrivilegedIntentPolicy(0.15),
            episodes=episodes,
            seed_base=seed + _FAMILY_SEED_STRIDE,
            observation_regime="hidden_confounding",
            propensity_quality="confounded",
        ),
        shifted_regime=rollout_regime(
            name="world_b_shifted",
            world=world_b,
            policy=RepeatSuccessPolicy(0.3),
            episodes=shift_episodes,
            seed_base=seed + _FAMILY_SEED_STRIDE + _SHIFT_SEED_OFFSET,
            observation_regime="clean",
            propensity_quality="exact",
        ),
        shift_kind="logging_policy_shift(confounded->clean)",
    )

    # World C: delayed conversion; shift = slower, heavier-tailed delays.
    world_c = DelayedConversionWorld(DelayedConversionConfig())
    world_c_shifted = DelayedConversionWorld(
        DelayedConversionConfig(
            base_hazard=0.005,
            delay_geometric_p=(1.0, 0.35, 0.15),
            kernel_decay=0.75,
        )
    )
    policy_c = SpacedOutreachPolicy(0.2)
    specs["world_c"] = FamilySpec(
        name="world_c",
        train_regime=rollout_regime(
            name="world_c",
            world=world_c,
            policy=policy_c,
            episodes=episodes,
            seed_base=seed + 2 * _FAMILY_SEED_STRIDE,
            observation_regime="partial",
            propensity_quality="exact",
        ),
        shifted_regime=rollout_regime(
            name="world_c_shifted",
            world=world_c_shifted,
            policy=policy_c,
            episodes=shift_episodes,
            seed_base=seed + 2 * _FAMILY_SEED_STRIDE + _SHIFT_SEED_OFFSET,
            observation_regime="partial",
            propensity_quality="exact",
        ),
        shift_kind="parameter_shift(delay structure)",
    )

    # World D: proxy-utility divergence; shift = stronger engagement bait.
    world_d = ProxyUtilityWorld(ProxyUtilityConfig())
    world_d_shifted = ProxyUtilityWorld(
        ProxyUtilityConfig(
            engagement_lifts=(0.0, 0.25, 0.85, 0.7, 1.1),
            trust_lifts=(0.0, 0.2, -0.3, -0.15, -0.6),
            engagement_noise_std=0.25,
        )
    )
    policy_d = MixedInfluencePolicy(0.3)
    specs["world_d"] = FamilySpec(
        name="world_d",
        train_regime=rollout_regime(
            name="world_d",
            world=world_d,
            policy=policy_d,
            episodes=episodes,
            seed_base=seed + 3 * _FAMILY_SEED_STRIDE,
            observation_regime="partial",
            propensity_quality="exact",
        ),
        shifted_regime=rollout_regime(
            name="world_d_shifted",
            world=world_d_shifted,
            policy=policy_d,
            episodes=shift_episodes,
            seed_base=seed + 3 * _FAMILY_SEED_STRIDE + _SHIFT_SEED_OFFSET,
            observation_regime="partial",
            propensity_quality="exact",
        ),
        shift_kind="parameter_shift(engagement/trust lifts)",
    )
    return specs


def _uncertainty_metrics(
    ensemble: BootstrapEnsemble,
    regime: RegimeData,
    labels: OracleLabels,
    indices: np.ndarray,
) -> dict[str, float]:
    """Uncertainty quality on labeled credit targets for chosen episodes."""

    dataset = _sequence_dataset(regime, indices)
    credit_mean, credit_std = ensemble.predict_logged_credit(dataset)
    mask = labels.mask[indices]
    targets = labels.targets[indices]
    errors = np.abs(credit_mean[mask] - targets[mask])
    uncertainties = credit_std[mask]
    outcome_mean, _ = ensemble.predict_outcome_probability(dataset)
    return {
        "outcome_auroc": auroc(regime.proxies[indices], outcome_mean),
        "credit_spearman": credit_recovery_summary(targets[mask], credit_mean[mask])["spearman"],
        "uncertainty_error_spearman": spearman_correlation(errors, uncertainties),
        "error_detection_auroc": error_detection_auroc(errors, uncertainties),
        "mean_uncertainty": float(np.mean(uncertainties)),
        "labeled_steps": float(np.sum(mask)),
    }


def _real_log_criterion(config: GateBConfig, repository: Path) -> dict[str, Any]:
    path = config.decision.e1_metrics_path
    if not path.is_absolute():
        path = repository / path
    if not path.is_file():
        return {
            "available": False,
            "pass": not config.decision.require_real_log,
            "note": f"E1 metrics not found at {path}; run `make e1` after `make data-lmsys`",
        }
    e1_metrics = json.loads(path.read_text(encoding="utf-8"))
    decision = e1_metrics.get("e1_decision", {})
    return {
        "available": True,
        "pass": bool(decision.get("learnable_above_trivial", False)),
        "best_informed_auroc": decision.get("best_informed_auroc"),
        "trivial_length_auroc": decision.get("trivial_length_auroc"),
        "e1_scientific_metrics_sha256": e1_metrics.get("scientific_metrics_sha256"),
    }


def _repository_root() -> Path:
    working_directory = Path.cwd().resolve()
    for candidate in (working_directory, *working_directory.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and (candidate / "src" / "longfeedback").is_dir():
            return candidate
    return working_directory


def run_gate_b(config: GateBConfig, *, output_dir: Path | None = None) -> ExperimentResult:
    """Run the Gate B experiment and persist an auditable set of artifacts."""

    started = time.perf_counter()
    repository = _repository_root()
    resolved_output = output_dir or config.experiment.output_dir
    if not resolved_output.is_absolute():
        resolved_output = repository / resolved_output
    resolved_output.mkdir(parents=True, exist_ok=True)

    specs = build_family_specs(config)
    family_metrics: dict[str, Any] = {}
    winning_families = 0
    uncertainty_pass_families: list[str] = []
    capacity_matched_per_family: list[bool] = []

    for name, spec in specs.items():
        regime = spec.train_regime
        train_idx, test_idx = deterministic_split(
            len(regime.episodes),
            train_fraction=config.experiment.train_fraction,
            seed=config.experiment.seed,
        )
        labeled_train = train_idx[: config.families.label_train_episodes]
        labels = label_oracle_credit(
            regime, np.concatenate((labeled_train, test_idx)), config.oracle
        )
        stability = oracle_stability_pearson(regime, labels, config.oracle)
        variants: dict[str, VariantEvaluation] = train_and_evaluate_variants(
            regime,
            labels,
            train_idx,
            test_idx,
            model_settings=config.model,
            training_settings=config.training,
            reference_action=config.oracle.reference_action,
            seed=config.experiment.seed,
        )
        # Capacity matching is a within-family property: variants of one
        # regime share an architecture; observation dims differ across worlds.
        family_parameter_counts = {
            evaluation.model.parameter_count() for evaluation in variants.values()
        }
        capacity_matched_per_family.append(len(family_parameter_counts) == 1)
        credit_spearman = {
            variant: evaluation.credit_metrics["spearman"]
            for variant, evaluation in variants.items()
        }
        wins = credit_spearman["docm_credit"] >= (
            max(credit_spearman["docm_outcome"], credit_spearman["docm_prefix"])
            + config.decision.credit_spearman_margin
        )
        winning_families += int(wins)

        ensemble = BootstrapEnsemble(
            observation_dim=int(regime.observations.shape[-1]),
            n_actions=regime.n_actions,
            horizon=regime.horizon,
            reference_action=config.oracle.reference_action,
            architecture=EncoderArchitecture(
                d_model=config.model.d_model,
                n_layers=config.model.n_layers,
                n_heads=config.model.n_heads,
                dropout=config.model.dropout,
            ),
            loss_weights=variant_loss_weights("docm_credit"),
            members=config.ensemble_members,
            seed=config.experiment.seed,
        )
        ensemble.fit(
            _sequence_dataset(regime, train_idx, labels),
            training=TrainingSettings(
                epochs=config.training.epochs,
                batch_size=config.training.batch_size,
                learning_rate=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
                grad_clip=config.training.grad_clip,
            ),
        )
        in_distribution = _uncertainty_metrics(ensemble, regime, labels, test_idx)

        shifted = spec.shifted_regime
        shifted_indices = np.arange(len(shifted.episodes), dtype=np.int64)
        shifted_labels = label_oracle_credit(shifted, shifted_indices, config.oracle)
        under_shift = _uncertainty_metrics(ensemble, shifted, shifted_labels, shifted_indices)
        uncertainty_pass = (
            under_shift["uncertainty_error_spearman"]
            >= config.decision.uncertainty_spearman_threshold
            and under_shift["error_detection_auroc"]
            >= config.decision.error_detection_auroc_threshold
        )
        if uncertainty_pass:
            uncertainty_pass_families.append(name)

        family_metrics[name] = {
            "shift_kind": spec.shift_kind,
            "oracle": {
                "labeled_steps": labels.labeled_steps,
                "max_monte_carlo_se": labels.max_se,
                "stability_pearson": stability,
            },
            "variants": {
                variant: {
                    "outcome": evaluation.outcome_metrics,
                    "credit": evaluation.credit_metrics,
                }
                for variant, evaluation in variants.items()
            },
            "credit_supervision_wins": bool(wins),
            "ensemble": {
                "members": config.ensemble_members,
                "in_distribution": in_distribution,
                "under_shift": under_shift,
                "outcome_auroc_degradation": (
                    in_distribution["outcome_auroc"] - under_shift["outcome_auroc"]
                ),
                "credit_spearman_degradation": (
                    in_distribution["credit_spearman"] - under_shift["credit_spearman"]
                ),
                "uncertainty_pass_under_shift": bool(uncertainty_pass),
            },
        }

    real_log = _real_log_criterion(config, repository)
    decision = {
        "credit_recovery_across_families": winning_families >= config.decision.min_winning_families,
        "winning_families": winning_families,
        "capacity_matched": all(capacity_matched_per_family),
        "uncertainty_under_shift": bool(uncertainty_pass_families),
        "uncertainty_pass_families": uncertainty_pass_families,
        "real_log_learnable": real_log["pass"],
        "real_log": real_log,
    }
    decision["pass"] = bool(
        decision["credit_recovery_across_families"]
        and decision["capacity_matched"]
        and decision["uncertainty_under_shift"]
        and decision["real_log_learnable"]
    )

    elapsed = time.perf_counter() - started
    metrics: dict[str, Any] = {
        "experiment": "gate_b",
        "status": "pass" if decision["pass"] else "fail",
        "seed": config.experiment.seed,
        "runtime_seconds": elapsed,
        "gate_b_decision": decision,
        "families": family_metrics,
        "metric_conventions": {
            "credit_target": (
                "oracle interventional utility credit with frozen continuation; "
                "predictive-variant scores are associations, not causal estimates"
            ),
            "uncertainty": "between-member std of a bootstrap ensemble (epistemic)",
        },
    }

    plot_outcomes = []
    plot_credits = []
    plot_labels = []
    for name, family in family_metrics.items():
        for variant, values in family["variants"].items():
            plot_outcomes.append(values["outcome"]["auroc"])
            plot_credits.append(values["credit"]["spearman"])
            plot_labels.append(f"{name}:{variant}")
    metrics_path = resolved_output / config.report.metrics_filename
    plot_path = resolved_output / config.report.plot_filename
    manifest_path = resolved_output / config.report.manifest_filename
    plot_outcome_vs_credit(
        np.asarray(plot_outcomes),
        np.asarray(plot_credits),
        plot_path,
        labels=plot_labels,
        outcome_label="Terminal outcome AUROC (behavioral proxy)",
        credit_label="Oracle utility-credit Spearman",
        title="Outcome accuracy vs. credit recovery across four families (Gate B)",
    )

    elapsed = time.perf_counter() - started
    metrics["runtime_seconds"] = elapsed
    scientific_metrics = {key: value for key, value in metrics.items() if key != "runtime_seconds"}
    metrics["scientific_metrics_sha256"] = sha256_json(scientific_metrics)
    write_metrics_json(metrics, metrics_path)
    manifest = build_run_manifest(
        repository=repository,
        resolved_config=dump_resolved_config(config),
        artifacts={"metrics": metrics_path.name, "plot": plot_path.name},
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ExperimentResult(
        metrics=metrics,
        output_dir=resolved_output,
        artifacts={"metrics": metrics_path, "plot": plot_path, "manifest": manifest_path},
    )
