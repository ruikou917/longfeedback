"""Validated configuration for reproducible LongFeedback experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base configuration model that rejects misspelled settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentSettings(StrictModel):
    """Settings controlling dataset generation and reproducibility."""

    name: Literal["e0"] = "e0"
    seed: int = 0
    episodes: int = Field(default=512, ge=32)
    horizon: int = Field(default=8, ge=2, le=128)
    train_fraction: float = Field(default=0.8, gt=0.5, lt=1.0)
    output_dir: Path = Path("artifacts/e0")


class WorldSettings(StrictModel):
    """Parameters for deterministic World A (fatigue and habit)."""

    habit_decay: float = Field(default=0.90, ge=0.0, le=1.0)
    fatigue_decay: float = Field(default=0.80, ge=0.0, le=1.0)
    fatigue_response_penalty: float = Field(default=0.80, ge=0.0)
    fatigue_utility_cost: float = Field(default=0.08, ge=0.0)
    action_utility_cost: float = Field(default=0.04, ge=0.0)
    proxy_threshold: float = Field(default=1.73, ge=0.0)
    behavior_epsilon: float = Field(default=0.25, ge=0.0, le=1.0)


class OracleSettings(StrictModel):
    """Settings for paired counterfactual credit estimation."""

    continuation_mode: Literal["frozen"] = "frozen"
    mc_rollouts: int = Field(default=16, ge=2)
    max_examples: int = Field(default=4096, ge=32)
    reference_action: int = Field(default=0, ge=0)


class BaselineSettings(StrictModel):
    """Settings for lightweight diagnostic baselines."""

    ridge_alpha: float = Field(default=1.0e-8, ge=0.0)


class ReportSettings(StrictModel):
    """Names of immutable artifacts emitted by an E0 run."""

    metrics_filename: str = "metrics.json"
    predictions_filename: str = "predictions.csv"
    plot_filename: str = "outcome_vs_credit.png"
    manifest_filename: str = "run_manifest.json"

    @model_validator(mode="after")
    def filenames_are_local(self) -> ReportSettings:
        for value in (
            self.metrics_filename,
            self.predictions_filename,
            self.plot_filename,
            self.manifest_filename,
        ):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("report filenames must stay inside the output directory")
        return self


class E0Config(StrictModel):
    """Complete configuration for the E0 scientific vertical slice."""

    experiment: ExperimentSettings = ExperimentSettings()
    world: WorldSettings = WorldSettings()
    oracle: OracleSettings = OracleSettings()
    baselines: BaselineSettings = BaselineSettings()
    report: ReportSettings = ReportSettings()


class GateAExperimentSettings(StrictModel):
    """Reproducibility settings for the Gate A experiment."""

    name: Literal["gate_a"] = "gate_a"
    seed: int = 0
    train_fraction: float = Field(default=0.8, gt=0.5, lt=1.0)
    output_dir: Path = Path("artifacts/gate_a")


class GateAWorldASettings(StrictModel):
    """Stochastic World A dataset regimes (fatigue and habit)."""

    episodes: int = Field(default=384, ge=16)
    horizon: int = Field(default=12, ge=2, le=64)
    observabilities: tuple[Literal["oracle", "noisy", "partial"], ...] = ("oracle", "partial")
    response_noise_std: float = Field(default=0.6, ge=0.0)
    habit_noise_std: float = Field(default=0.05, ge=0.0)
    fatigue_noise_std: float = Field(default=0.05, ge=0.0)
    observation_noise_std: float = Field(default=0.1, ge=0.0)
    proxy_threshold: float = Field(default=2.1, ge=0.0)
    behavior_epsilon: float = Field(default=0.3, ge=0.0, le=1.0)
    response_seek_threshold: float = Field(default=0.55, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def observabilities_unique(self) -> GateAWorldASettings:
        if not self.observabilities:
            raise ValueError("at least one World A observability regime is required")
        if len(set(self.observabilities)) != len(self.observabilities):
            raise ValueError("World A observability regimes must be unique")
        return self


class GateAWorldBSettings(StrictModel):
    """World B dataset regimes (hidden intent with confounded logging)."""

    episodes: int = Field(default=384, ge=16)
    horizon: int = Field(default=12, ge=2, le=64)
    regimes: tuple[Literal["clean", "confounded"], ...] = ("clean", "confounded")
    stay_probability: float = Field(default=0.85, ge=0.0, le=1.0)
    match_logit: float = 1.2
    mismatch_logit: float = -1.8
    progress_shock_scale: float = Field(default=1.0, ge=0.0)
    proxy_threshold: float = 6.5
    signal_accuracy: float = Field(default=0.85, ge=0.0, le=1.0)
    clean_epsilon: float = Field(default=0.3, ge=0.0, le=1.0)
    privileged_epsilon: float = Field(default=0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def regimes_unique(self) -> GateAWorldBSettings:
        if not self.regimes:
            raise ValueError("at least one World B regime is required")
        if len(set(self.regimes)) != len(self.regimes):
            raise ValueError("World B regimes must be unique")
        return self


class GateAOracleSettings(StrictModel):
    """Adaptive paired Monte Carlo oracle labeling for stochastic worlds."""

    continuation_mode: Literal["frozen"] = "frozen"
    reference_action: int = Field(default=0, ge=0)
    initial_rollouts: int = Field(default=16, ge=2)
    max_rollouts: int = Field(default=64, ge=2)
    se_threshold: float = Field(default=0.15, gt=0.0)
    label_train_episodes: int = Field(default=160, ge=8)
    stability_examples: int = Field(default=96, ge=8)
    stability_seed_offset: int = Field(default=100_003, ge=1)

    @model_validator(mode="after")
    def rollouts_ordered(self) -> GateAOracleSettings:
        if self.max_rollouts < self.initial_rollouts:
            raise ValueError("max_rollouts cannot be below initial_rollouts")
        return self


class GateAModelSettings(StrictModel):
    """Shared capacity for every DOCM variant."""

    d_model: int = Field(default=64, ge=8)
    n_layers: int = Field(default=2, ge=1)
    n_heads: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def heads_divide_model(self) -> GateAModelSettings:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self


class GateATrainingSettings(StrictModel):
    epochs: int = Field(default=40, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=1.0e-3, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    grad_clip: float = Field(default=1.0, gt=0.0)


class GateAPolicyCheckSettings(StrictModel):
    """Discrete policy sanity check for Gate A criterion 3."""

    regime: str = "world_a_oracle"
    evaluation_episodes: int = Field(default=256, ge=16)
    bc_epochs: int = Field(default=120, ge=1)
    bc_learning_rate: float = Field(default=0.05, gt=0.0)


class GateADecisionSettings(StrictModel):
    """Thresholds for the three Gate A decision criteria."""

    outcome_auroc_tolerance: float = Field(default=0.05, gt=0.0)
    credit_spearman_gap: float = Field(default=0.15, gt=0.0)
    stability_pearson_threshold: float = Field(default=0.95, gt=0.0, le=1.0)
    policy_utility_margin: float = Field(default=0.0, ge=0.0)


class GateAReportSettings(StrictModel):
    """Names of immutable artifacts emitted by a Gate A run."""

    metrics_filename: str = "metrics.json"
    predictions_filename: str = "predictions.csv"
    plot_filename: str = "outcome_vs_credit.png"
    manifest_filename: str = "run_manifest.json"

    @model_validator(mode="after")
    def filenames_are_local(self) -> GateAReportSettings:
        for value in (
            self.metrics_filename,
            self.predictions_filename,
            self.plot_filename,
            self.manifest_filename,
        ):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("report filenames must stay inside the output directory")
        return self


class GateAConfig(StrictModel):
    """Complete configuration for the Gate A experiment."""

    experiment: GateAExperimentSettings = GateAExperimentSettings()
    world_a: GateAWorldASettings = GateAWorldASettings()
    world_b: GateAWorldBSettings = GateAWorldBSettings()
    oracle: GateAOracleSettings = GateAOracleSettings()
    model: GateAModelSettings = GateAModelSettings()
    training: GateATrainingSettings = GateATrainingSettings()
    policy_check: GateAPolicyCheckSettings = GateAPolicyCheckSettings()
    decision: GateADecisionSettings = GateADecisionSettings()
    report: GateAReportSettings = GateAReportSettings()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return raw


def load_e0_config(path: Path) -> E0Config:
    """Load and validate an E0 YAML file."""

    return E0Config.model_validate(_load_yaml_mapping(path))


def load_gate_a_config(path: Path) -> GateAConfig:
    """Load and validate a Gate A YAML file."""

    return GateAConfig.model_validate(_load_yaml_mapping(path))


def dump_resolved_config(config: E0Config | GateAConfig) -> dict[str, Any]:
    """Return a JSON-safe representation suitable for run manifests."""

    return config.model_dump(mode="json")
