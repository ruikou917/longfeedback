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


class LmsysDataConfig(StrictModel):
    """Configuration for preparing the local LMSYS-Chat-1M snapshot."""

    input_dir: Path = Path("data/lmsys-chat-data")
    output_dir: Path = Path("data/processed/lmsys")
    max_conversations: int = Field(default=20_000, ge=0, description="0 keeps every row")
    language: str | None = "English"
    min_assistant_turns: int = Field(default=3, ge=1)
    max_message_chars: int = Field(default=8_000, ge=1)
    exclude_flagged: bool = True
    include_redacted: bool = True
    seed: int = 0
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    events_filename: str = "events.parquet"
    manifest_filename: str = "source_manifest.json"
    stats_filename: str = "stats.json"

    @model_validator(mode="after")
    def fractions_leave_test_split(self) -> LmsysDataConfig:
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train and validation fractions must leave room for test")
        for value in (self.events_filename, self.manifest_filename, self.stats_filename):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("artifact filenames must stay inside the output directory")
        return self


class WildChatDataConfig(StrictModel):
    """Configuration for preparing the local WildChat-1M snapshot.

    WildChat is the project's *primary* conversational source (ODC-BY,
    ungated); LMSYS-Chat-1M is the secondary replication source. See
    docs/data_governance.md.
    """

    input_dir: Path = Path("data/wildchat-data")
    output_dir: Path = Path("data/processed/wildchat")
    max_conversations: int = Field(default=20_000, ge=0, description="0 keeps every row")
    language: str | None = "English"
    min_assistant_turns: int = Field(default=3, ge=1)
    max_message_chars: int = Field(default=8_000, ge=1)
    exclude_flagged: bool = True
    exclude_toxic: bool = True
    include_redacted: bool = True
    seed: int = 0
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    events_filename: str = "events.parquet"
    manifest_filename: str = "source_manifest.json"
    stats_filename: str = "stats.json"

    @model_validator(mode="after")
    def fractions_leave_test_split(self) -> WildChatDataConfig:
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train and validation fractions must leave room for test")
        for value in (self.events_filename, self.manifest_filename, self.stats_filename):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("artifact filenames must stay inside the output directory")
        return self


class KuaiRandDataConfig(StrictModel):
    """Configuration for preparing a local KuaiRand-Pure snapshot (E6).

    Only ``log_random_files`` carry a known-uniform exposure policy; every
    other logged file is production-confounded like WildChat/LMSYS. See
    ``docs/scientific_contract.md`` ("E6 acceptance contract") and ADR-008/009.
    """

    input_dir: Path = Path("data/kuairand-data")
    output_dir: Path = Path("data/processed/kuairand")
    log_random_files: tuple[str, ...] = ("log_random_4_22_to_5_08_pure.csv",)
    log_standard_files: tuple[str, ...] = (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    max_rows_per_file: int = Field(default=200_000, ge=0, description="0 keeps every row")
    seed: int = 0
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    events_filename: str = "events.parquet"
    manifest_filename: str = "source_manifest.json"
    stats_filename: str = "stats.json"

    @model_validator(mode="after")
    def fractions_leave_test_split(self) -> KuaiRandDataConfig:
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train and validation fractions must leave room for test")
        for value in (self.events_filename, self.manifest_filename, self.stats_filename):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("artifact filenames must stay inside the output directory")
        if not self.log_random_files:
            raise ValueError("at least one log_random file is required for the randomized bridge")
        return self


class GateBExperimentSettings(StrictModel):
    """Reproducibility settings for the Gate B experiment."""

    name: Literal["gate_b"] = "gate_b"
    seed: int = 0
    train_fraction: float = Field(default=0.8, gt=0.5, lt=1.0)
    output_dir: Path = Path("artifacts/gate_b")


class GateBFamilySettings(StrictModel):
    """Per-family dataset sizes; world parameters live in the experiment."""

    episodes: int = Field(default=256, ge=16)
    label_train_episodes: int = Field(default=96, ge=8)
    shift_episodes: int = Field(default=128, ge=16)


class GateBDecisionSettings(StrictModel):
    """Thresholds for the five Gate B decision criteria."""

    credit_spearman_margin: float = Field(default=0.02, ge=0.0)
    min_winning_families: int = Field(default=3, ge=1, le=4)
    uncertainty_spearman_threshold: float = Field(default=0.2, gt=0.0)
    error_detection_auroc_threshold: float = Field(default=0.6, gt=0.5)
    require_real_log: bool = True
    e1_metrics_path: Path = Path("artifacts/e1/metrics.json")
    # Leave-one-family-out transfer: a family-agnostic (z-scored) uncertainty
    # threshold is fit on three families and applied, unadjusted, to the
    # fourth. min_winning_transfers of 4 must beat chance by this margin.
    transfer_balanced_accuracy_margin: float = Field(default=0.05, ge=0.0, lt=0.5)
    min_winning_transfers: int = Field(default=3, ge=1, le=4)


class GateBConfig(StrictModel):
    """Complete configuration for the Gate B experiment."""

    experiment: GateBExperimentSettings = GateBExperimentSettings()
    families: GateBFamilySettings = GateBFamilySettings()
    oracle: GateAOracleSettings = GateAOracleSettings()
    model: GateAModelSettings = GateAModelSettings()
    training: GateATrainingSettings = GateATrainingSettings()
    ensemble_members: int = Field(default=5, ge=2)
    decision: GateBDecisionSettings = GateBDecisionSettings()
    report: GateAReportSettings = GateAReportSettings()


class E1Config(StrictModel):
    """Real-log delayed-outcome prediction on prepared conversation events."""

    processed_dir: Path = Path("data/processed/lmsys")
    events_filename: str = "events.parquet"
    output_dir: Path = Path("artifacts/e1")
    seed: int = 0
    window: int = Field(default=4, ge=1, le=16)
    repetition_threshold: float = Field(default=0.6, gt=0.0, le=1.0)
    max_train_examples: int = Field(default=30_000, ge=100)
    max_eval_examples: int = Field(default=10_000, ge=100)
    ridge_alpha: float = Field(default=1.0, ge=0.0)
    d_model: int = Field(default=32, ge=8)
    n_layers: int = Field(default=1, ge=1)
    n_heads: int = Field(default=2, ge=1)
    epochs: int = Field(default=8, ge=1)
    batch_size: int = Field(default=256, ge=1)
    learning_rate: float = Field(default=1.0e-3, gt=0.0)
    min_auroc: float = Field(default=0.55, gt=0.5, lt=1.0)
    auroc_margin_over_trivial: float = Field(default=0.02, gt=0.0)
    metrics_filename: str = "metrics.json"
    predictions_filename: str = "predictions.csv"
    manifest_filename: str = "run_manifest.json"

    @model_validator(mode="after")
    def validate_model_shape(self) -> E1Config:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self


class E6Config(StrictModel):
    """Randomized-bridge bias comparison on prepared KuaiRand events (E6).

    Compares a per-video engagement rate estimated from the confounded
    ``standard`` log against the unbiased rate observed on the genuinely
    randomized ``random`` log for the same videos. See
    ``docs/scientific_contract.md`` ("E6 acceptance contract").
    """

    processed_dir: Path = Path("data/processed/kuairand")
    events_filename: str = "events.parquet"
    output_dir: Path = Path("artifacts/e6")
    seed: int = 0
    min_exposures_per_video: int = Field(default=5, ge=1)
    bias_detection_threshold: float = Field(default=0.02, gt=0.0)
    rank_correlation_threshold: float = Field(default=0.2, gt=0.0, lt=1.0)
    # Feature-adjustment comparison (does conditioning on real confounders,
    # not just naive log-averaging, reduce the measured bias?). Only
    # user/video *content* features are used, never video_features_statistic
    # -- those are aggregated engagement counts and would leak the label.
    raw_dir: Path = Path("data/kuairand-data")
    user_features_filename: str = "user_features_pure.csv"
    video_features_filename: str = "video_features_basic_pure.csv"
    ridge_alpha: float = Field(default=1.0, ge=0.0)
    feature_adjustment_improvement_margin: float = Field(default=0.0, ge=0.0)
    metrics_filename: str = "metrics.json"
    predictions_filename: str = "predictions.csv"
    manifest_filename: str = "run_manifest.json"


class E5RewardModelSettings(StrictModel):
    """Learned proxy reward models for the overoptimization study."""

    d_model: int = Field(default=32, ge=8)
    n_layers: int = Field(default=1, ge=1)
    n_heads: int = Field(default=2, ge=1)
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=1.0e-3, gt=0.0)
    ensemble_members: int = Field(default=5, ge=2)
    lcb_lambda: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def heads_divide_model(self) -> E5RewardModelSettings:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self


class E5OptimizationSettings(StrictModel):
    """REINFORCE budget and regularization."""

    updates: int = Field(default=240, ge=1)
    checkpoint_every: int = Field(default=15, ge=1)
    batch_episodes: int = Field(default=32, ge=4)
    learning_rate: float = Field(default=0.05, gt=0.0)
    entropy_coefficient: float = Field(default=0.01, ge=0.0)
    kl_beta: float = Field(default=0.3, gt=0.0)
    behavior_clone_epochs: int = Field(default=120, ge=1)
    behavior_clone_learning_rate: float = Field(default=0.05, gt=0.0)


class E5DecisionSettings(StrictModel):
    """Thresholds for the E5 decision criteria."""

    hacking_gap_threshold: float = Field(default=0.5, gt=0.0)
    mitigation_margin: float = Field(default=0.1, ge=0.0)
    rm_error_active_threshold: float = Field(default=0.05, gt=0.0)


class E5LoggingRegime(StrictModel):
    """One logged-data support regime (design doc logging.support switch)."""

    name: str = Field(min_length=1)
    episodes: int = Field(default=384, ge=32)
    behavior_epsilon: float = Field(default=0.3, ge=0.0, le=1.0)


class E5Config(StrictModel):
    """Complete configuration for the E5 overoptimization experiment."""

    name: Literal["e5"] = "e5"
    seed: int = 0
    train_fraction: float = Field(default=0.8, gt=0.5, lt=1.0)
    output_dir: Path = Path("artifacts/e5")
    logging_regimes: tuple[E5LoggingRegime, ...] = (
        E5LoggingRegime(name="broad", episodes=384, behavior_epsilon=0.3),
        E5LoggingRegime(name="narrow", episodes=96, behavior_epsilon=0.1),
    )
    evaluation_episodes: int = Field(default=128, ge=16)
    reward_model: E5RewardModelSettings = E5RewardModelSettings()
    optimization: E5OptimizationSettings = E5OptimizationSettings()
    decision: E5DecisionSettings = E5DecisionSettings()

    @model_validator(mode="after")
    def regimes_unique(self) -> E5Config:
        names = [regime.name for regime in self.logging_regimes]
        if not names or len(set(names)) != len(names):
            raise ValueError("logging regimes must be non-empty and uniquely named")
        return self

    metrics_filename: str = "metrics.json"
    predictions_filename: str = "predictions.csv"
    plot_filename: str = "optimization_curves.png"
    manifest_filename: str = "run_manifest.json"


class MultiSeedBootstrapSettings(StrictModel):
    """Percentile-bootstrap settings for across-seed confidence intervals."""

    resamples: int = Field(default=10_000, ge=100)
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    seed: int = 0


class MultiSeedConfig(StrictModel):
    """Multi-seed statistical protocol over Gate B and E5 (design doc 13.5).

    Reruns the embedded experiment configurations once per seed and reports
    the predeclared primary metrics with percentile-bootstrap confidence
    intervals over seeds. Comparisons are paired within a seed (variants of
    one run already share evaluation seeds), so the bootstrap resamples
    per-seed paired differences, and every effect is reported in the metric's
    native units rather than as a bare p-value.
    """

    name: Literal["multiseed"] = "multiseed"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    # The design-doc protocol requires at least five seeds; fewer seeds still
    # run (for smoke tests) but the decision block records the protocol as
    # unmet.
    min_seeds: int = Field(default=5, ge=2)
    experiments: tuple[Literal["gate_b", "e5"], ...] = ("gate_b", "e5")
    gate_b: GateBConfig = GateBConfig()
    e5: E5Config = E5Config()
    bootstrap: MultiSeedBootstrapSettings = MultiSeedBootstrapSettings()
    output_dir: Path = Path("artifacts/multiseed")
    metrics_filename: str = "metrics.json"
    seed_table_filename: str = "seed_metrics.csv"
    manifest_filename: str = "run_manifest.json"

    @model_validator(mode="after")
    def seeds_distinct(self) -> MultiSeedConfig:
        if len(self.seeds) < 2:
            raise ValueError("at least two seeds are required for across-seed statistics")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        if not self.experiments or len(set(self.experiments)) != len(self.experiments):
            raise ValueError("experiments must be non-empty and unique")
        return self


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


def load_lmsys_data_config(path: Path) -> LmsysDataConfig:
    """Load and validate an LMSYS data-preparation YAML file."""

    return LmsysDataConfig.model_validate(_load_yaml_mapping(path))


def load_wildchat_data_config(path: Path) -> WildChatDataConfig:
    """Load and validate a WildChat data-preparation YAML file."""

    return WildChatDataConfig.model_validate(_load_yaml_mapping(path))


def load_e1_config(path: Path) -> E1Config:
    """Load and validate an E1 real-log YAML file."""

    return E1Config.model_validate(_load_yaml_mapping(path))


def load_gate_b_config(path: Path) -> GateBConfig:
    """Load and validate a Gate B YAML file."""

    return GateBConfig.model_validate(_load_yaml_mapping(path))


def load_e5_config(path: Path) -> E5Config:
    """Load and validate an E5 YAML file."""

    return E5Config.model_validate(_load_yaml_mapping(path))


def load_kuairand_data_config(path: Path) -> KuaiRandDataConfig:
    """Load and validate a KuaiRand data-preparation YAML file."""

    return KuaiRandDataConfig.model_validate(_load_yaml_mapping(path))


def load_e6_config(path: Path) -> E6Config:
    """Load and validate an E6 YAML file."""

    return E6Config.model_validate(_load_yaml_mapping(path))


def load_multiseed_config(path: Path) -> MultiSeedConfig:
    """Load and validate a multi-seed protocol YAML file."""

    return MultiSeedConfig.model_validate(_load_yaml_mapping(path))


def dump_resolved_config(config: StrictModel) -> dict[str, Any]:
    """Return a JSON-safe representation suitable for run manifests."""

    return config.model_dump(mode="json")
