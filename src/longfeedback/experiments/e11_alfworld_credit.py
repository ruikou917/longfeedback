"""E11: replay-verified action credit in a resettable text environment.

Estimation question: can the candidate DOCM recover replay-based action credit
at held-out states while preserving terminal-outcome quality? The CPU smoke
profile runs the full pipeline on the deterministic fake world with a mock
actor; the ALFWorld backends plug in through the same environment protocol.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, Field

from longfeedback.actors.base import CandidatePolicy
from longfeedback.actors.mock import MockCandidatePolicy
from longfeedback.budget import BudgetLedger, BudgetLimits
from longfeedback.credit.branching import (
    BranchGenerationSettings,
    BranchSelectionRule,
    BranchTargetRow,
    CandidateRule,
    EpisodeRecord,
    collect_episode,
    derive_seed,
    generate_branch_targets,
    replay_audit,
)
from longfeedback.credit.c3 import c3_estimates
from longfeedback.credit.gigpo import GigpoSettings, anchor_diagnostics, gigpo_step_credits
from longfeedback.environments.base import EnvironmentClient
from longfeedback.environments.fake import FakeTextEnvironment, FakeWorldSettings
from longfeedback.evaluation import write_metrics_json
from longfeedback.experiments.e9 import _cluster_mean_ci, _repository_root
from longfeedback.experiments.manifest import build_run_manifest, sha256_json
from longfeedback.experiments.types import ExperimentResult
from longfeedback.models.candidate_data import (
    CandidateSequenceDataset,
    assert_no_locked_rows,
    build_candidate_dataset,
)
from longfeedback.models.candidate_docm import (
    CANDIDATE_VARIANTS,
    CandidateDelayedOutcomeCreditModel,
    CandidateTrainingSettings,
    candidate_variant_spec,
)
from longfeedback.models.encoders import EncoderArchitecture
from longfeedback.models.text_embeddings import HashedTextEmbedder
from longfeedback.models.uncertainty import CriticEnsemble, coverage_report, fit_conformal

FloatArray = npt.NDArray[np.float64]

PRIMARY_VARIANTS = ("docm_outcome", "docm_prefix", "docm_dueling_credit")


class FakeEnvironmentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    train_games: int = Field(ge=1)
    valid_seen_games: int = Field(ge=1)
    valid_unseen_games: int = Field(ge=1)
    code_length: int = Field(ge=1)
    noop_actions: int = Field(ge=0)
    trap_actions: int = Field(ge=0)
    max_steps: int = Field(ge=1)


class EnvironmentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["fake", "alfworld_inprocess", "alfworld_subprocess"]
    fake: FakeEnvironmentSettings | None = None
    data_dir: Path | None = None
    max_steps: int = Field(ge=1)
    request_timeout_seconds: float = Field(default=120.0, gt=0.0)


class ActorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["mock"]
    seed: int
    temperature: float = Field(gt=0.0)
    noise_scale: float = Field(ge=0.0)
    prompt_noise_scale: float = Field(ge=0.0)
    verb_bias: dict[str, float] = Field(default_factory=dict)


class CollectionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_episodes: int = Field(ge=1)
    calibration_episodes: int = Field(ge=1)
    reference_episodes: int = Field(ge=1)
    branch_states_per_episode: int = Field(ge=1)
    uniform_selection_weight: float = Field(ge=0.0, le=1.0)
    full_enumeration_limit: int = Field(ge=1)
    top_actor_candidates: int = Field(ge=0)
    random_candidates: int = Field(ge=0)
    train_rollouts_per_action: int = Field(ge=1)
    unforced_rollouts: int = Field(ge=1)
    child_direct_value_fraction: float = Field(ge=0.0, le=1.0)
    child_unforced_rollouts: int = Field(ge=0)
    calibration_rollouts_per_action: int = Field(ge=1)
    reference_rollouts_per_action: int = Field(ge=1)
    replay_audit_prefixes: int = Field(ge=1)


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding_dim: int = Field(ge=8)
    d_model: int = Field(ge=8)
    n_layers: int = Field(ge=1)
    n_heads: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)
    action_mlp_hidden: int = Field(ge=4)
    target_network_ema: float = Field(ge=0.0, lt=1.0)
    policy_center_tolerance: float = Field(gt=0.0)


class UncertaintySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    members: int = Field(ge=2)
    target_coverage: float = Field(gt=0.0, lt=1.0)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seeds: tuple[int, ...]
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    grad_clip: float = Field(gt=0.0)


class DecisionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_success_min: float = Field(ge=0.0, le=1.0)
    actor_success_max: float = Field(ge=0.0, le=1.0)
    reference_median_se_max: float = Field(gt=0.0)
    informative_q_spread: float = Field(gt=0.0)
    informative_state_fraction_min: float = Field(ge=0.0, le=1.0)
    minimum_reference_states: int = Field(ge=1)
    regret_relative_improvement: float = Field(ge=0.0)
    outcome_brier_tolerance: float = Field(ge=0.0)
    minimum_positive_seeds: int = Field(ge=1)
    bootstrap_resamples: int = Field(ge=100)
    familywise_confidence: float = Field(gt=0.5, lt=1.0)
    uncertainty_coverage_tolerance: float = Field(ge=0.0, le=1.0)


class BudgetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_actor_forward_tokens: int = Field(ge=1)
    max_environment_steps: int = Field(ge=1)
    max_wall_time_seconds: float = Field(gt=0.0)


class E11Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["e11_alfworld_credit"]
    seed: int
    output_dir: Path
    environment: EnvironmentSettings
    actor: ActorSettings
    collection: CollectionSettings
    model: ModelSettings
    uncertainty: UncertaintySettings
    training: TrainingConfig
    decision: DecisionSettings
    budget: BudgetSettings


def load_config(path: Path) -> E11Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("E11 config must be a YAML mapping")
    return E11Config.model_validate(raw)


def build_environment(config: EnvironmentSettings) -> EnvironmentClient:
    if config.backend == "fake":
        if config.fake is None:
            raise ValueError("fake backend requires environment.fake settings")
        return FakeTextEnvironment(
            FakeWorldSettings(
                train_games=config.fake.train_games,
                valid_seen_games=config.fake.valid_seen_games,
                valid_unseen_games=config.fake.valid_unseen_games,
                code_length=config.fake.code_length,
                noop_actions=config.fake.noop_actions,
                trap_actions=config.fake.trap_actions,
                max_steps=config.fake.max_steps,
            )
        )
    from longfeedback.environments.alfworld import (
        AlfworldSettings,
        InProcessAlfworldClient,
        SubprocessAlfworldClient,
    )

    if config.data_dir is None:
        raise ValueError("ALFWorld backends require environment.data_dir")
    settings = AlfworldSettings(
        data_dir=config.data_dir,
        max_steps=config.max_steps,
        request_timeout_seconds=config.request_timeout_seconds,
    )
    if config.backend == "alfworld_inprocess":
        return InProcessAlfworldClient(settings)
    return SubprocessAlfworldClient(settings)


def build_actor(config: ActorSettings) -> CandidatePolicy:
    return MockCandidatePolicy(
        seed=config.seed,
        temperature=config.temperature,
        noise_scale=config.noise_scale,
        prompt_noise_scale=config.prompt_noise_scale,
        verb_bias=dict(config.verb_bias),
    )


def _collect_split(
    client: EnvironmentClient,
    actor: CandidatePolicy,
    split: str,
    count: int,
    *,
    run_seed: int,
    max_steps: int,
    ledger: BudgetLedger,
) -> list[EpisodeRecord]:
    games = client.list_games(split)
    episodes = []
    for index in range(count):
        game = games[index % len(games)]
        episode_id = f"e11:{split}:s{run_seed}:e{index}"
        episodes.append(
            collect_episode(
                client,
                game,
                actor,
                episode_id=episode_id,
                environment_seed=derive_seed("e11-env", episode_id) % 2**31,
                max_steps=max_steps,
                ledger=ledger,
            )
        )
    return episodes


def _labeled_arrays(
    dataset: CandidateSequenceDataset,
    rows: list[BranchTargetRow],
) -> dict[str, npt.NDArray[Any]]:
    """Align labeled locked rows with dataset (branch, slot) coordinates."""

    grouped: dict[tuple[str, int], int] = {}
    keys = sorted({(row.episode_id, row.step_index) for row in rows})
    for index, key in enumerate(keys):
        grouped[key] = index
    row_branch: list[int] = []
    row_slot: list[int] = []
    q_ref: list[float] = []
    q_se: list[float] = []
    v_ref: list[float] = []
    games: list[str] = []
    logged: list[bool] = []
    counts = dataset.q_count
    labeled_slots: dict[tuple[str, int], dict[str, int]] = {}
    # Recover the slot of each labeled candidate from the dataset masks: rows
    # were placed by sorted candidate order inside build_candidate_dataset.
    for row in sorted(rows, key=lambda r: (r.episode_id, r.step_index, r.candidate_action)):
        branch = grouped[(row.episode_id, row.step_index)]
        slot_map = labeled_slots.setdefault((row.episode_id, row.step_index), {})
        if row.candidate_action not in slot_map:
            labeled = np.flatnonzero(counts[branch] > 0)
            ordered_actions = sorted(
                {
                    r.candidate_action
                    for r in rows
                    if (r.episode_id, r.step_index) == (row.episode_id, row.step_index)
                }
            )
            for action, slot in zip(ordered_actions, labeled.tolist(), strict=True):
                slot_map[action] = slot
        slot = slot_map[row.candidate_action]
        row_branch.append(branch)
        row_slot.append(slot)
        q_ref.append(row.q_hat)
        q_se.append(row.q_se)
        v_ref.append(row.v_hat)
        games.append(row.game_id)
        logged.append(row.logged_action)
    return {
        "branch": np.asarray(row_branch, dtype=np.int64),
        "slot": np.asarray(row_slot, dtype=np.int64),
        "q_ref": np.asarray(q_ref, dtype=np.float64),
        "q_se": np.asarray(q_se, dtype=np.float64),
        "v_ref": np.asarray(v_ref, dtype=np.float64),
        "games": np.asarray(games, dtype=np.str_),
        "logged": np.asarray(logged, dtype=np.bool_),
    }


def variant_candidate_scores(
    model: CandidateDelayedOutcomeCreditModel,
    dataset: CandidateSequenceDataset,
    variant: str,
) -> FloatArray:
    if variant in ("docm_dueling_credit", "docm_dueling_no_tree", "docm_independent_q"):
        return model.predict_branch_q(dataset)
    if variant == "docm_outcome":
        return model.predict_appended_candidate_scores(dataset, head="outcome")
    if variant == "docm_prefix":
        return model.predict_appended_candidate_scores(dataset, head="value")
    raise ValueError(f"unknown variant {variant!r}")


def _spearman(x: FloatArray, y: FloatArray) -> float:
    if x.size < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    sx = rx.std()
    sy = ry.std()
    if sx <= 0 or sy <= 0:
        return 0.0
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def _state_metrics(
    scores: FloatArray,
    labeled: dict[str, npt.NDArray[Any]],
) -> dict[str, npt.NDArray[Any]]:
    """Per-state regret / rank metrics over the labeled candidate set."""

    branches = np.unique(labeled["branch"])
    regrets: list[float] = []
    spearmans: list[float] = []
    top1: list[float] = []
    state_games: list[str] = []
    for branch in branches.tolist():
        select = labeled["branch"] == branch
        if int(select.sum()) < 2:
            continue
        q_ref = labeled["q_ref"][select]
        model_scores = scores[branch][labeled["slot"][select]]
        best_ref = float(q_ref.max())
        chosen = int(np.argmax(model_scores))
        regrets.append(best_ref - float(q_ref[chosen]))
        spearmans.append(_spearman(model_scores, q_ref))
        top1.append(float(np.argmax(q_ref) == chosen))
        state_games.append(str(labeled["games"][select][0]))
    return {
        "regret": np.asarray(regrets, dtype=np.float64),
        "spearman": np.asarray(spearmans, dtype=np.float64),
        "top1": np.asarray(top1, dtype=np.float64),
        "games": np.asarray(state_games, dtype=np.str_),
    }


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not records:
        records = [{}]
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def _episode_records(episodes: list[EpisodeRecord], split: str) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": episode.episode_id,
            "game_id": episode.game.game_id,
            "split": split,
            "task_type": "fake_panel" if episode.game.game_id.startswith("fake/") else "alfworld",
            "actor_policy_id": episode.actor_policy_id,
            "environment_seed": episode.environment_seed,
            "success": episode.success,
            "terminal_score": episode.terminal_score,
            "terminal_reason": episode.terminal_reason,
            "episode_steps": len(episode.steps),
            "actor_forward_tokens": episode.actor_forward_tokens,
            "prompt_template_hash": episode.steps[0].prompt_hash if episode.steps else "",
        }
        for episode in episodes
    ]


def _step_records(episodes: list[EpisodeRecord]) -> list[dict[str, Any]]:
    records = []
    for episode in episodes:
        for step in episode.steps:
            records.append(
                {
                    "episode_id": episode.episode_id,
                    "game_id": episode.game.game_id,
                    "step_index": step.step_index,
                    "goal": step.observation.goal,
                    "observation": step.observation.observation,
                    "admissible_actions_json": json.dumps(
                        list(step.observation.admissible_actions)
                    ),
                    "action": step.decision.action,
                    "action_probability": step.decision.probability,
                    "action_log_probability": step.decision.log_probability,
                    "actor_entropy": step.decision.entropy,
                    "prompt_hash": step.prompt_hash,
                    "state_hash": step.observation.state_hash,
                    "replay_prefix_hash": step.replay_handle.prefix_hash,
                    "next_observation": step.next_observation.observation,
                    "done": step.next_observation.done,
                    "score_after_step": step.next_observation.score,
                }
            )
    return records


def run_e11(config: E11Config, *, output_dir: Path | None = None) -> ExperimentResult:
    started = time.perf_counter()
    repository = _repository_root()
    resolved_output = (output_dir or config.output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    client = build_environment(config.environment)
    actor = build_actor(config.actor)
    embedder = HashedTextEmbedder(dim=config.model.embedding_dim)
    ledger = BudgetLedger(
        BudgetLimits(
            max_actor_forward_tokens=config.budget.max_actor_forward_tokens,
            max_environment_steps=config.budget.max_environment_steps,
            max_wall_time_seconds=config.budget.max_wall_time_seconds,
        )
    )
    collection = config.collection
    max_steps = config.environment.max_steps

    train_episodes = _collect_split(
        client,
        actor,
        "train",
        collection.base_episodes,
        run_seed=config.seed,
        max_steps=max_steps,
        ledger=ledger,
    )
    calibration_episodes = _collect_split(
        client,
        actor,
        "valid_seen",
        collection.calibration_episodes,
        run_seed=config.seed,
        max_steps=max_steps,
        ledger=ledger,
    )
    reference_episodes = _collect_split(
        client,
        actor,
        "valid_unseen",
        collection.reference_episodes,
        run_seed=config.seed,
        max_steps=max_steps,
        ledger=ledger,
    )

    audit = replay_audit(client, train_episodes, max_prefixes=collection.replay_audit_prefixes)
    actor_success = float(np.mean([episode.success for episode in train_episodes]))

    train_prefixes = {step.replay_handle.prefix_hash for e in train_episodes for step in e.steps}
    locked_prefixes = {
        step.replay_handle.prefix_hash for e in reference_episodes for step in e.steps
    }
    train_games = {episode.game.game_id for episode in train_episodes}
    locked_games = {episode.game.game_id for episode in reference_episodes}
    overlap_free = not (train_prefixes & locked_prefixes) and not (train_games & locked_games)

    selection_rule = BranchSelectionRule(
        states_per_episode=collection.branch_states_per_episode,
        uniform_weight=collection.uniform_selection_weight,
    )
    candidate_rule = CandidateRule(
        full_enumeration_limit=collection.full_enumeration_limit,
        top_actor_candidates=collection.top_actor_candidates,
        random_candidates=collection.random_candidates,
    )

    train_rows, train_distributions = generate_branch_targets(
        client,
        actor,
        train_episodes,
        BranchGenerationSettings(
            selection_rule=selection_rule,
            candidate_rule=candidate_rule,
            forced_rollouts=collection.train_rollouts_per_action,
            unforced_rollouts=collection.unforced_rollouts,
            child_value_fraction=collection.child_direct_value_fraction,
            child_rollouts=collection.child_unforced_rollouts,
            max_continuation_steps=max_steps,
            target_role="train",
        ),
        seed=f"e11:train:{config.seed}",
        ledger=ledger,
    )
    calibration_rows, calibration_distributions = generate_branch_targets(
        client,
        actor,
        calibration_episodes,
        BranchGenerationSettings(
            selection_rule=selection_rule,
            candidate_rule=candidate_rule,
            forced_rollouts=collection.calibration_rollouts_per_action,
            unforced_rollouts=collection.calibration_rollouts_per_action,
            child_value_fraction=0.0,
            child_rollouts=1,
            max_continuation_steps=max_steps,
            target_role="validation",
        ),
        seed=f"e11:calibration:{config.seed}",
        ledger=ledger,
    )
    locked_rows, locked_distributions = generate_branch_targets(
        client,
        actor,
        reference_episodes,
        BranchGenerationSettings(
            selection_rule=selection_rule,
            candidate_rule=candidate_rule,
            forced_rollouts=collection.reference_rollouts_per_action,
            unforced_rollouts=collection.reference_rollouts_per_action,
            child_value_fraction=0.0,
            child_rollouts=1,
            max_continuation_steps=max_steps,
            target_role="locked_reference",
        ),
        seed=f"e11:locked:{config.seed}",
        ledger=ledger,
    )
    # Direct C3 baseline: same locked states/candidates, independent low-K
    # rollouts matching the training budget per action.
    c3_rows, _ = generate_branch_targets(
        client,
        actor,
        reference_episodes,
        BranchGenerationSettings(
            selection_rule=selection_rule,
            candidate_rule=candidate_rule,
            forced_rollouts=collection.train_rollouts_per_action,
            unforced_rollouts=collection.unforced_rollouts,
            child_value_fraction=0.0,
            child_rollouts=1,
            max_continuation_steps=max_steps,
            target_role="validation",
        ),
        seed=f"e11:locked:{config.seed}",
        rollout_seed=f"e11:c3:{config.seed}",
        ledger=ledger,
    )

    budget_dimension = ledger.exceeded()
    run_complete = budget_dimension is None

    locked_labeled = [row for row in locked_rows if row.rollout_count > 0]
    locked_states = sorted({(row.episode_id, row.step_index) for row in locked_labeled})
    median_reference_se = (
        float(np.median([row.q_se for row in locked_labeled])) if locked_labeled else 1.0
    )
    spreads = []
    for state in locked_states:
        state_qs = [
            row.q_hat for row in locked_labeled if (row.episode_id, row.step_index) == state
        ]
        spreads.append(max(state_qs) - min(state_qs))
    informative_fraction = (
        float(np.mean([spread >= config.decision.informative_q_spread for spread in spreads]))
        if spreads
        else 0.0
    )

    decision = config.decision
    signal_gate = {
        "replay_match_rate": audit["replay_match_rate"],
        "replay_pass": audit["replay_match_rate"] == 1.0,
        "split_overlap_free": overlap_free,
        "actor_success_rate": actor_success,
        "actor_success_in_range": bool(
            decision.actor_success_min <= actor_success <= decision.actor_success_max
        ),
        "median_reference_q_se": median_reference_se,
        "reference_se_pass": median_reference_se <= decision.reference_median_se_max,
        "reference_states": len(locked_states),
        "reference_states_pass": len(locked_states) >= decision.minimum_reference_states,
        "informative_state_fraction": informative_fraction,
        "informative_pass": informative_fraction >= decision.informative_state_fraction_min,
        "budget_complete": run_complete,
    }
    signal_gate["stage1_pass"] = bool(
        signal_gate["replay_pass"]
        and signal_gate["split_overlap_free"]
        and signal_gate["actor_success_in_range"]
        and signal_gate["reference_se_pass"]
        and signal_gate["reference_states_pass"]
        and signal_gate["informative_pass"]
        and run_complete
    )

    horizon = max_steps
    assert_no_locked_rows(train_rows)
    train_dataset = build_candidate_dataset(
        train_episodes, train_rows, train_distributions, embedder, horizon=horizon
    )
    calibration_dataset = build_candidate_dataset(
        calibration_episodes,
        calibration_rows,
        calibration_distributions,
        embedder,
        horizon=horizon,
    )
    locked_dataset = build_candidate_dataset(
        reference_episodes, locked_rows, locked_distributions, embedder, horizon=horizon
    )
    locked_arrays = _labeled_arrays(locked_dataset, locked_labeled)
    calibration_labeled = [row for row in calibration_rows if row.rollout_count > 0]
    calibration_arrays = _labeled_arrays(calibration_dataset, calibration_labeled)

    architecture = EncoderArchitecture(
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        dropout=config.model.dropout,
    )
    training_settings = CandidateTrainingSettings(
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        grad_clip=config.training.grad_clip,
    )

    locked_success = np.asarray(
        [float(episode.success) for episode in reference_episodes], dtype=np.float64
    )

    per_seed: dict[str, Any] = {}
    positive = {"outcome": 0, "prefix": 0}
    adverse = {"outcome": 0, "prefix": 0}
    parameter_counts: dict[str, int] = {}
    seed0_scores: dict[str, FloatArray] = {}
    seed0_models: dict[str, CandidateDelayedOutcomeCreditModel] = {}
    pairwise_confidence = 1.0 - (1.0 - decision.familywise_confidence) / 2.0

    for seed in config.training.seeds:
        variant_results: dict[str, Any] = {}
        squared_errors: dict[str, FloatArray] = {}
        regrets: dict[str, FloatArray] = {}
        state_games: npt.NDArray[np.str_] | None = None
        for variant in CANDIDATE_VARIANTS:
            weights, parameterization = candidate_variant_spec(variant)
            model = CandidateDelayedOutcomeCreditModel(
                state_dim=embedder.dimension,
                action_dim=embedder.dimension,
                max_horizon=horizon,
                architecture=architecture,
                loss_weights=weights,
                action_value_parameterization=parameterization,
                action_mlp_hidden=config.model.action_mlp_hidden,
                target_network_ema=config.model.target_network_ema,
                policy_center_tolerance=config.model.policy_center_tolerance,
                seed=seed,
            )
            model.fit(train_dataset, training=training_settings)
            parameter_counts[variant] = model.parameter_count()
            scores = variant_candidate_scores(model, locked_dataset, variant)
            picked = scores[locked_arrays["branch"], locked_arrays["slot"]]
            squared = np.square(picked - locked_arrays["q_ref"])
            squared_errors[variant] = squared
            state_summary = _state_metrics(scores, locked_arrays)
            regrets[variant] = state_summary["regret"]
            state_games = state_summary["games"]
            terminal = model.predict_terminal_probability(locked_dataset)
            brier = float(np.mean(np.square(terminal - locked_success)))
            values = model.predict_branch_value(locked_dataset)
            branch_v_ref = np.zeros(locked_dataset.branches, dtype=np.float64)
            branch_v_ref[locked_arrays["branch"]] = locked_arrays["v_ref"]
            mse_v = float(np.mean(np.square(values - branch_v_ref)))
            residual = (
                model.policy_centering_residual(locked_dataset)
                if parameterization == "policy_centered_dueling"
                else None
            )
            variant_results[variant] = {
                "q_mse": float(np.mean(squared)),
                "mean_regret": (
                    float(np.mean(state_summary["regret"])) if state_summary["regret"].size else 0.0
                ),
                "mean_within_state_spearman": (
                    float(np.mean(state_summary["spearman"]))
                    if state_summary["spearman"].size
                    else 0.0
                ),
                "top1_agreement": (
                    float(np.mean(state_summary["top1"])) if state_summary["top1"].size else 0.0
                ),
                "terminal_brier": brier,
                "v_mse": mse_v,
                "policy_centering_residual": residual,
            }
            if seed == config.training.seeds[0]:
                seed0_scores[variant] = scores
                seed0_models[variant] = model

        comparisons: dict[str, Any] = {}
        for baseline_name, variant in (("outcome", "docm_outcome"), ("prefix", "docm_prefix")):
            gap = squared_errors[variant] - squared_errors["docm_dueling_credit"]
            summary = _cluster_mean_ci(
                gap,
                locked_arrays["games"],
                resamples=decision.bootstrap_resamples,
                confidence=pairwise_confidence,
                seed=derive_seed("e11-boot", str(seed), baseline_name) % 2**31,
            )
            positive[baseline_name] += int(summary["estimate"] > 0.0)
            adverse[baseline_name] += int(summary["ci_high"] < 0.0)
            comparisons[f"{baseline_name}_minus_credit_q_mse"] = summary
        per_seed[str(seed)] = {
            "variants": variant_results,
            "paired_q_mse_gaps": comparisons,
        }
        if seed == config.training.seeds[0]:
            per_seed[str(seed)]["regret_comparison"] = {}
            best_predictive = min(
                ("docm_outcome", "docm_prefix"),
                key=lambda name: float(np.mean(regrets[name])) if regrets[name].size else 0.0,
            )
            if regrets["docm_dueling_credit"].size and state_games is not None:
                regret_gap = regrets[best_predictive] - regrets["docm_dueling_credit"]
                per_seed[str(seed)]["regret_comparison"] = {
                    "best_predictive_baseline": best_predictive,
                    "gap_summary": _cluster_mean_ci(
                        regret_gap,
                        state_games.astype(np.str_),
                        resamples=decision.bootstrap_resamples,
                        confidence=decision.familywise_confidence,
                        seed=derive_seed("e11-regret", str(seed)) % 2**31,
                    ),
                }

    primary_seed = str(config.training.seeds[0])
    primary = per_seed[primary_seed]
    capacity_matched = len({parameter_counts[name] for name in PRIMARY_VARIANTS}) == 1
    primary_positive = {
        name: bool(
            primary["paired_q_mse_gaps"][f"{name}_minus_credit_q_mse"]["estimate"] > 0.0
            and primary["paired_q_mse_gaps"][f"{name}_minus_credit_q_mse"]["ci_low"] > 0.0
        )
        for name in ("outcome", "prefix")
    }
    credit_metrics = primary["variants"]["docm_dueling_credit"]
    outcome_metrics = primary["variants"]["docm_outcome"]
    prefix_metrics = primary["variants"]["docm_prefix"]
    best_predictive_regret = min(outcome_metrics["mean_regret"], prefix_metrics["mean_regret"])
    regret_pass = False
    regret_summary = primary.get("regret_comparison", {}).get("gap_summary")
    if best_predictive_regret > 0.0 and regret_summary is not None:
        relative = 1.0 - credit_metrics["mean_regret"] / best_predictive_regret
        regret_pass = bool(
            relative >= decision.regret_relative_improvement and regret_summary["ci_low"] > 0.0
        )
    brier_pass = bool(
        credit_metrics["terminal_brier"]
        <= outcome_metrics["terminal_brier"] + decision.outcome_brier_tolerance
    )
    robust = {
        name: bool(positive[name] >= decision.minimum_positive_seeds and adverse[name] == 0)
        for name in ("outcome", "prefix")
    }
    centering_pass = all(
        per_seed[str(seed)]["variants"][variant]["policy_centering_residual"]
        <= config.model.policy_center_tolerance
        for seed in config.training.seeds
        for variant in ("docm_dueling_credit", "docm_dueling_no_tree")
    )
    tree_residuals: dict[str, float] = {}
    for variant in ("docm_dueling_credit", "docm_dueling_no_tree"):
        scores = seed0_scores[variant]
        terminal_mask = (locked_dataset.q_count > 0) & locked_dataset.forced_done
        if terminal_mask.any():
            tree_residuals[variant] = float(
                np.mean(
                    np.square(scores[terminal_mask] - locked_dataset.forced_success[terminal_mask])
                )
            )
        else:
            tree_residuals[variant] = 0.0
    tree_pass = tree_residuals["docm_dueling_credit"] <= tree_residuals["docm_dueling_no_tree"]

    ensemble = CriticEnsemble(
        state_dim=embedder.dimension,
        action_dim=embedder.dimension,
        max_horizon=horizon,
        architecture=architecture,
        loss_weights=candidate_variant_spec("docm_dueling_credit")[0],
        action_mlp_hidden=config.model.action_mlp_hidden,
        target_network_ema=config.model.target_network_ema,
        members=config.uncertainty.members,
        base_seed=config.training.seeds[0],
    )
    ensemble.fit(train_dataset, training=training_settings)
    calibration_mean, _ = ensemble.predict_q(calibration_dataset)
    calibration_picked = calibration_mean[calibration_arrays["branch"], calibration_arrays["slot"]]
    conformal = fit_conformal(
        calibration_picked,
        calibration_arrays["q_ref"],
        target_coverage=config.uncertainty.target_coverage,
    )
    locked_mean, locked_disagreement = ensemble.predict_q(locked_dataset)
    locked_picked = locked_mean[locked_arrays["branch"], locked_arrays["slot"]]
    coverage = coverage_report(conformal, locked_picked, locked_arrays["q_ref"])
    coverage_pass = bool(
        coverage["empirical_coverage"]
        >= config.uncertainty.target_coverage - decision.uncertainty_coverage_tolerance
    )

    c3_predictions = c3_estimates(c3_rows)
    c3_q_by_key = {
        (estimate.episode_id, estimate.step_index, action): q
        for estimate in c3_predictions
        for action, q in zip(estimate.actions, estimate.q_hats, strict=True)
    }
    c3_squared = []
    for row in locked_labeled:
        key = (row.episode_id, row.step_index, row.candidate_action)
        if key in c3_q_by_key:
            c3_squared.append((c3_q_by_key[key] - row.q_hat) ** 2)
    c3_q_mse = float(np.mean(c3_squared)) if c3_squared else None

    gigpo_credits = gigpo_step_credits(reference_episodes, GigpoSettings())
    gigpo_by_key = {(credit.episode_id, credit.step_index): credit for credit in gigpo_credits}
    gigpo_pairs_model: list[float] = []
    gigpo_pairs_ref: list[float] = []
    for row in locked_labeled:
        if not row.logged_action:
            continue
        credit = gigpo_by_key.get((row.episode_id, row.step_index))
        if credit is not None and credit.anchor_group_size >= 2:
            gigpo_pairs_model.append(credit.combined_advantage)
            gigpo_pairs_ref.append(row.q_hat - row.v_hat)
    gigpo_rank_correlation = (
        _spearman(
            np.asarray(gigpo_pairs_model, dtype=np.float64),
            np.asarray(gigpo_pairs_ref, dtype=np.float64),
        )
        if len(gigpo_pairs_model) >= 2
        else None
    )

    pass_gate = {
        "stage1_pass": signal_gate["stage1_pass"],
        "capacity_matched": capacity_matched,
        "seed0_positive_ci": primary_positive,
        "regret_pass": regret_pass,
        "terminal_brier_pass": brier_pass,
        "positive_gap_seeds": positive,
        "adverse_ci_seeds": adverse,
        "robust": robust,
        "policy_centering_pass": centering_pass,
        "tree_residual_pass": tree_pass,
        "uncertainty_coverage_pass": coverage_pass,
    }
    e11_pass = bool(
        pass_gate["stage1_pass"]
        and capacity_matched
        and all(primary_positive.values())
        and regret_pass
        and brier_pass
        and all(robust.values())
        and centering_pass
        and tree_pass
        and coverage_pass
    )

    metrics: dict[str, Any] = {
        "experiment": "e11_alfworld_credit",
        "status": "pass" if e11_pass else "fail",
        "environment_backend": config.environment.backend,
        "actor_policy_id": actor.policy_id,
        "embedding_id": embedder.embedding_id,
        "data": {
            "train_episodes": len(train_episodes),
            "calibration_episodes": len(calibration_episodes),
            "reference_episodes": len(reference_episodes),
            "train_branch_rows": len(train_rows),
            "calibration_branch_rows": len(calibration_rows),
            "locked_branch_rows": len(locked_rows),
            "locked_states": len(locked_states),
        },
        "signal_gate": signal_gate,
        "model": {
            "parameter_counts": parameter_counts,
            "capacity_matched": capacity_matched,
        },
        "per_seed": per_seed,
        "uncertainty": {
            "conformal": conformal.as_dict(),
            "coverage": coverage,
            "mean_ensemble_disagreement": float(
                np.mean(locked_disagreement[locked_arrays["branch"], locked_arrays["slot"]])
            ),
        },
        "baselines": {
            "c3_direct_q_mse": c3_q_mse,
            "c3_states": len(c3_predictions),
            "gigpo_anchor_diagnostics": anchor_diagnostics(gigpo_credits),
            "gigpo_reference_rank_correlation": gigpo_rank_correlation,
            "tree_terminal_residuals": tree_residuals,
        },
        "e11_decision": pass_gate,
        "e11_pass": e11_pass,
        "claim_scope": (
            "amortized recovery of continuation-policy-specific action credit in a "
            "resettable, verifiable multi-step environment at fixed model capacity; "
            "not superiority over direct-credit methods, not causal claims about "
            "observational logs"
        ),
    }

    artifacts: dict[str, Path] = {}
    episodes_path = resolved_output / "episodes.parquet"
    _write_parquet(
        episodes_path,
        _episode_records(train_episodes, "train")
        + _episode_records(calibration_episodes, "valid_seen")
        + _episode_records(reference_episodes, "valid_unseen"),
    )
    artifacts["episodes"] = episodes_path
    steps_path = resolved_output / "steps.parquet"
    _write_parquet(
        steps_path,
        _step_records(train_episodes)
        + _step_records(calibration_episodes)
        + _step_records(reference_episodes),
    )
    artifacts["steps"] = steps_path
    branch_path = resolved_output / "branch_targets.parquet"
    _write_parquet(
        branch_path,
        [asdict(row) for row in train_rows + calibration_rows + locked_rows],
    )
    artifacts["branch_targets"] = branch_path
    distributions_path = resolved_output / "state_policy_distributions.parquet"
    _write_parquet(
        distributions_path,
        [
            {
                "state_hash": d.state_hash,
                "full_policy_distribution_hash": d.full_policy_distribution_hash,
                "actions_json": json.dumps(list(d.actions)),
                "probabilities_json": json.dumps(list(d.probabilities)),
            }
            for d in train_distributions + calibration_distributions + locked_distributions
        ],
    )
    artifacts["state_policy_distributions"] = distributions_path
    predictions_path = resolved_output / "predictions.parquet"
    prediction_records = []
    for index in range(locked_arrays["q_ref"].size):
        branch = int(locked_arrays["branch"][index])
        slot = int(locked_arrays["slot"][index])
        record = {
            "row": index,
            "game_id": str(locked_arrays["games"][index]),
            "q_ref": float(locked_arrays["q_ref"][index]),
            "q_se_ref": float(locked_arrays["q_se"][index]),
            "ensemble_mean": float(locked_mean[branch, slot]),
            "ensemble_disagreement": float(locked_disagreement[branch, slot]),
        }
        for variant in CANDIDATE_VARIANTS:
            record[f"score_{variant}"] = float(seed0_scores[variant][branch, slot])
        prediction_records.append(record)
    _write_parquet(predictions_path, prediction_records)
    artifacts["predictions"] = predictions_path

    c3_path = resolved_output / "c3_direct_predictions.parquet"
    _write_parquet(
        c3_path,
        [
            {
                "episode_id": estimate.episode_id,
                "step_index": estimate.step_index,
                "state_hash": estimate.state_hash,
                "actions_json": json.dumps(list(estimate.actions)),
                "q_hats_json": json.dumps(list(estimate.q_hats)),
                "advantages_json": json.dumps(list(estimate.advantages)),
                "rollouts_used": estimate.rollouts_used,
            }
            for estimate in c3_predictions
        ],
    )
    artifacts["c3_direct_predictions"] = c3_path
    gigpo_path = resolved_output / "gigpo_credit_diagnostics.parquet"
    _write_parquet(
        gigpo_path,
        [
            {
                "episode_id": credit.episode_id,
                "step_index": credit.step_index,
                "state_hash": credit.state_hash,
                "macro_advantage": credit.macro_advantage,
                "micro_advantage": credit.micro_advantage,
                "combined_advantage": credit.combined_advantage,
                "anchor_group_size": credit.anchor_group_size,
            }
            for credit in gigpo_credits
        ],
    )
    artifacts["gigpo_credit_diagnostics"] = gigpo_path

    replay_path = resolved_output / "replay_audit.json"
    replay_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    artifacts["replay_audit"] = replay_path
    budget_path = resolved_output / "budget_ledger.json"
    budget_path.write_text(json.dumps(ledger.as_dict(), indent=2, sort_keys=True) + "\n")
    artifacts["budget_ledger"] = budget_path
    config_path = resolved_output / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True))
    artifacts["resolved_config"] = config_path

    _write_plots(resolved_output, artifacts, seed0_scores, locked_arrays, locked_mean, conformal)

    metrics["runtime_seconds"] = time.perf_counter() - started
    scientific = {key: value for key, value in metrics.items() if key != "runtime_seconds"}
    metrics["scientific_metrics_sha256"] = sha256_json(scientific)
    metrics_path = resolved_output / "metrics.json"
    write_metrics_json(metrics, metrics_path)
    artifacts["metrics"] = metrics_path
    manifest_path = resolved_output / "run_manifest.json"
    manifest = build_run_manifest(
        repository=repository,
        resolved_config=config.model_dump(mode="json"),
        artifacts={name: path.name for name, path in artifacts.items()},
    )
    manifest["environment"] = {
        "backend": config.environment.backend,
        "actor_policy_id": actor.policy_id,
        "embedding_id": embedder.embedding_id,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    artifacts["manifest"] = manifest_path
    client.close()
    return ExperimentResult(metrics=metrics, output_dir=resolved_output, artifacts=artifacts)


def _write_plots(
    output: Path,
    artifacts: dict[str, Path],
    seed0_scores: dict[str, FloatArray],
    locked_arrays: dict[str, npt.NDArray[Any]],
    locked_mean: FloatArray,
    conformal: Any,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_ref = locked_arrays["q_ref"]
    predicted = seed0_scores["docm_dueling_credit"][locked_arrays["branch"], locked_arrays["slot"]]
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(q_ref, predicted, s=12, alpha=0.7)
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axis.set_xlabel("reference q_hat (high K)")
    axis.set_ylabel("model Q (seed 0)")
    axis.set_title("E11 credit calibration")
    calibration_path = output / "credit_calibration.png"
    figure.savefig(calibration_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    artifacts["credit_calibration"] = calibration_path

    picked_mean = locked_mean[locked_arrays["branch"], locked_arrays["slot"]]
    lower, upper = conformal.interval(picked_mean)
    order = np.argsort(picked_mean)
    figure, axis = plt.subplots(figsize=(6, 4))
    positions = np.arange(order.size)
    axis.fill_between(positions, lower[order], upper[order], alpha=0.3, label="conformal")
    axis.plot(positions, picked_mean[order], linewidth=1, label="ensemble mean")
    axis.scatter(positions, q_ref[order], s=8, label="reference")
    axis.set_xlabel("locked rows (sorted by prediction)")
    axis.set_ylabel("success probability")
    axis.legend()
    axis.set_title("E11 uncertainty calibration")
    uncertainty_path = output / "uncertainty_calibration.png"
    figure.savefig(uncertainty_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    artifacts["uncertainty_calibration"] = uncertainty_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/e11_alfworld_credit_smoke.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    result = run_e11(load_config(arguments.config), output_dir=arguments.output_dir)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
