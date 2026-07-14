"""E12: online post-training with credit-based advantages, budget-matched.

Control question: does LongFeedback credit improve task success under a
matched rollout-token budget? The CPU smoke profile runs every method on the
fake world with the trainable softmax actor standing in for the LoRA LLM; the
optimization shell, budget accounting, label freshness, and evaluation
schedule are the real ones.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from longfeedback.actors.trainable import TrainableSoftmaxCandidatePolicy
from longfeedback.credit.branching import BranchSelectionRule, CandidateRule
from longfeedback.credit.gigpo import GigpoSettings
from longfeedback.evaluation import write_metrics_json
from longfeedback.experiments.e9 import _repository_root
from longfeedback.experiments.e11_alfworld_credit import (
    EnvironmentSettings,
    build_environment,
)
from longfeedback.experiments.manifest import build_run_manifest, sha256_json
from longfeedback.experiments.types import ExperimentResult
from longfeedback.models.candidate_docm import CandidateTrainingSettings
from longfeedback.models.encoders import EncoderArchitecture
from longfeedback.models.text_embeddings import HashedTextEmbedder
from longfeedback.training.group_policy import GroupPolicySettings
from longfeedback.training.online import (
    E12_METHODS,
    OnlineLoopSettings,
    run_online_method,
)


class OnlineSettingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    iterations: int = Field(ge=1)
    base_episodes_per_iteration: int = Field(ge=1)
    iteration_token_budget: int = Field(ge=1)
    max_extra_episodes: int = Field(ge=0)
    branch_states_per_episode: int = Field(ge=1)
    uniform_selection_weight: float = Field(ge=0.0, le=1.0)
    full_enumeration_limit: int = Field(ge=1)
    top_actor_candidates: int = Field(ge=0)
    random_candidates: int = Field(ge=0)
    branch_rollouts_per_action: int = Field(ge=1)
    unforced_rollouts: int = Field(ge=1)
    max_policy_lag: int = Field(ge=0, le=0)
    evaluation_interval: int = Field(ge=1)


class ActorTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ratio_clip: float = Field(gt=0.0, lt=1.0)
    update_epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    kl_coefficient: float = Field(ge=0.0)
    entropy_coefficient: float = Field(ge=0.0)
    max_grad_norm: float = Field(gt=0.0)


class CriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding_dim: int = Field(ge=8)
    d_model: int = Field(ge=8)
    n_layers: int = Field(ge=1)
    n_heads: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)
    action_mlp_hidden: int = Field(ge=4)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    grad_clip: float = Field(gt=0.0)
    policy_center_tolerance: float = Field(gt=0.0)


class E12DecisionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_success_lift: float = Field(ge=0.0)
    minimum_positive_seeds: int = Field(ge=1)


class E12Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["e12_alfworld_online"]
    seed: int
    seeds: tuple[int, ...]
    output_dir: Path
    environment: EnvironmentSettings
    methods: tuple[str, ...]
    online: OnlineSettingsConfig
    actor_training: ActorTrainingConfig
    critic: CriticConfig
    decision: E12DecisionSettings

    def model_post_init(self, context: Any) -> None:
        del context
        for method in self.methods:
            if method not in E12_METHODS:
                raise ValueError(f"unknown E12 method {method!r}")


def load_config(path: Path) -> E12Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("E12 config must be a YAML mapping")
    return E12Config.model_validate(raw)


def _loop_settings(config: E12Config) -> OnlineLoopSettings:
    return OnlineLoopSettings(
        iterations=config.online.iterations,
        base_episodes_per_iteration=config.online.base_episodes_per_iteration,
        iteration_token_budget=config.online.iteration_token_budget,
        max_extra_episodes=config.online.max_extra_episodes,
        max_steps=config.environment.max_steps,
        branch_selection=BranchSelectionRule(
            states_per_episode=config.online.branch_states_per_episode,
            uniform_weight=config.online.uniform_selection_weight,
        ),
        candidate_rule=CandidateRule(
            full_enumeration_limit=config.online.full_enumeration_limit,
            top_actor_candidates=config.online.top_actor_candidates,
            random_candidates=config.online.random_candidates,
        ),
        branch_rollouts_per_action=config.online.branch_rollouts_per_action,
        unforced_rollouts=config.online.unforced_rollouts,
        max_policy_lag=config.online.max_policy_lag,
        evaluation_interval=config.online.evaluation_interval,
        gigpo=GigpoSettings(),
        group_policy=GroupPolicySettings(
            ratio_clip=config.actor_training.ratio_clip,
            kl_coefficient=config.actor_training.kl_coefficient,
            entropy_coefficient=config.actor_training.entropy_coefficient,
            update_epochs=config.actor_training.update_epochs,
            learning_rate=config.actor_training.learning_rate,
            max_grad_norm=config.actor_training.max_grad_norm,
        ),
        critic_training=CandidateTrainingSettings(
            epochs=config.critic.epochs,
            batch_size=config.critic.batch_size,
            learning_rate=config.critic.learning_rate,
            weight_decay=config.critic.weight_decay,
            grad_clip=config.critic.grad_clip,
        ),
        critic_architecture=EncoderArchitecture(
            d_model=config.critic.d_model,
            n_layers=config.critic.n_layers,
            n_heads=config.critic.n_heads,
            dropout=config.critic.dropout,
        ),
        action_mlp_hidden=config.critic.action_mlp_hidden,
        policy_center_tolerance=config.critic.policy_center_tolerance,
    )


def run_e12(config: E12Config, *, output_dir: Path | None = None) -> ExperimentResult:
    started = time.perf_counter()
    repository = _repository_root()
    resolved_output = (output_dir or config.output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    client = build_environment(config.environment)
    embedder = HashedTextEmbedder(dim=config.critic.embedding_dim)
    settings = _loop_settings(config)
    train_games = client.list_games("train")
    valid_seen_games = client.list_games("valid_seen")
    valid_unseen_games = client.list_games("valid_unseen")

    per_seed: dict[str, Any] = {}
    locked_by_method: dict[str, list[float]] = {method: [] for method in config.methods}
    budget_violations: dict[str, int] = {method: 0 for method in config.methods}
    for seed in config.seeds:
        # All methods start from the identical immutable initial checkpoint.
        initial_policy = TrainableSoftmaxCandidatePolicy(embedder, seed=config.seed + 100 * seed)
        initial_state = initial_policy.state_dict()
        seed_results: dict[str, Any] = {}
        for method in config.methods:
            outcome = run_online_method(
                method,
                client,
                embedder,
                settings,
                seed=seed,
                initial_policy_state=initial_state,
                train_games=train_games,
                valid_seen_games=valid_seen_games,
                valid_unseen_games=valid_unseen_games,
            )
            locked_by_method[method].append(outcome.locked_success)
            budget_violations[method] += int(outcome.budget_exceeded)
            seed_results[method] = {
                "locked_valid_unseen_success": outcome.locked_success,
                "success_curve": outcome.success_curve,
                "policy_ids": {
                    "initial": outcome.policy_ids[0],
                    "final": outcome.policy_ids[-1],
                    "distinct": len(set(outcome.policy_ids)),
                },
                "diagnostics": {
                    key: value
                    for key, value in outcome.diagnostics.items()
                    if key != "update_stats"
                },
                "update_stats": outcome.diagnostics.get("update_stats", []),
                # Wall time is honest accounting but nondeterministic; it
                # stays out of the hashed scientific metrics.
                "ledger": {
                    key: value
                    for key, value in outcome.ledger.items()
                    if key != "wall_time_seconds"
                },
                "budget_exceeded": outcome.budget_exceeded,
            }
        per_seed[str(seed)] = seed_results

    decision: dict[str, Any] = {}
    comparisons_pass: dict[str, bool] = {}
    if "longfeedback_group" in config.methods:
        lf = np.asarray(locked_by_method["longfeedback_group"], dtype=np.float64)
        for baseline in ("terminal_grpo", "prefix_group"):
            if baseline not in config.methods:
                continue
            base = np.asarray(locked_by_method[baseline], dtype=np.float64)
            lifts = lf - base
            positive_seeds = int(np.sum(lifts > 0.0))
            mean_lift = float(np.mean(lifts))
            comparisons_pass[baseline] = bool(
                mean_lift >= config.decision.minimum_success_lift
                and positive_seeds >= config.decision.minimum_positive_seeds
            )
            decision[f"longfeedback_vs_{baseline}"] = {
                "mean_lift": mean_lift,
                "per_seed_lift": lifts.tolist(),
                "positive_seeds": positive_seeds,
                "pass": comparisons_pass[baseline],
            }
    no_budget_violation = all(count == 0 for count in budget_violations.values())
    decision["budget_violations"] = budget_violations
    decision["no_budget_violation"] = no_budget_violation
    core_gate_pass = bool(
        no_budget_violation and comparisons_pass and all(comparisons_pass.values())
    )
    decision["core_mechanism_gate_pass"] = core_gate_pass
    paper_methods_present = {"c3_group", "gigpo"} <= set(config.methods)
    decision["paper_ready_baselines_present"] = paper_methods_present

    metrics: dict[str, Any] = {
        "experiment": "e12_alfworld_online",
        "status": "pass" if core_gate_pass else "fail",
        "environment_backend": config.environment.backend,
        "methods": list(config.methods),
        "locked_success_by_method": {method: values for method, values in locked_by_method.items()},
        "per_seed": per_seed,
        "e12_decision": decision,
        "claim_scope": (
            "core mechanism gate against terminal and prefix credit under a matched "
            "rollout-token budget; paper-ready comparative claims additionally "
            "require audited C3 and GiGPO runs under a frozen budget contract"
        ),
    }

    curves_path = resolved_output / "learning_curves.json"
    curves_path.write_text(
        json.dumps(
            {
                seed: {method: result["success_curve"] for method, result in seed_results.items()}
                for seed, seed_results in per_seed.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    config_path = resolved_output / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True))

    metrics["runtime_seconds"] = time.perf_counter() - started
    scientific = {key: value for key, value in metrics.items() if key != "runtime_seconds"}
    metrics["scientific_metrics_sha256"] = sha256_json(scientific)
    metrics_path = resolved_output / "metrics.json"
    write_metrics_json(metrics, metrics_path)
    manifest_path = resolved_output / "run_manifest.json"
    manifest = build_run_manifest(
        repository=repository,
        resolved_config=config.model_dump(mode="json"),
        artifacts={
            "metrics": metrics_path.name,
            "learning_curves": curves_path.name,
            "resolved_config": config_path.name,
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    client.close()
    return ExperimentResult(
        metrics=metrics,
        output_dir=resolved_output,
        artifacts={
            "metrics": metrics_path,
            "learning_curves": curves_path,
            "resolved_config": config_path,
            "manifest": manifest_path,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/e12_alfworld_online_smoke.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    result = run_e12(load_config(arguments.config), output_dir=arguments.output_dir)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
