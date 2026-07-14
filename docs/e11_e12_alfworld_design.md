# E11/E12 design: replay-verified credit and online LLM post-training in ALFWorld

Status: implementation specification draft v0.2  
Date: 2026-07-13  
Owners: LongFeedback project  
Target environment: text-only [ALFWorld](https://arxiv.org/abs/2010.03768)
(`AlfredTWEnv`)

This document is deliberately more specific than a research proposal. It defines the
scientific quantities, software boundaries, data contracts, experiment variants, metrics,
acceptance gates, artifacts, tests, and implementation order for E11 and E12.

The thresholds below are **proposed, not frozen**. They become a confirmatory contract only
after the team reviews this document, records the dependency/model revisions and compute
budget, and commits the final configuration before inspecting full-run results.

Revision v0.2 changes the implementation baseline before model coding:

- preserves the terminal, prefix-value, and action-value heads;
- replaces independent Q outputs with bounded policy-centered dueling Q;
- trains V from direct unforced continuation targets;
- adds architectural Q-V centering and parent-Q/child-V tree consistency;
- scores full semantic candidate-action embeddings;
- adds bootstrap/conformal calibrated uncertainty;
- makes direct C3 and GiGPO required paper-facing baselines; and
- confines semantic hindsight feedback to an optional auxiliary ablation.

## 1. Objective

E11 and E12 add an online, resettable, verifiable environment to LongFeedback.

- **E11 asks an estimation question:** can LongFeedback recover replay-based action credit
  at held-out ALFWorld states while preserving terminal-outcome quality?
- **E12 asks a control question:** does using that credit to post-train an LLM actor improve
  task success and sample efficiency under a matched rollout budget?

The experiments must keep these questions separate. E11 may pass even if E12 later fails;
that would mean the project can estimate credit but has not yet converted it into better
policy learning. E12 may not claim success unless E11 has passed its replay-integrity,
signal, and credit-recovery gates.

## 2. Relationship to the existing project

The existing `DelayedOutcomeCreditModel` (DOCM) assumes:

- a fixed horizon;
- numeric observation features;
- one fixed global action vocabulary represented by integer IDs;
- one scalar response per step; and
- credit for the logged action relative to a fixed reference action.

ALFWorld instead has variable-length episodes and a state-dependent set of text commands.
Commands such as `take apple 1 from countertop 2` cannot be reduced to a small global
action ID without discarding the object argument that determines success.

E11 therefore introduces a **candidate-action DOCM** alongside, not in place of, the
existing DOCM. Earlier experiments and their reproducibility hashes must remain unchanged.
The new model reuses the causal sequence-encoder pattern and the three-head scientific
decomposition:

1. terminal-outcome head;
2. prefix-value head; and
3. candidate action-value head.

## 3. Scientific quantities

Let `h_t` be the complete observable ALFWorld history before action `t`, let `a` be an
admissible command at that state, and let `pi_k` be the frozen continuation policy recorded
by checkpoint/version `k`.

### 3.1 Terminal outcome

`Y in {0, 1}` is ALFWorld task success at termination. Timeouts, invalid environment
failures, and infrastructure errors are not silently converted into task failures; terminal
reason is stored separately.

### 3.2 Prefix value

```text
V^{pi_k}(h_t) = E[Y | start at h_t and follow pi_k]
```

This is policy-dependent predictive value. In a resettable environment it can be estimated
by repeated continuations from the restored prefix.

### 3.3 Forced-action value

```text
Q^{pi_k}(h_t, a) = E[Y | force action a at h_t, then follow pi_k]
```

This is the E11 action-value target. ALFWorld dynamics are deterministic in text mode once
the game and command prefix are fixed; the remaining randomness comes from continuation
policy sampling.

### 3.4 Action credit

The primary credit definition is the actor-policy advantage:

```text
A^{pi_k}(h_t, a) = Q^{pi_k}(h_t, a) - V^{pi_k}(h_t)
```

There is no arbitrary global `reference_action=0` in ALFWorld. The model uses the frozen
actor policy as its center:

```text
V^{pi_k}(h_t) = sum_{a in A(h_t)} pi_k(a|h_t) Q^{pi_k}(h_t,a)
```

The equality is imposed by the action-value parameterization in Section 9, not merely
encouraged by a loss. It is computed over the **full admissible action set**, even when only
a subset has Monte Carlo branch labels. `V` is also estimated independently using unforced
continuations. Those direct targets both train the value head and test whether the
policy-centered identity is empirically well calibrated.

For transfer to settings with a meaningful operational reference action, such as sending
versus not sending a notification, also report the contrast
`Q(h_t,a) - Q(h_t,a_ref)`. This is a derived estimand; it does not replace the primary
actor-policy advantage in ALFWorld.

### 3.5 Monte Carlo targets

For candidate action `a`, run `K` continuation rollouts with frozen `pi_k`:

```text
q_hat(h_t,a) = successes(h_t,a) / K
se_q          = sqrt(q_hat * (1 - q_hat) / K)
```

Store the integer success count and `K`, not just `q_hat`. The action head is trained with
the binomial likelihood implemented as weighted binary cross-entropy on the success rate.
The reference evaluation set uses a much larger `K` than the training set.

Unforced continuations provide an analogous direct target:

```text
v_hat(h_t) = unforced_successes(h_t) / K_v
```

The primary value head is trained on this target. Broadcasting the one observed terminal
outcome to every prefix remains only a predictive baseline and fallback ablation; it is not
the causal-continuation target for the revised model.

The target is always annotated with:

- continuation policy ID and checkpoint hash;
- environment revision and game ID;
- exact replay-prefix hash;
- branch action;
- rollout seed block;
- rollout count, success count, estimate, and standard error; and
- candidate-selection rule and probability, when applicable.

### 3.6 Tree consistency target

For a replayed edge produced by forcing `a` at `h_t`, let `h_{t+1}` be the resulting history.
ALFWorld has zero intermediate task reward for this binary-success estimand, so:

```text
Q^{pi_k}(h_t,a) = E[V^{pi_k}(h_{t+1}) | h_t,a]
```

For a terminal edge, the right side is the observed terminal success. For a nonterminal
edge, it is a direct high-quality continuation estimate at the child when available, or a
stop-gradient target-network value otherwise. This constraint is called **Q-V/tree
consistency** below. It must never use the locked-reference split during training.

## 4. Scope and non-goals

### In scope

- Text-only ALFWorld with admissible commands.
- A frozen LLM candidate policy for E11.
- Exact prefix restoration by reset plus deterministic action replay.
- Low-rollout training targets and high-rollout held-out reference targets.
- Capacity-matched LongFeedback variants.
- LoRA/parameter-efficient online post-training in E12.
- Terminal GRPO, prefix-credit group optimization, direct C3, GiGPO, and LongFeedback group
  optimization comparisons.
- Compute and rollout accounting.

### Not in the first implementation

- Visual/THOR ALFWorld.
- Free-form commands outside the admissible command list.
- Hidden chain-of-thought collection or optimization.
- Multi-agent collaboration.
- Asynchronous distributed actor/learner infrastructure.
- Reusing credit labels across materially changed continuation policies.
- Claims that ALFWorld results identify causal effects in LinkedIn logs.

The first actor chooses among admissible commands. This removes parsing failures and makes
the policy a well-defined finite categorical distribution at each state. Free-form action
generation is a later stress test, not part of the E11/E12 core claim.

## 5. Environment and dependency boundary

The official ALFWorld package separates text-only and visual installations. Text-only
execution does not need a GPU, but Apple Silicon setup may require an x86 Python environment.
LongFeedback currently requires Python 3.11+, while the most portable ALFWorld setup may use
a separate Python environment. E11/E12 must therefore depend on an environment protocol,
not direct imports throughout the experiment code.

### 5.1 Environment client protocol

Create `longfeedback.environments.base.EnvironmentClient` with these operations:

```python
class EnvironmentClient(Protocol):
    def list_games(self, split: str) -> Sequence[GameRef]: ...
    def reset(self, game: GameRef, *, seed: int) -> EnvObservation: ...
    def step(self, action: str) -> EnvTransition: ...
    def close(self) -> None: ...
```

`EnvObservation` contains:

```python
game_id: str
goal: str
observation: str
admissible_actions: tuple[str, ...]
step_index: int
done: bool
score: float
state_hash: str
```

`state_hash` is a canonical SHA-256 hash of game ID, step index, normalized observation,
sorted admissible actions, score, and done status.

### 5.2 Backends

Implement two interchangeable backends:

1. `InProcessAlfworldClient` for Linux/compatible Python environments.
2. `SubprocessAlfworldClient` using line-delimited JSON messages to a dedicated ALFWorld
   worker environment.

The subprocess backend is the portability default. Each parallel rollout worker owns one
environment process; shared mutable ALFWorld instances are forbidden.

### 5.3 Exact replay

ALFWorld does not need an opaque binary checkpoint for E11. A replay handle is:

```python
game_id: str
environment_seed: int
action_prefix: tuple[str, ...]
expected_state_hash: str
```

Restoration resets the same game and replays `action_prefix`. A branch is valid only when
the restored state hash exactly matches `expected_state_hash`. Any mismatch aborts the
branch and increments a replay-integrity failure counter; it is never treated as a failed
task rollout.

Before E11 target generation, run a replay audit over at least 1,000 sampled prefixes or all
available pilot prefixes, whichever is smaller.

### 5.4 Version pinning

The run manifest must record:

- ALFWorld package version and git revision;
- downloaded game-data manifest hashes;
- TextWorld version;
- Python version for both the main and environment processes;
- actor model ID, immutable revision, tokenizer revision, and license;
- text-embedding model ID and immutable revision;
- prompt template hash; and
- all LoRA and optimizer settings.

Do not use floating model revisions such as `main` in a confirmatory run.

## 6. Actor policy

### 6.1 Candidate-scoring policy

The LLM actor scores every admissible command rather than generating unrestricted text.
For prompt `x_t` and candidate command tokens `a`, define:

```text
s_theta(a | x_t) = mean token log-probability of candidate a conditioned on x_t
pi_theta(a | x_t) = softmax(s_theta(a | x_t) / temperature)
```

Length normalization is part of the policy definition and must not change between methods.
Commands are normalized and sorted deterministically before scoring. Candidate batching may
use prefix KV caching, but the logical scores must match the uncached implementation within
floating tolerance.

### 6.2 Prompt contract

The prompt contains only information available before the decision:

- task goal;
- observations and selected actions through `t-1`;
- current observation;
- current sorted admissible commands; and
- fixed action-selection instructions.

It contains no future observation, expert plan, optimal action, terminal success, branch
outcome, or reference target. The prompt renderer returns both text and a canonical hash.

### 6.3 E11 actor

E11 uses one frozen actor checkpoint for collection and all continuation rollouts. The actor
version is part of the estimand. A smoke run may use a mock or API actor, but the full E11
run should use the same locally post-trainable base actor intended for E12.

If the frozen actor's success rate is outside `[0.10, 0.90]` on the pilot development set,
the target distribution is likely too degenerate. A short behavior-cloning/SFT warm start on
ALFWorld training demonstrations is then allowed, but it must be completed and frozen before
E11 reference targets are generated. The identical warm-start checkpoint is used by every
E12 method.

### 6.4 Actor interface

```python
class CandidatePolicy(Protocol):
    @property
    def policy_id(self) -> str: ...
    def score(self, prompt: str, candidates: Sequence[str]) -> PolicyScores: ...
    def sample(
        self,
        scores: PolicyScores,
        *,
        random_value: float,
    ) -> PolicyDecision: ...
```

`PolicyScores` stores raw scores, normalized probabilities, token counts, and model-forward
token accounting. `PolicyDecision` stores chosen action, probability, log-probability,
entropy, and RNG provenance.

## 7. E11 data generation

### 7.1 Dataset splits

Use ALFWorld's official game splits without moving games between them:

- `train`: trajectory collection and low-K branch targets;
- `valid_seen`: development, threshold-free model selection, and pilot signal audit;
- `valid_unseen`: locked final evaluation only.

All trajectories, branches, and continuations from one game remain in one split. Duplicate
game IDs or replay-prefix hashes across training and locked evaluation are hard failures.

### 7.2 Base trajectory collection

For each selected game:

1. reset the game with a derived environment seed;
2. render the leakage-safe prompt;
3. score and sample one admissible action with the frozen actor;
4. record the decision and transition;
5. continue until success, environment termination, or `max_steps`; and
6. store terminal reason and success separately.

RNG streams are derived independently for environment selection, actor sampling, branch
state selection, branch candidate selection, and continuation sampling. Adding one branch
must not change the base trajectory.

### 7.3 Branch-state selection

Branch states must be chosen without inspecting branch outcomes. Use a fixed stratified
rule:

- divide each nontrivial trajectory into early, middle, and late normalized-time strata;
- choose at most one state per stratum;
- within a stratum, sample proportional to a frozen mixture of uniform weight and normalized
  actor entropy; and
- exclude terminal states and states with fewer than two admissible actions.

Record selection probabilities. The primary evaluation reports unweighted performance on
the fixed selected-state distribution; an inverse-selection-weighted sensitivity analysis
reports trajectory-wide estimates.

The locked reference set is selected once by this outcome-blind rule. After a fixed pilot
warm-up, additional **training-only** rollout allocation may prioritize states with high
ensemble disagreement, actor entropy, and remaining horizon. This acquisition rule and its
probability are logged, and it may not change which states enter locked evaluation.

### 7.4 Candidate-action selection

Branching every admissible action can be expensive. For a state with at most
`full_enumeration_limit` commands, evaluate them all. Otherwise construct a deterministic
candidate set containing:

- the logged action;
- the actor's top `top_actor_candidates` commands;
- `random_candidates` commands sampled from the remaining admissible set with a separate
  seed; and
- no expert/optimal action unless it entered through the rules above.

Deduplicate commands while preserving a canonical sorted order. The full admissible set and
the evaluated subset are both stored. Metrics concerning action ranking are explicitly
scoped to the evaluated candidate set.

### 7.5 Continuation rollout

For each `(state, candidate action, rollout seed)`:

1. restore and verify the replay handle;
2. force the candidate action;
3. if the forced action terminates the task, record the result;
4. otherwise follow the frozen continuation policy until termination or `max_steps`;
5. store success, terminal reason, continuation length, actor-forward tokens, and hashes;
6. reuse the same derived policy-randomness seed block across candidate actions when
   possible, providing paired common random numbers without forcing identical downstream
   actions in changed states.

Separately estimate `V^{pi_k}(h_t)` through unforced continuations from the restored state.
This avoids defining the baseline from an incomplete candidate subset.

For every forced first transition, also store the resulting child-state hash, termination
status, and terminal success when applicable. A predeclared subset of nonterminal training
children receives direct unforced continuation targets for auditing and training tree
consistency. The other training children use a lagged target-network value; validation and
locked-reference children are never used to update the model.

### 7.6 Budget tiers

The configuration supports three named profiles. Exact full-run sizes are frozen only after
the pilot reports cost and signal.

| Profile | Base episodes | Branch states/episode | Candidate cap | K train | Reference states | K reference |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 8-32 | 1 | 3 | 2 | 8 | 4 |
| pilot | about 200 | 2 | 4 | 4 | 50 | 32 |
| full proposal | about 1,000 | 3 | 8 | 8 | at least 200 | 128 |

The runner stops before exceeding explicit limits for:

- actor-forward tokens;
- environment steps;
- wall-clock time; and
- API cost, if any external inference is used.

Partial runs are marked incomplete and cannot produce a passing gate.

### 7.7 Preliminary GPU-hour plan

Count one H100 used for one hour as one H100-GPU-hour; using eight GPUs for ten hours is 80
GPU-hours. These are planning ranges, not a frozen budget. The pilot must replace them with
measured actor tokens/second, mean continuation length, batching efficiency, and utilization.

Under the current maximum candidate counts, the pilot implies approximately 38,000
continuations: 6,400 forced training continuations, 12,800 direct-V continuations, 6,400
child-V continuations, 6,400 references, and 6,400 calibration continuations. The proposed
full E11 profile can reach approximately 603,000 continuations: 192,000 forced training,
96,000 direct-V, 96,000 child-V, 204,800 locked reference, and 12,800 calibration
continuations, plus base episodes. Actual counts fall when states have fewer actions or
episodes terminate early.

The estimate below assumes text-only ALFWorld, batched rollout serving, bf16, LoRA for actor
updates, an average continuation of roughly 8-20 decisions, and the current five-seed
confirmatory plan. It is anchored to the published
[GiGPO ALFWorld setup](https://openreview.net/pdf/c28e200ee92ae5eef9869fe35dbd6fc859cd04cf.pdf),
which uses 128 parallel environments for 150 iterations and reports 2 H100s for
Qwen2.5-1.5B and 4 H100s for Qwen2.5-7B. That paper does not report wall-clock hours, so the
ranges remain deliberately broad.

| Stage | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---:|---:|
| Environment engineering, smoke, and E11 pilot | 20-60 H100-h | 50-150 H100-h |
| Full E11 collection and reference construction | 150-400 H100-h | 400-1,200 H100-h |
| Small critic, five-member ensemble, five seeds | 10-30 H100-h | 10-30 H100-h |
| One-seed E12 feasibility run for three methods | 100-250 H100-h | 250-700 H100-h |
| Full E12, five trainable methods by five seeds | 1,000-2,500 H100-h | 3,000-7,500 H100-h |
| **Paper-track total** | **about 1,300-3,200 H100-h** | **about 3,700-9,600 H100-h** |

The recommended funding sequence is therefore:

1. reserve 20-60 H100-hours for the 1.5B pilot;
2. only if replay and signal gates pass, reserve another 150-400 for full E11; and
3. only if E11 passes, fund E12 incrementally, beginning with a one-seed three-method run.

Do not reserve the 7B paper track first. Establish the mechanism with 1.5B, then run a
predeclared 7B confirmation only if model scale is itself part of the intended claim.

## 8. E11 artifact schema

Use Parquet for tabular artifacts and JSON for small manifests/metrics.

### 8.1 `episodes.parquet`

Required columns:

```text
episode_id, game_id, split, task_type, actor_policy_id,
environment_seed, success, terminal_score, terminal_reason,
episode_steps, actor_forward_tokens, prompt_template_hash
```

### 8.2 `steps.parquet`

Required columns:

```text
episode_id, game_id, step_index, goal, observation,
admissible_actions_json, action, action_probability, action_log_probability,
actor_entropy, prompt_hash, state_hash, replay_prefix_hash,
next_observation, done, score_after_step
```

### 8.3 `branch_targets.parquet`

Required columns:

```text
game_id, episode_id, step_index, state_hash, replay_prefix_hash,
candidate_action, candidate_set_id, candidate_selected_probability,
candidate_policy_probability, full_policy_distribution_hash,
continuation_policy_id, rollout_seed_block_hash,
rollout_count, success_count, q_hat, q_se,
unforced_rollout_count, unforced_success_count, v_hat, v_se,
forced_next_state_hash, forced_done, forced_terminal_success,
child_unforced_rollout_count, child_unforced_success_count, child_v_hat, child_v_se,
advantage_hat, target_role
```

`target_role` is one of `train`, `validation`, or `locked_reference`. Locked-reference rows
must never be loaded by a training code path.

The full frozen-actor probability vector is stored once per state in a normalized side
table keyed by `full_policy_distribution_hash`. Policy-centered Q must use the entire
admissible set, not only the labeled candidate subset.

### 8.4 `hindsight_labels.parquet`

Optional semantic hindsight supervision is stored separately from branch targets:

```text
game_id, episode_id, step_index, state_hash, action,
teacher_id, prompt_hash, label_schema_version,
progress_label, error_type, confidence, rationale_hash, target_role
```

Only training trajectories may receive labels used for fitting. The teacher may inspect the
completed trajectory, but its labels feed only the auxiliary head in Section 9. They are
never interpreted as Q/V ground truth, used in the primary E11 pass gate, or generated for
locked-reference examples before model outputs are frozen.

### 8.5 Other artifacts

```text
resolved_config.yaml
metrics.json
run_manifest.json
predictions.parquet
state_policy_distributions.parquet
replay_audit.json
budget_ledger.json
credit_calibration.png
uncertainty_calibration.png
action_regret.png
credit_by_temporal_distance.png
c3_direct_predictions.parquet
gigpo_credit_diagnostics.parquet
```

Raw prompts and full continuation traces are optional large artifacts. If retained, store
them outside Git and reference their hashes in the manifest.

## 9. Candidate-action DOCM

### 9.1 Do not mutate the existing dataset/model contract

Add `CandidateSequenceDataset` and `CandidateDelayedOutcomeCreditModel`. Existing
`SequenceDataset` and `DelayedOutcomeCreditModel` behavior must remain byte-for-byte stable
for existing tests and artifacts.

### 9.2 Semantic state and candidate-action features

The first implementation uses a frozen, revision-pinned semantic text embedding provider:

- state text: task goal plus current observation;
- action text: the complete normalized candidate command, including verb, object, and
  receptacle arguments; and
- optionally, task-type metadata encoded separately.

Embeddings are cached by content hash. The critic's text encoder is separate from the E12
actor so critic inputs do not drift merely because the actor LoRA changes. A bag of global
action IDs is not an acceptable substitute: held-out commands must be scorable from their
meaning. Report a compositional generalization slice containing unseen command strings and,
when feasible, held-out object/receptacle combinations.

This state/action semantic-scoring pattern is related to the
[Deep Reinforcement Relevance Network](https://arxiv.org/abs/1511.04636), which embeds
natural-language states and actions separately before scoring their interaction. Our model
adds a causal history encoder, explicit continuation targets, and the three-head credit
decomposition.

The existing small causal Transformer consumes interleaved state/action embedding tokens.
It must be extended with an optional padding mask for variable-length episodes while
preserving current behavior when no mask is provided.

### 9.3 Three heads with policy-centered dueling Q

The value/advantage factorization is inspired by
[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581).
The bounded probability-space, full-actor-policy centering below is a LongFeedback design
adaptation, not an algorithm claimed by that paper.

Preserve all three scientific heads. At state `t`, the causal history representation is
`h_t`:

1. `outcome_head(h_T)` predicts terminal success;
2. `prefix_value_head(h_t)` predicts `V^{pi_k}(h_t)`; and
3. `action_advantage_head(h_t,e_a)` emits an unconstrained raw advantage `u(h_t,a)` for
   every semantic candidate-action embedding.

Recommended raw action scorer:

```text
z(h_t,a) = [h_t, e_a, h_t * P(e_a), |h_t - P(e_a)|]
u(h_t,a) = MLP(z(h_t,a))
```

Convert raw scores into policy-centered action values over the **full admissible set**:

```text
V_hat(h_t)       = sigmoid(V_logit(h_t))
c(h_t,a)         = u(h_t,a) - sum_b pi_k(b|h_t) * u(h_t,b)
r(h_t,a)         = tanh(c(h_t,a))
d(h_t,a)         = r(h_t,a) - sum_b pi_k(b|h_t) * r(h_t,b)
s_max(h_t)       = largest nonnegative s keeping V_hat + s*d in [eps, 1-eps]
s(h_t)           = sigmoid(scale_head(h_t)) * s_max(h_t)
Q_hat(h_t,a)     = V_hat(h_t) + s(h_t) * d(h_t,a)
A_hat(h_t,a)     = Q_hat(h_t,a) - V_hat(h_t)
```

Compute `s_max` from the tightest upper or lower action bound:

```text
s_upper = min_{a:d(a)>0} (1-eps-V_hat) / d(a)
s_lower = min_{a:d(a)<0} (V_hat-eps) / (-d(a))
s_max   = min(s_upper, s_lower)
```

An empty bound contributes positive infinity; when all `d` are zero, define `s_max=0` and
`Q_hat=V_hat`. The second centering after `tanh` is necessary because a nonlinear transform
does not preserve a zero policy mean. This bounded probability-space construction makes the
unit-tested contract
`abs(V_hat - sum_a pi_k(a|h_t) Q_hat) <= 1e-6`. This makes `Q - V` the actor-policy
advantage by construction. Candidate padding masks and zero-probability actions never enter
the center. Missing full-action probabilities are a hard data error.

### 9.4 Direct value, branch Q, and tree targets

The main model is trained from three kinds of environment evidence:

- base-trajectory terminal success for the terminal head;
- unforced repeated continuations for the value head; and
- forced-action repeated continuations for the action-value head.

Tree consistency connects the latter two. For a forced edge `(h_t,a,h_{t+1})`, define:

```text
tree_target = terminal_success                         if forced_done
              child_v_hat                             if direct child rollouts exist
              stop_gradient(V_target(h_{t+1}))        otherwise
```

`V_target` is an exponential-moving-average copy of the encoder/value head. Tree targets
are refreshed only from training data and are versioned in the manifest. The policy-center
identity is architectural; `L_tree` is the distinct temporal consistency constraint
between a parent Q and its realized child value.

### 9.5 Losses

```text
L = lambda_y       * L_terminal
  + lambda_v       * L_direct_V
  + lambda_q       * L_branch_Q
  + lambda_tree    * L_tree
  + lambda_prefix  * L_prefix_MC
  + lambda_hind    * L_hindsight_aux
```

- `L_terminal`: BCE on base-trajectory success.
- `L_direct_V`: binomial negative log likelihood for unforced continuation success counts.
- `L_branch_Q`: binomial negative log likelihood for forced-action success counts.
- `L_tree`: BCE or squared probability error between `Q_hat(h_t,a)` and `tree_target`, with
  target uncertainty used as a weight when a direct child estimate exists.
- `L_prefix_MC`: the legacy terminal-label-broadcast prefix loss. Its weight is zero in the
  primary revised model and nonzero only for the prefix baseline or an explicit ablation.
- `L_hindsight_aux`: optional semantic-label classification/progress loss described below.
  It may shape the shared representation but must not directly regress Q, V, or advantage.

Normalize binomial losses by total rollout count within each target family, and cap
per-example normalized weights if mixed K would otherwise let a few rows dominate. Locked
high-K reference rows are never used in any loss, early stopping, calibration,
target-network refresh, or hyperparameter selection.

### 9.6 Capacity-matched LongFeedback variants

Every primary LongFeedback variant instantiates the identical encoder and all three heads.
Only loss weights and the scoring head used for evaluation differ.

| Variant | Training supervision | Candidate score used for E11 grading |
|---|---|---|
| `docm_outcome` | terminal BCE only | outcome head after appending the candidate |
| `docm_prefix` | terminal BCE + legacy prefix broadcast | prefix head after appending the candidate |
| `docm_dueling_credit` | terminal + direct V + branch Q + tree | policy-centered candidate Q |
| `docm_dueling_no_tree` | terminal + direct V + branch Q | policy-centered candidate Q; ablation |

The first three are the primary capacity-matched comparison. The no-tree variant isolates
the tree constraint. A legacy independent-Q action head is retained as a separately
parameter-matched architecture ablation to test whether policy centering itself matters.

### 9.7 Direct C3 and GiGPO baselines

Two contemporary direct-credit families are required, not deferred to a future paper:

- **[C3](https://arxiv.org/abs/2603.06859) direct branching:** at each labeled state, use fixed-history forced-action
  continuations and a leave-one-action/policy-centered contrast. It is evaluated at locked
  branch states under the same rollout budget. It is a non-amortized estimator and therefore
  is not credited with predictions at unbranched states. Use the
  [official C3 implementation](https://github.com/EIT-EAST-Lab/C3) as the reference.
- **[GiGPO](https://arxiv.org/abs/2505.10978):** group trajectories at repeated anchor states and compute macro group-relative
  outcome credit plus micro step credit following the pinned public implementation. E11
  reports its credit-ranking diagnostic where the required grouping exists; E12 includes
  the direct policy-optimization baseline under matched total environment interaction. The
  reference code is [langfengQ/verl-agent](https://github.com/langfengq/verl-agent).

Comparisons must distinguish statistical target quality, amortized generalization, and
online rollout cost. Parameter-count matching applies to the three LongFeedback variants;
C3 and GiGPO instead receive a matched environment/token/accelerator ledger.

### 9.8 Semantic hindsight is auxiliary only

An optional teacher may inspect a completed **training** trajectory and label whether an
action made progress, caused a recoverable detour, or introduced a persistent error. The
student predicts these labels from the leakage-safe prefix and current action only.

The semantic role idea is motivated most directly by
[TRIAGE: Role-Typed Credit Assignment for Agentic Reinforcement Learning](https://arxiv.org/abs/2606.32017).
LongFeedback deliberately uses it more conservatively than TRIAGE: semantic labels are an
auxiliary representation target rather than direct process reward or primary credit.

Hindsight labels:

- never replace branch Q or direct V targets;
- never enter the primary model by default (`lambda_hind = 0`);
- never affect locked-reference selection or labels;
- require an explicit `+hindsight_aux` ablation and teacher/prompt hash; and
- support only a representation-learning claim, not causal credit identification.

### 9.9 Calibrated uncertainty

The primary uncertainty implementation follows the practical motivation of
[Deep Ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)
and uses a bootstrap ensemble of independently initialized critics. It separates:

- Monte Carlo target noise, represented by binomial counts/standard errors; and
- model uncertainty, represented by between-member disagreement.

The ensemble mean is calibrated on a dedicated `valid_seen` calibration subset. Following
the split-conformal framework summarized in
[A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511),
prediction intervals are conformalized against high-K development targets to achieve the
predeclared coverage level. Report NLL/Brier calibration for point probabilities and empirical
coverage/width for intervals. Calibration state IDs and fitted parameters are frozen before
`valid_unseen` is opened. The locked split is evaluation-only.

### 9.10 Candidate scoring for predictive heads

To grade outcome-only and prefix models on actions without using the untrained action head,
append a candidate action token to the leakage-safe history and apply the corresponding head
at that candidate boundary. No post-action environment observation is appended.

This mirrors the existing project's attribution baselines while allowing a variable
semantic action space.

## 10. E11 experimental protocol

### 10.1 Stage 0: environment smoke

Required checks:

- install and download text-only ALFWorld;
- enumerate official splits and task types;
- complete at least one game with a scripted or random admissible policy;
- verify no visual/THOR dependency is loaded;
- verify environment subprocess shutdown and timeout behavior; and
- emit an environment/source manifest.

### 10.2 Stage 1: replay and signal gate

Run before fitting LongFeedback.

Proposed gate:

1. `replay_match_rate == 1.0` over the replay audit.
2. No training/locked-reference game or prefix-hash overlap.
3. Frozen actor pilot success rate is within `[0.10, 0.90]`.
4. Median locked-reference `q_se <= 0.05`.
5. At least 200 locked-reference states are evaluated in the full run.
6. At least 30% of locked-reference states are informative, where
   `max_a q_ref(h,a) - min_a q_ref(h,a) >= 0.10`.

If criterion 3 fails before reference generation, perform the predeclared common warm start
and restart E11 with a new actor policy ID. If criteria 4-6 fail after the full reference set
is observed, model grading is not authorized; do not tune the definition of informative
states after seeing results.

### 10.3 Training and calibration

- Use five fixed training seeds: `0,1,2,3,4`.
- Split by game ID, never by step or branch row.
- Fit feature normalization using training games only.
- Fit point and interval calibration on the dedicated `valid_seen` calibration games only.
- Use the same architecture, optimizer steps, batch schedule, and candidate batches for all
  capacity-matched LongFeedback variants.
- Assert exact parameter-count equality for the three primary LongFeedback variants. Report
  compute matching, not parameter matching, for C3 and GiGPO.
- Fit direct-V and branch-Q losses from their integer success/rollout counts; do not replace
  them with denoised labels computed from locked data.
- Select the tree-loss weight, target-network update rate, ensemble size, and conformal
  coverage on `valid_seen`, then freeze them.
- Select hyperparameters on `valid_seen` without inspecting `valid_unseen`.

### 10.4 Primary held-out metrics

All primary comparisons are paired on the same locked state/action rows and clustered by
game ID.

1. **Reference action-value loss**

   ```text
   MSE_Q = mean((q_model(h,a) - q_ref(h,a))^2)
   ```

   Also report precision-weighted MSE as a sensitivity analysis; unweighted MSE is primary
   because all locked rows use the same high rollout count.

2. **Candidate-set action regret**

   ```text
   regret(h) = max_a q_ref(h,a) - q_ref(h, argmax_a score_model(h,a))
   ```

3. **Terminal outcome quality**

   Report Brier score, log loss, AUROC, and calibration. Brier score is the noninferiority
   metric because it is proper and remains defined when one class is rare.

4. **Direct continuation value loss**

   ```text
   MSE_V = mean((v_model(h) - v_ref(h))^2)
   ```

5. **Uncertainty calibration**

   Report predeclared interval coverage and mean width against the high-K locked targets,
   plus Q/V Brier score and NLL. Coverage without interval width is insufficient.

### 10.5 Secondary diagnostics

- mean within-state Spearman correlation across candidate actions;
- top-1 and top-k action agreement with reference;
- signed-advantage accuracy outside an indifference band;
- prefix-value Brier/RMSE against high-K unforced continuation values;
- probability-space policy-centering residual and Q-V/tree residual;
- ensemble error-versus-disagreement correlation and risk-coverage curves;
- performance on unseen command strings and held-out semantic compositions;
- credit metrics by early/middle/late decision stratum;
- metrics by task type and `valid_seen` versus `valid_unseen`;
- performance versus rollout count `K`;
- calibration by predicted Q decile;
- actor log-probability baseline;
- direct C3 estimates at the same locked branch states and rollout budgets;
- GiGPO anchor-state credit diagnostics where grouping is well-defined;
- no-tree, independent-Q, and optional hindsight-aux ablations;
- environment steps, actor tokens, critic tokens, wall time, and accelerator hours.

### 10.6 Inference

- Bootstrap game IDs with all their states/actions intact.
- Use paired differences between variants within each bootstrap replicate.
- Report native effect sizes and confidence intervals, not bare p-values.
- The two primary baseline comparisons use familywise 95% coverage, preferably a paired
  max-statistic bootstrap or, as a simpler implementation, Bonferroni-adjusted 97.5%
  intervals.

### 10.7 Proposed E11 pass gate

E11 passes only if all conditions hold:

1. Stage-1 replay and signal gate passes.
2. All variants have identical parameter counts.
3. For seed 0, `docm_dueling_credit` has lower locked-reference Q MSE than both
   `docm_outcome` and `docm_prefix`, with both simultaneous paired intervals excluding zero
   in the favorable direction.
4. Mean candidate-set action regret is at least 10% lower than the best predictive baseline,
   with the paired interval excluding zero.
5. `docm_dueling_credit` terminal Brier score is no more than `0.02` worse than
   `docm_outcome`.
6. For each primary Q-MSE comparison, at least four of five seeds favor
   `docm_dueling_credit`, and
   no seed has an interval excluding zero in the adverse direction.
7. The policy-centering identity holds to `1e-6` on every unmasked evaluation row, and the
   tree-consistent model improves held-out tree residual over `docm_dueling_no_tree` without
   adverse Q-MSE evidence.
8. The 90% conformal Q interval achieves at least 87% empirical locked coverage (the exact
   finite-sample tolerance is frozen from the planned reference-set size) while reporting
   its width.

C3 and GiGPO are required reported baselines but are not part of the initial binary E11
gate because they answer partly different amortization and optimization questions. A
paper-facing claim that LongFeedback is better than modern direct-credit alternatives must
add a separately frozen superiority or compute-efficiency criterion against them; merely
running them is not evidence of superiority.

The 10% regret and 0.02 Brier thresholds are proposals to freeze before the full run. Pilot
results may inform compute size and variance, but not the sign or definition of these metrics.

## 11. E12: online LLM post-training

E12 starts from the exact actor checkpoint used in full E11. If E11 used a common SFT warm
start, all E12 methods start from that same immutable checkpoint.

### 11.1 Compared methods

Required methods:

1. `frozen_actor`: no post-training; reference success level.
2. `terminal_grpo`: grouped complete trajectories with terminal success advantage.
3. `prefix_group`: group-policy update using a prefix-value baseline trained from terminal
   returns.
4. `c3_group`: direct fixed-history branch credit used without an amortized LongFeedback
   critic, under the same total rollout-token budget.
5. `gigpo`: anchor/repeated-state macro/micro group credit, using the authors' pinned public
   implementation or a clearly labeled faithful reimplementation.
6. `longfeedback_group`: group-policy update using policy-centered candidate action values
   trained from direct V, branch Q, and tree targets.

The first engineering vertical slice may run methods 1-3 and 6. Methods 4-5 are required
before a full E12 result is called paper-ready, and their reproduction audit must be
completed before opening locked `valid_unseen` results.

Diagnostic upper bounds, not fair deployable baselines:

- greedy high-K branch oracle on selected states;
- expert ALFWorld policy where available.

All trainable methods use the same actor model/revision, tokenizer, prompt, LoRA rank and
target modules, optimizer budget, maximum context, admissible-action policy, and initial
checkpoint.

### 11.2 Online iteration

For policy iteration `j`:

1. Freeze actor checkpoint `pi_j`; hash weights and adapter.
2. Collect on-policy base trajectories under `pi_j`.
3. For LongFeedback and C3, select branch states by the same frozen outcome-blind rule and
   collect their predeclared low-K branch targets under the same frozen checkpoint. GiGPO
   receives its method-specific grouped rollouts from the same primary token budget.
4. Train or refresh the critic using only labels whose continuation policy is `pi_j` or falls
   within the explicitly configured maximum policy lag.
5. Freeze the critic for the actor update.
6. For LongFeedback, compute selected-action advantage directly from the dueling critic:

   ```text
   A_LF(h_t,a_t) = Q_hat(h_t,a_t) - V_hat(h_t)
   ```

   The architecture asserts that `V_hat = sum_a pi_j(a|h_t) Q_hat`; log the numerical
   residual and abort the update if it exceeds tolerance. C3 uses its direct branch
   contrast, and GiGPO uses its published macro/micro grouped advantage. Center and
   normalize advantages using only each method's current on-policy batch.
7. Run a fixed number of shared group-policy update epochs with clipped categorical
   probability ratios, entropy regularization, and KL to the immutable initial reference
   actor. This optimization shell is held constant; the compared methods differ in credit.
8. Save `pi_{j+1}`, clear or quarantine stale policy-dependent credit labels, and evaluate on
   the development split at the predeclared interval.

The actor does not backpropagate through the critic. The critic and actor are frozen in turn,
avoiding a moving-target update inside one policy minibatch.

### 11.3 Shared group-policy objective

The action is one categorical choice among admissible commands. Let
`r_t(theta) = pi_theta(a_t|h_t) / pi_j(a_t|h_t)`.

```text
L_policy = -mean(min(r_t A_t,
                     clip(r_t, 1-eps, 1+eps) A_t))
L_total  = L_policy + beta_kl * KL(pi_theta || pi_reference)
                     - beta_entropy * H(pi_theta)
```

Only candidate-action policy probabilities are optimized. Prompt tokens, observations, and
environment text receive no direct loss. LoRA is the default update mechanism; full-parameter
training is out of scope for the first E12 run.

### 11.4 Terminal GRPO budget use

Branch rollouts are expensive. A compute-fair terminal baseline must be allowed to spend the
same actor-rollout token budget on additional complete grouped trajectories. It may not be
restricted to the number of LongFeedback base episodes while LongFeedback receives many
extra branch continuations.

For `terminal_grpo`, group trajectories by the same initial game and actor checkpoint. Use
group-normalized terminal rewards, with a declared convention for all-success/all-failure
groups. The number of group rollouts is selected to consume the same primary actor-forward
token budget as LongFeedback base plus branch continuations.

### 11.5 Prefix-credit group baseline

Use the capacity-matched prefix head trained from terminal returns. Its advantage is the
terminal return minus the learned prefix baseline or a fixed-lambda GAE estimate. Do not feed
branch targets to this baseline.

### 11.6 C3 and GiGPO baselines

`c3_group` uses direct forced continuations at the selected history and forms the same
policy-centered advantage target used to supervise LongFeedback. It does not train an
amortized Q model. If the selected logged action lacks enough direct samples, the row is
excluded under a rule frozen before evaluation; its unused budget may not be reassigned after
observing outcomes.

`gigpo` follows its pinned definition of repeated anchor states, group construction, macro
outcome credit, and micro step credit. Any ALFWorld adaptation needed to define anchor-state
equality must be written into the frozen configuration and validated in the fake
environment. Do not silently substitute ordinary GRPO when anchors are sparse.

### 11.7 Policy-dependent label freshness

Action credit is defined relative to a continuation policy. Every branch label stores
`continuation_policy_id`.

Default rule:

- current-iteration labels: allowed;
- one-iteration-old labels: allowed only in a declared sensitivity run;
- older labels: excluded from primary critic fitting;
- high-K E11 reference labels: evaluation only, never E12 actor training.

This intentionally sacrifices replay-buffer efficiency to keep the estimand coherent. A later
off-policy correction is a separate research contribution.

### 11.8 Compute accounting

Primary matched budget:

- total actor-forward tokens used for base rollouts, group rollouts, and branch
  continuations.

Also report:

- prompt and candidate tokens separately;
- actual model-forward tokens after batching/caching;
- critic embedding and training tokens/operations;
- environment steps;
- number of complete and partial rollouts;
- wall-clock time;
- peak memory;
- accelerator type/count and GPU hours; and
- API cost, if any.

The primary comparison is rollout-token matched. A secondary efficiency-frontier plot reports
success against total measured compute, making LongFeedback's critic overhead visible rather
than pretending it is free.

C3 receives the same branch-state selection opportunities and total continuation tokens;
GiGPO receives the same total actor-forward tokens in its required grouping structure.
Unused method-specific budget is reported, not covertly transferred. All methods also report
total accelerator compute so an amortized critic is not presented as costless.

### 11.9 Evaluation schedule

- Training and hyperparameter selection use `train` and `valid_seen` only.
- Run deterministic and fixed-temperature stochastic evaluations at predeclared checkpoints.
- Select one checkpoint per method using a rule frozen before `valid_unseen` access, such as
  best `valid_seen` success with a KL/step tie-breaker.
- Evaluate the selected checkpoints on locked `valid_unseen` once per seed.
- All methods use identical game IDs and evaluation seeds.

### 11.10 E12 metrics

Primary:

- locked `valid_unseen` task success rate under the rollout-token-matched budget.

Secondary:

- `valid_seen` success;
- success-area-under-the-training-curve versus actor-forward tokens;
- environment steps and tokens per successful task;
- task success by task type;
- terminal KL from the initial actor;
- policy entropy and collapse indicators;
- actor invalid-action rate (expected zero under admissible scoring);
- E11 reference-state Q MSE and action regret at each saved policy checkpoint;
- critic calibration and policy-lag sensitivity; and
- uncertainty coverage/width and advantage risk-coverage at each policy iteration;
- online branch rollouts required per unit success improvement, especially versus C3;
- GiGPO anchor availability and effective grouped sample size; and
- total-compute efficiency frontier.

### 11.11 E12 inference

- Use five training seeds with paired initialization and game schedules.
- For final success comparisons, bootstrap locked evaluation game IDs, paired across methods.
- Report per-seed results and a bootstrap interval over paired seed-level differences.
- Use familywise 95% coverage for the predeclared primary baseline comparisons.
- Do not average `valid_seen` and `valid_unseen` into one score.

### 11.12 Proposed E12 pass gate

E12 passes only if:

1. E11 passed.
2. No method exceeded the primary actor-rollout token budget.
3. `longfeedback_group` exceeds both `terminal_grpo` and `prefix_group` on locked
   `valid_unseen` success by at least 3 percentage points, with both simultaneous paired
   confidence intervals excluding zero.
4. At least four of five seeds favor LongFeedback against each required baseline, with no
   seed showing a significant adverse effect.
5. LongFeedback's Q-MSE and regret on the fixed E11 diagnostic reference set do not degrade
   beyond the E11 noninferiority tolerance at the selected final checkpoint.
6. No safety/reproducibility failure occurred: test leakage, replay mismatch, stale-label use,
   missing budget accounting, or unpinned actor revision is an automatic fail.

This is the **core mechanism gate** against terminal and prefix credit. It is not sufficient
for a paper-ready comparative claim. Paper readiness additionally requires audited C3 and
GiGPO runs under the frozen budget contract. The result must then be described honestly as
one of:

- superiority: LongFeedback's locked success interval excludes zero favorably;
- noninferiority plus efficiency: success is within the frozen noninferiority margin and
  LongFeedback uses materially fewer online branch/group rollout tokens; or
- no advantage: neither condition holds.

The superiority/noninferiority margins and the efficiency definition must be frozen before
the locked comparison. Optional semantic hindsight cannot rescue the primary verdict; it is
reported as a separate ablation.

## 12. Configuration design

Add strict Pydantic models local to the E11/E12 experiment modules initially, following the
E10 Phase-2 pattern. Promote shared settings only after both runners use them.

Suggested E11 YAML shape:

```yaml
name: e11_alfworld_credit
seed: 0
output_dir: artifacts/e11_alfworld_credit

environment:
  backend: subprocess
  env_type: AlfredTWEnv
  data_dir: data/alfworld
  max_steps: 50
  worker_count: 4
  request_timeout_seconds: 120

actor:
  model_id: REQUIRED_PINNED_MODEL
  model_revision: REQUIRED_COMMIT
  tokenizer_revision: REQUIRED_COMMIT
  prompt_template: alfworld_candidate_v1
  temperature: 1.0
  score_length_normalization: mean

collection:
  profile: pilot
  base_episodes: 200
  branch_states_per_episode: 2
  full_enumeration_limit: 8
  top_actor_candidates: 2
  random_candidates: 1
  train_rollouts_per_action: 4
  reference_states: 50
  reference_rollouts_per_action: 32
  unforced_rollouts: 32
  child_direct_value_fraction: 0.25
  child_unforced_rollouts: 16
  calibration_states: 50
  calibration_rollouts_per_action: 32

model:
  embedding_model_id: REQUIRED_PINNED_MODEL
  embedding_revision: REQUIRED_COMMIT
  d_model: 64
  n_layers: 2
  n_heads: 4
  action_mlp_hidden: 128
  dropout: 0.0
  action_value_parameterization: policy_centered_dueling
  center_over: full_admissible_set
  policy_center_tolerance: 1.0e-6
  target_network_ema: 0.99

uncertainty:
  method: bootstrap_ensemble
  members: 5
  interval_method: conformalized_ensemble
  target_coverage: 0.90
  calibration_split: valid_seen

hindsight_aux:
  enabled: false
  teacher_id: null
  prompt_version: null
  label_schema_version: v1

training:
  seeds: [0, 1, 2, 3, 4]
  epochs: 40
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.01
  grad_clip: 1.0
  loss_weights:
    outcome: 1.0
    direct_v: 1.0
    branch_q: 1.0
    tree: 0.25
    prefix_mc: 0.0
    hindsight_aux: 0.0

baselines:
  required_e11: [docm_outcome, docm_prefix, docm_dueling_credit, c3_direct, gigpo_credit]
  architecture_ablations: [docm_dueling_no_tree, docm_independent_q]
  optional_ablations: [docm_dueling_hindsight_aux]

decision:
  actor_success_min: 0.10
  actor_success_max: 0.90
  reference_median_se_max: 0.05
  informative_q_spread: 0.10
  informative_state_fraction_min: 0.30
  minimum_reference_states: 200
  regret_relative_improvement: 0.10
  outcome_brier_tolerance: 0.02
  minimum_positive_seeds: 4
  bootstrap_resamples: 2000
  familywise_confidence: 0.95
  uncertainty_coverage: 0.90
  uncertainty_coverage_tolerance: 0.03
  policy_center_tolerance: 1.0e-6

budget:
  max_actor_forward_tokens: REQUIRED
  max_environment_steps: REQUIRED
  max_wall_time_seconds: REQUIRED
  max_api_cost_usd: 0.0
```

The tracked full configuration must not contain placeholders when the confirmatory contract
is frozen.

Suggested E12 adds:

```yaml
name: e12_alfworld_online
initial_actor_manifest: artifacts/e11_alfworld_credit/actor_manifest.json
methods: [frozen_actor, terminal_grpo, prefix_group, c3_group, gigpo, longfeedback_group]

online:
  iterations: REQUIRED
  base_episodes_per_iteration: REQUIRED
  branch_states_per_episode: REQUIRED
  branch_rollouts_per_action: REQUIRED
  max_policy_lag: 0
  evaluation_interval: REQUIRED

actor_training:
  lora_rank: REQUIRED
  lora_alpha: REQUIRED
  ratio_clip: 0.2
  update_epochs: REQUIRED
  learning_rate: REQUIRED
  kl_coefficient: REQUIRED
  entropy_coefficient: REQUIRED
  max_grad_norm: 1.0

decision:
  minimum_success_lift: 0.03
  minimum_positive_seeds: 4
  bootstrap_resamples: 2000
  familywise_confidence: 0.95
  paper_superiority_margin: REQUIRED
  paper_noninferiority_margin: REQUIRED
  paper_efficiency_definition: REQUIRED
```

## 13. Proposed package layout

```text
src/longfeedback/environments/
    base.py                     # protocols and immutable inner-loop records
    alfworld.py                 # clients, normalization, replay handles
src/longfeedback/actors/
    base.py                     # CandidatePolicy protocol
    llm_candidates.py           # frozen and LoRA candidate-scoring actor
src/longfeedback/credit/
    branching.py                # replay verification and MC Q/V targets
    tree_targets.py             # parent-Q/child-V target construction
    c3.py                       # direct fixed-history credit baseline
    gigpo.py                    # pinned anchor/group credit adapter
    hindsight.py                # auxiliary label schema only
src/longfeedback/models/
    candidate_docm.py           # semantic policy-centered three-head model
    uncertainty.py              # bootstrap ensemble and conformal calibration
src/longfeedback/training/
    group_policy.py             # shared categorical group-policy update
    online.py                   # synchronous actor/critic iteration
src/longfeedback/experiments/
    e11_alfworld_credit.py
    e12_alfworld_online.py
scripts/alfworld/
    worker.py                   # optional Python-3.9 subprocess service
configs/experiments/
    e11_alfworld_credit_smoke.yaml
    e11_alfworld_credit.yaml
    e12_alfworld_online_smoke.yaml
    e12_alfworld_online.yaml
```

Because E11 and E12 both use the environment, actor, and branching lifecycle, extracting
these shared modules is consistent with ADR-001/Gate-C's “at least two consumers” rule.

## 14. CLI and Make targets

Proposed commands:

```bash
longfeedback environment prepare alfworld --config configs/environments/alfworld.yaml
longfeedback environment audit alfworld --output-dir artifacts/alfworld_audit

longfeedback experiment run e11_alfworld_credit \
  --config configs/experiments/e11_alfworld_credit.yaml

longfeedback experiment run e12_alfworld_online \
  --config configs/experiments/e12_alfworld_online.yaml
```

Make targets:

```text
make alfworld-bootstrap
make alfworld-audit
make e11-smoke
make e11
make e12-smoke
make e12
```

The existing core bootstrap and CPU experiments must not install ALFWorld, Transformers,
PEFT, CUDA, or visual dependencies. Put agentic dependencies behind a separate environment
or optional extra.

## 15. Testing plan

### 15.1 Unit tests

- Environment text/action normalization and stable hashing.
- Replay-handle serialization.
- Branch target mean, standard error, and seed derivation.
- Candidate selection, deduplication, and selection probabilities.
- Candidate masks and variable-horizon padding.
- Semantic candidate Q-head shapes and no padded-action influence.
- Policy-centered dueling Q uses the full admissible policy distribution.
- `V == sum_a pi(a|h) Q(h,a)` to `1e-6`, including extreme logits and masks.
- Adding a constant to every raw advantage does not change Q.
- Outcome/prefix candidate scoring without post-action leakage.
- Binomial direct-V and branch-Q losses from `(success_count, K)`.
- Tree targets use terminal success, direct child V, and lagged target V in the correct cases.
- Gradients never flow through lagged tree targets.
- Hindsight auxiliary gradients cannot directly update Q/V output parameters.
- C3 policy-centered/leave-one-out contrast and budget accounting.
- GiGPO anchor grouping and macro/micro credit on a hand-computed fixture.
- Conformal interval fitting and serialization.
- Actor length-normalized candidate probabilities.
- Group-policy ratio clipping, KL, entropy, and advantage centering.
- Policy-lag filter rejects stale labels.
- Budget ledger counts every base/group/branch continuation.

### 15.2 Property tests

- Prefix representations are invariant to changes in future observations/actions.
- Reordering the displayed admissible action list does not change command probabilities after
  canonical normalization.
- Adding a new branch does not change the base trajectory RNG stream.
- Candidate padding does not change predictions for real candidates.
- Reordering candidates and matching probabilities leaves policy-centered Q unchanged.
- Semantically identical canonical commands have identical frozen embeddings.
- Locked-reference rows cannot enter a training dataset.
- Calibration and target-network datasets cannot contain locked-reference rows.
- Future trajectory edits may change teacher hindsight labels but cannot change the
  hindsight student's inference input.

### 15.3 Integration tests

- A tiny deterministic fake text environment exercises reset, step, replay, branching, and
  target aggregation in CI without downloading ALFWorld.
- E11 smoke runs end to end with a mock candidate policy and one training epoch.
- The fake environment verifies direct V, branch Q, terminal/nonterminal tree targets, C3,
  and uncertainty artifacts against known values.
- E12 smoke runs two synchronous iterations on the fake environment and verifies checkpoint,
  policy ID, stale-label, C3/GiGPO grouping, policy-centering, and budget behavior.
- A local/optional ALFWorld marker runs one real TextWorld game and replay audit when data is
  installed.

### 15.4 Reproducibility tests

- Identical E11 smoke configuration and seeds produce identical scientific metrics.
- Identical E12 fake-environment smoke runs produce identical checkpoints or declared
  deterministic metric hashes on supported hardware.
- Cached and uncached actor candidate scores agree within tolerance.

## 16. Failure handling

| Failure | Required behavior |
|---|---|
| Environment/replay hash mismatch | abort affected target; fail replay gate |
| Worker timeout/crash | retry with bounded count; record infrastructure failure separately |
| Actor returns NaN scores | abort episode; never reinterpret as task failure |
| All candidate scores equal | sample uniformly; record degeneracy |
| All branch outcomes identical | retain valid target with zero empirical variance; signal gate handles lack of action spread |
| Budget exhausted | stop cleanly, mark run incomplete, no pass verdict |
| Stale continuation-policy label | exclude from primary critic dataset |
| Locked-reference leakage | hard fail and invalidate run |
| Missing full actor distribution or policy hash | hard fail; policy-centered Q is undefined |
| Policy-centering residual above tolerance | abort batch/update and emit diagnostic |
| Missing/mismatched forced child-state hash | discard target and fail tree audit if systematic |
| Uncertainty calibrator fitted on locked data | hard fail and invalidate run |
| Hindsight label reaches Q/V target path or locked selection | hard fail and invalidate run |
| C3/GiGPO exceeds or silently reallocates budget | mark comparison invalid |
| Missing model/data revision | configuration validation failure |
| OOM during full run | reduce batch/worker implementation settings only if logical policy and frozen compute contract remain unchanged; otherwise create a new run contract |

## 17. Execution sequence

### Milestone A: environment foundation

1. Add environment protocols and fake deterministic environment.
2. Implement ALFWorld worker/client.
3. Pin source and dependency revisions.
4. Implement environment preparation and source manifest.
5. Pass real text-only reset/step smoke.
6. Pass the replay audit.

Exit criterion: exact replay works and CI tests do not require ALFWorld data.

### Milestone B: frozen actor and collection

1. Implement prompt renderer and candidate normalization.
2. Implement frozen candidate-scoring actor plus mock actor.
3. Add token/budget accounting.
4. Collect base trajectories.
5. Implement outcome-blind branch state/candidate selection.
6. Generate direct unforced V targets, low-K branch Q targets, forced child edges, and
   artifact tables.

Exit criterion: E11 smoke produces auditable base and branch artifacts.

### Milestone C: candidate DOCM

1. Add padding-mask support without altering existing no-mask behavior.
2. Implement frozen text embedding cache.
3. Implement `CandidateSequenceDataset`.
4. Implement the three heads, semantic action scorer, and policy-centered dueling Q.
5. Implement direct-V/branch-Q binomial losses, target-network tree consistency, and the
   capacity-matched variants.
6. Implement the bootstrap ensemble and development-only conformal calibration.
7. Add unit/property/reproducibility tests.

Exit criterion: models recover known Q values in the fake environment.

### Milestone D: E11 pilot and contract freeze

1. Run pilot collection on `train`/`valid_seen`.
2. Determine whether a common actor warm start is required.
3. Measure rollout cost and reference-target precision.
4. Audit direct C3 and GiGPO credit adapters on fake and development data.
5. Select tree and uncertainty settings using development data.
6. Set the full token/environment-step budget.
7. Finalize, review, and commit the E11 acceptance contract.
8. Generate the locked reference set only after the contract is frozen.

Exit criterion: no placeholders remain in the full E11 configuration.

### Milestone E: E11 full run

1. Generate training branches and locked high-K references.
2. Evaluate the signal gate.
3. If authorized, train all variants for all seeds and run required C3/GiGPO diagnostics.
4. Produce paired clustered intervals, uncertainty audits, and diagnostic plots.
5. Record an immutable pass/fail decision and scientific-metrics hash.

Exit criterion: E11 has an auditable verdict; thresholds are not changed afterward.

### Milestone F: E12 actor update

1. Implement LoRA candidate policy and the shared categorical group-policy update.
2. Implement terminal GRPO, prefix-group, direct C3-group, and pinned GiGPO baselines.
3. Implement synchronous actor/critic iteration and label freshness.
4. Implement checkpoint selection and locked evaluation rules.
5. Verify budget matching in the fake environment.

Exit criterion: E12 smoke improves a known toy policy without budget or policy-ID errors.

### Milestone G: E12 pilot and full run

1. Pilot only on `train`/`valid_seen`.
2. Freeze iterations, budgets, group-policy settings, and checkpoint-selection rule.
3. Complete the C3/GiGPO reproduction and budget audit before locked evaluation.
4. Run all methods and seeds under identical schedules.
5. Select checkpoints without `valid_unseen` access.
6. Run locked evaluation once per seed.
7. Produce the core and paper-readiness decision blocks, learning curves, compute frontier,
   and manifest.

Exit criterion: E12 has an auditable pass/fail verdict.

### Milestone H: optional extensions and domain transfer

1. Run the optional semantic hindsight auxiliary ablation without changing the primary
   verdict.
2. Add any newly published compatible baseline under a separately frozen contract.
3. Connect the target-provider interface to randomized DCEE/HeartSteps and later
   observational OMA-DCEE/LinkedIn.

## 18. Minimum implementation pull requests

Keep changes reviewable. A recommended PR sequence is:

1. `E11-01`: environment protocols, fake environment, hashing, replay tests.
2. `E11-02`: ALFWorld subprocess client/worker and local smoke command.
3. `E11-03`: candidate actor protocol, prompt contract, budget ledger.
4. `E11-04`: direct V/branch Q generator, child-tree edges, and Parquet artifacts.
5. `E11-05`: semantic policy-centered dueling DOCM, padding, and losses.
6. `E11-06`: tree targets, bootstrap uncertainty, calibration, and tests.
7. `E11-07`: direct C3 and GiGPO credit baselines plus budget audit.
8. `E11-08`: E11 runner, configs, CLI, metrics, and smoke integration.
9. `E11-09`: pilot contract update, followed by locked full run artifacts.
10. `E12-01`: LoRA candidate actor and shared group-policy objective tests.
11. `E12-02`: terminal GRPO, prefix-group, C3-group, and GiGPO baselines.
12. `E12-03`: synchronous online runner, policy freshness, and compute matching.
13. `E12-04`: E12 configs, CLI, smoke test, reporting, and locked run.
14. `E12-05`: optional semantic hindsight auxiliary ablation.

## 19. Claim boundaries

If E11 passes, the permitted claim is:

> At fixed model capacity, direct continuation V targets, forced-action Q targets,
> policy-centered dueling structure, and tree consistency improve recovery of
> continuation-policy-specific action credit in a resettable, verifiable, multi-step public
> environment relative to terminal- and prefix-only supervision.

This is an amortized prediction claim. It does not say LongFeedback invented fixed-history
branching or outperforms C3/GiGPO unless the separately frozen comparisons support those
statements. Calibrated uncertainty means empirical coverage at the reported width; it does
not make model predictions ground truth.

If E12 also passes, the permitted additional claim is:

> Under a matched actor-rollout budget, LongFeedback action credit improves online LLM
> post-training success and/or sample efficiency over terminal- and prefix-credit baselines.

A comparison to modern direct-credit methods is publishable only after the audited C3 and
GiGPO results are included and labeled as superiority, noninferiority-plus-efficiency, or no
advantage according to the frozen rule. Semantic hindsight supports only an auxiliary
representation result.

Neither result by itself permits the claim that historical LinkedIn email effects are causal.
For LinkedIn, the same action-value head will require randomized or doubly robust DCEE target
construction, overlap diagnostics, and ideally an untouched internal randomized validation
slice. The scientific bridge is a shared estimand and learning architecture, not an assertion
that replay and observational identification are the same procedure.

## 20. Reference papers and implementations

These references define the external methods or motivate specific design components. A link
does not mean LongFeedback is an exact reproduction; the `Use in this design` column states
the relationship.

| Component | Primary paper | Official/reference implementation | Use in this design |
|---|---|---|---|
| Credit-assignment landscape | [From Reasoning to Agentic: Credit Assignment in RL for LLMs](https://arxiv.org/abs/2604.09459) | Not applicable | Taxonomy and contemporary-method review |
| ALFWorld | [ALFWorld: Aligning Text and Embodied Environments for Interactive Learning](https://arxiv.org/abs/2010.03768) | [alfworld/alfworld](https://github.com/alfworld/alfworld) | Resettable, verifiable E11/E12 environment |
| Semantic action scoring | [Deep Reinforcement Learning with a Natural Language Action Space](https://arxiv.org/abs/1511.04636) | Paper reference | Motivates separate semantic state/action representations; not the full DOCM |
| Dueling V/A factorization | [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) | Paper reference | Architectural ancestor; LongFeedback's bounded full-policy centering is new adaptation |
| Prefix/return redistribution baseline | [RUDDER: Return Decomposition for Delayed Rewards](https://arxiv.org/abs/1806.07857) | Paper reference | Historical predictive-credit baseline only |
| Direct contextual counterfactual credit | [C3: Contextual Counterfactual Credit Assignment](https://arxiv.org/abs/2603.06859) | [EIT-EAST-Lab/C3](https://github.com/EIT-EAST-Lab/C3) | Required direct fixed-history branching baseline |
| Hierarchical group credit | [Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978) | [langfengQ/verl-agent](https://github.com/langfengq/verl-agent) | Required critic-free anchor-state baseline |
| Semantic role feedback | [TRIAGE: Role-Typed Credit Assignment for Agentic RL](https://arxiv.org/abs/2606.32017) | No verified official repository at design time | Motivation for optional auxiliary labels; not primary Q/V supervision |
| Deep-ensemble uncertainty | [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | Paper reference | Between-model epistemic uncertainty |
| Conformal calibration | [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511) | [aangelopoulos/conformal-prediction](https://github.com/aangelopoulos/conformal-prediction) | Development-only interval calibration and coverage audit |
| GRPO terminal baseline | [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) | Method implemented through the E12 framework | Terminal group-relative baseline; adapted to complete ALFWorld trajectories |

Before implementation, record immutable commit hashes for ALFWorld, C3, and GiGPO in the
resolved experiment configuration. A moving `main` or `master` branch is not a reproducible
method definition.

## 21. Decisions required before the full implementation freezes

The following values intentionally remain unresolved in this draft:

1. exact actor model and immutable revision;
2. exact frozen text-embedding model and revision;
3. whether a common SFT warm start is necessary;
4. full E11 actor-token/environment-step budget;
5. full E12 actor-token budget and iteration count;
6. LoRA rank/target modules;
7. whether the primary full runs use local GPU inference or an approved provider;
8. final simultaneous-inference implementation;
9. target-network EMA, tree-loss weight, and direct-child rollout allocation;
10. ensemble size, interval method, calibration-set size, and finite-sample coverage
    tolerance;
11. exact C3 contrast/budget convention and pinned GiGPO revision plus ALFWorld anchor-state
    definition;
12. paper-facing superiority/noninferiority and rollout-efficiency margins; and
13. whether to run the optional hindsight auxiliary, including its teacher, prompt, schema,
    loss weight, and compute budget.

These are implementation or resource choices, not reasons to delay the environment, replay,
fake-world, schema, model-interface, and test work in Milestones A-C.
