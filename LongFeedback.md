# LongFeedback: Complete Project Design

> Combined GPT-friendly specification. The source documents below remain the canonical editable files.

---

# LongFeedback: RL from Delayed, Implicit Behavioral Outcomes

> **Working project name:** `LongFeedback`  
> **Status:** implementation design v0.1  
> **Primary objective:** create a credible research result and an open-source system for training language agents from delayed, implicit user outcomes.  
> **Target audience:** researchers and research engineers working on LLM post-training, reward modeling, agent learning, offline RL, causal evaluation, and production feedback loops.

---

## 1. Program thesis

Modern LLM post-training mostly assumes that a reward can be produced shortly after a model response: a verifier checks an answer, a judge scores a completion, or a human compares two outputs. Deployed agents often receive their most important signals later and indirectly:

- the user corrects the model two turns later;
- the user repeats the request because the original answer failed;
- the user abandons the workflow;
- the task is completed after several interactions;
- the user returns, renews, or follows through days later;
- an intervention improves engagement while reducing trust or user utility.

This program studies **delayed implicit behavioral outcomes** rather than treating all of these phenomena as a single vague category of “non-verifiable reward.” The program separates six sources of difficulty:

1. **Reward delay:** the outcome is observed after many decisions.
2. **Outcome stochasticity:** the same action history can lead to different outcomes.
3. **Partial observability:** relevant user state is hidden.
4. **Logging-policy bias:** logged actions were chosen by another policy.
5. **Hidden confounding:** the logging policy may use information unavailable to the learner.
6. **Reward-proxy error:** a learned reward model may be optimized outside its reliable region.

The central scientific question is:

> **Can a model trained from delayed behavioral outcomes recover useful credit for earlier agent actions, and can policy optimization use that credit without exploiting reward-model errors?**

---

## 2. One project, two phases

The program should be presented as one coherent project with two phases.

### Phase 1 — Research system

Build and evaluate methods for:

- constructing delayed outcomes from public interaction logs;
- predicting trajectory-level outcomes;
- assigning predictive and, where identifiable, causal credit to earlier actions;
- quantifying uncertainty in the learned reward and credit signal;
- improving a policy conservatively;
- measuring reward-model overoptimization in controlled environments.

**Primary outputs:** a paper-style technical report, reproducible experiments, processed datasets, controlled structural environments, and a reference implementation of the proposed method.

### Phase 2 — Delayed-RL infrastructure

Extract the reusable system abstractions required to train from rewards that arrive after rollout completion:

- event-sourced trajectories;
- asynchronous outcome joining and reward backfill;
- censoring, expiry, and late corrections;
- policy and reward-model versioning;
- stale-policy handling and off-policy correction;
- integrations with `verl`, TRL, and optionally OpenRLHF;
- monitoring of proxy reward, observed outcome, utility, uncertainty, and policy drift.

**Primary outputs:** a reusable Python library, storage schema, trainer adapters, examples, reliability tests, and contributor documentation.

### Why the order matters

Phase 2 must be extracted from working Phase 1 experiments. Infrastructure built first would likely generalize the wrong abstractions. The research phase establishes what data must be stored, which reward lifecycle states matter, which corrections are necessary, and what evaluation signals are meaningful.

---

## 3. Success criteria

The project is successful only if it produces all three of the following.

### 3.1 Scientific result

At least one non-trivial empirical claim should survive multiple structural environments and a real-log replication. A candidate claim is:

> A trajectory model that jointly learns terminal outcome, per-step intervention value, and epistemic uncertainty recovers delayed credit more accurately than outcome-only reward models or return redistribution, and uncertainty-aware policy optimization substantially delays reward hacking.

This is a hypothesis, not a promised result. The implementation plan includes stop/go criteria if it fails.

### 3.2 Reusable artifact

An external researcher should be able to:

1. convert a sequence of interaction events into a versioned trajectory;
2. attach an outcome that arrives later;
3. train an outcome or credit model;
4. backfill rewards into stored rollouts;
5. run offline or delayed-online optimization;
6. evaluate calibration, credit recovery, and proxy-versus-true performance.

### 3.3 Clear career narrative

The public narrative should be:

> “I develop methods and systems for training language agents from delayed, implicit user outcomes. I evaluate causal credit in controlled structural environments, validate behavioral outcome modeling on real interaction logs, and build infrastructure for reliable asynchronous post-training.”

Do not lead with “email simulator” or “anti-SWE-Bench.” The lifecycle-messaging domain may remain an illustrative environment, but it is not the identity of the project.

---

## 4. Scope decisions

### 4.1 In scope

- Multi-turn language interactions with delayed future-user feedback.
- Sequential behavioral logs with randomized or known exposure when available.
- Controlled structural causal environments with exact interventions.
- Trajectory-level reward models.
- Per-step predictive contribution and interventional credit.
- Outcome censoring and delayed arrival.
- Conservative offline policy improvement.
- Small-model or LLM post-training demonstrations.
- Reward-model overoptimization and proxy/utility divergence.

### 4.2 Explicitly out of scope for the first release

- Claiming that a synthetic user model predicts real users.
- A public leaderboard.
- Generated natural-language actions in the initial controlled environments.
- Production-scale streaming infrastructure.
- Cross-organization user identity resolution.
- Clinical, financial, or other high-stakes deployment recommendations.
- Claiming causal effects from observational chat logs without defensible assumptions.
- Fitting proprietary employer data or publishing employer-derived parameters.

### 4.3 Terminology

Use these terms consistently:

- **Outcome:** an observed variable measured after one or more actions.
- **Behavioral proxy:** an observable user behavior correlated with, but not identical to, user utility.
- **Utility:** the objective the system should ultimately improve. It may be latent in real logs and explicit only in simulation.
- **Predictive contribution:** how much an event changes a model’s prediction of the final outcome.
- **Causal credit:** the effect of intervening on an earlier action while specifying the future policy.
- **Reward redistribution:** a shaped per-step signal whose sum approximates the terminal return.
- **Reward backfill:** attaching a late-arriving outcome or learned reward to a previously stored rollout.
- **Censoring:** the outcome is not observed within the available window.
- **Policy staleness:** a rollout was produced by an older policy than the policy being optimized.

---

## 5. Program architecture

```text
PUBLIC INTERACTION LOGS                  CONTROLLED STRUCTURAL WORLDS
(WildChat / WildFB / KuaiRand)           (known SCM + interventions)
              │                                      │
              ├──────────► Canonical event schema ◄──┤
              │                                      │
              ▼                                      ▼
       Outcome construction                   Oracle counterfactuals
       + censoring metadata                   + true utility/proxy
              │                                      │
              └──────────► Phase 1 models ◄──────────┘
                              │
                ┌─────────────┼──────────────┐
                ▼             ▼              ▼
          Outcome RM   Credit model    Policy optimizer
                │             │              │
                └─────────────┼──────────────┘
                              ▼
                  Evaluation + overoptimization
                              │
                              ▼
                 Phase 2 reusable infrastructure
```

The two data tracks have different purposes:

- **Real logs provide ecological validity:** authentic language, user diversity, realistic ambiguity, and naturally occurring feedback.
- **Structural worlds provide identification:** exact intervention effects, repeated counterfactual rollouts, known latent state, and a separate true utility.

Neither track alone is sufficient.

---

## 6. Recommended repository organization

```text
longfeedback/
├── README.md
├── pyproject.toml
├── LICENSE
├── CITATION.cff
├── configs/
│   ├── data/
│   ├── worlds/
│   ├── models/
│   └── experiments/
├── docs/
│   ├── phase1_research_design.md
│   ├── phase2_infrastructure_design.md
│   ├── data_cards/
│   ├── model_cards/
│   └── causal_assumptions.md
├── src/longfeedback/
│   ├── schema/
│   ├── data/
│   ├── outcomes/
│   ├── worlds/
│   ├── models/
│   ├── credit/
│   ├── policies/
│   ├── ope/
│   ├── evaluation/
│   └── infra/
├── integrations/
│   ├── trl/
│   ├── verl/
│   └── openrlhf/
├── experiments/
│   ├── phase1/
│   └── phase2/
├── tests/
│   ├── unit/
│   ├── property/
│   ├── integration/
│   └── reproducibility/
├── scripts/
└── reports/
```

During the first four weeks, it is acceptable to keep Phase 1 under `research/` and postpone clean packaging. Refactor only after the first stable experiment.

---

## 7. Release sequence

### v0.1 — Scientific vertical slice

- one conversational dataset adapter;
- two structural worlds;
- canonical trajectory schema;
- outcome Transformer;
- RUDDER-style redistribution baseline;
- oracle credit evaluator;
- one policy-learning experiment;
- one end-to-end reproducibility command.

### v0.2 — Proposed method and robust evaluation

- joint outcome/credit/uncertainty model;
- four structural world families;
- temporal and structural distribution shift;
- hidden-confounding setting;
- real-log outcome modeling;
- calibration and uncertainty metrics;
- paper-style report.

### v0.3 — LLM post-training demonstration

- LLM or small language model as reward/credit model;
- candidate-response reranking or controlled action selection;
- proxy-versus-utility optimization curves;
- uncertainty-aware mitigation;
- compute and reproducibility report.

### v0.4 — Infrastructure extraction

- event store and trajectory state machine;
- outcome resolver and reward backfill;
- trainer-neutral batch API;
- TRL and `verl` adapters;
- local reference deployment using Parquet + SQLite/Postgres;
- failure recovery and idempotency tests.

### v1.0 — Stable research and infrastructure release

- frozen schema version;
- migration tools;
- documentation and tutorials;
- data/model cards;
- contributor guide;
- benchmark results treated as examples, not a universal leaderboard.

---

## 8. Evaluation philosophy

The project should distinguish five validity levels.

1. **Implementation validity:** mathematical quantities and data joins are computed correctly.
2. **Internal validity:** a method recovers known effects in controlled worlds.
3. **Robustness validity:** results persist across distinct causal structures, not just parameter variations.
4. **Ecological validity:** models operate on authentic public interaction logs.
5. **Predictive deployment validity:** method rankings transfer to a real product experiment.

The public project can establish levels 1–4. Level 5 requires a future external or private replication and should be named as an open limitation.

---

## 9. Recommended compute envelope

The initial program should remain executable by an individual researcher.

| Component | Default | Stretch |
|---|---:|---:|
| Structural simulations | CPU, 1–10M episodes | distributed CPU/Ray |
| Sequence baselines | 1 GPU, <24 GB | 4 GPUs |
| Text encoder/RM | 0.5B–3B model with LoRA | 7B–8B model |
| Policy experiment | discrete or reranking policy | full generated response |
| Storage | Parquet + DuckDB | object store + Postgres |
| Orchestration | local Python/Hydra | Ray/Slurm/Kubernetes |

A credible result with a 1–3B model is more valuable than an expensive but under-analyzed 8B run.

---

## 10. Security, privacy, and IP principles

- Preserve dataset license and attribution metadata.
- Never republish raw data if the source license prohibits redistribution.
- Store source row identifiers rather than raw sensitive text in public experiment manifests.
- Run PII and safety filters before creating derived examples.
- Split by conversation/user identifier before feature or label generation to prevent leakage.
- Treat implicit feedback labels as noisy behavioral proxies, not psychological ground truth.
- Do not use proprietary metrics, internal system names, confidential data, or employer code.
- Include a misuse section: optimizing behavioral signals can create manipulative policies.

---

## 11. Decision gates

### Gate A — after two weeks

Continue only if:

- at least one structural world shows a meaningful gap between terminal-outcome prediction and true per-step credit recovery;
- the oracle intervention evaluator is stable under repeated seeds;
- at least one baseline can improve policy value above behavior cloning without catastrophic proxy exploitation.

Otherwise simplify the world or change the target quantity before adding real logs.

### Gate B — after four to six weeks

Continue toward a research report only if:

- the proposed method beats outcome-only and RUDDER-style baselines on credit recovery in at least three structural families;
- the result is not explained only by parameter count;
- uncertainty correlates with error under at least one distribution shift;
- the real-log outcome task is learnable above trivial baselines.

If credit recovery does not improve but uncertainty reliably predicts error, reposition the contribution around **uncertainty-aware delayed reward modeling**.

### Gate C — before Phase 2 extraction

Extract infrastructure only if at least two experiments require the same trajectory/outcome lifecycle abstractions. Do not generalize a component used by only one script.

---

## 12. Files in this design package

- [`01_PHASE1_RESEARCH_DESIGN.md`](01_PHASE1_RESEARCH_DESIGN.md): detailed scientific design, data sources, equations, method, environments, experiments, and implementation plan.
- [`02_PHASE2_INFRASTRUCTURE_DESIGN.md`](02_PHASE2_INFRASTRUCTURE_DESIGN.md): detailed software architecture, schemas, lifecycle, trainer integrations, reliability requirements, and testing plan.


---

# Phase 1 Design: Learning Reward and Credit from Delayed Implicit Outcomes

> **Document status:** implementation design v0.1  
> **Phase objective:** produce a defensible research result, not merely a benchmark.  
> **Primary deliverable:** a reproducible method and evaluation suite for learning trajectory reward, temporal credit, and uncertainty from delayed behavioral outcomes.

---

## 1. Executive summary

Phase 1 asks whether delayed user behavior can provide a useful post-training signal for language agents. The project deliberately combines two evidence sources:

1. **Authentic public logs** provide realistic language, ambiguous feedback, diverse users, and natural behavioral signals.
2. **Controlled structural environments** provide exact interventions, repeatable counterfactuals, known latent variables, and a separate user-utility objective.

The main proposed model is provisionally called the **Delayed Outcome Credit Model (DOCM)**. DOCM jointly learns:

- a terminal outcome distribution;
- a sequence of predictive reward increments;
- a per-action interventional value head where intervention data are available;
- epistemic uncertainty through an ensemble or distributional head.

The core evaluation is not “did the model predict retention?” It is:

- did it predict delayed outcomes on held-out real logs;
- did it recover true intervention effects in structurally different controlled worlds;
- did a policy trained on its reward improve true utility;
- how quickly did policy optimization exploit model error;
- did uncertainty-aware pessimism reduce that exploitation?

---

## 2. Research claims and non-claims

### 2.1 Intended claims

A successful Phase 1 should support claims of the following form:

1. Delayed future-user behavior contains learnable information about earlier agent quality beyond immediate explicit feedback.
2. Terminal-outcome prediction accuracy is not sufficient for accurate temporal or causal credit.
3. Joint outcome, credit, and uncertainty training improves credit recovery in controlled environments.
4. Reward models with similar held-out prediction accuracy can produce materially different policy-optimization outcomes.
5. Uncertainty-aware conservative optimization can reduce reward-model exploitation under structural and policy distribution shift.

### 2.2 Claims that must not be made

- “The model infers true human satisfaction from chat logs.”
- “The model identifies causal effects in observational conversations.”
- “The simulator accurately models real users.”
- “A method ranking in the simulator will necessarily transfer to production.”
- “Delayed behavioral outcomes are inherently superior to explicit human preferences.”
- “Engagement or return behavior equals user welfare.”

### 2.3 Minimum publishable unit

The minimum coherent report is:

> A new joint delayed-outcome credit model, evaluated against outcome-only, RUDDER-style, offline-RL, and attribution baselines across multiple structural worlds, with a real conversational-log replication and a reward-overoptimization study.

A simulator release without the method/result is not the target.

---

## 3. Formal problem definition

### 3.1 Interaction process

For episode or user trajectory \(i\), define:

- \(o_t\): observation available to the agent at decision time \(t\);
- \(a_t\): action selected by the behavior or learned policy;
- \(x_t\): observed user/event response after the action;
- \(z_t\): latent user state, unobserved by the learner;
- \(h_t = (o_0,a_0,x_0,\ldots,o_t)\): observable history before action \(a_t\);
- \(T_i\): final decision time;
- \(Y_i\): delayed observed behavioral outcome;
- \(U_i\): true user utility, observable only in controlled environments;
- \(C_i\): censoring indicator;
- \(D_i\): delay between the last relevant action and outcome observation;
- \(\mu_i\): behavior policy or logging policy.

A trajectory is

\[
\tau_i = (h_{i,0},a_{i,0},x_{i,0},\ldots,h_{i,T_i},a_{i,T_i},x_{i,T_i},Y_i,C_i,D_i).
\]

The logged dataset is

\[
\mathcal D = \{(\tau_i,\mu_i, m_i)\}_{i=1}^N,
\]

where \(m_i\) contains provenance, policy version, timestamps, and label-generation metadata.

### 3.2 Outcome timing and censoring

Let \(T_i^Y\) be the event time when the behavioral outcome occurs and \(T_i^C\) the end of observation. We observe

\[
\tilde T_i = \min(T_i^Y,T_i^C),
\qquad
\delta_i = \mathbf 1[T_i^Y \le T_i^C].
\]

For fixed-window binary outcomes, define

\[
Y_i^{(K)} = \mathbf 1[T_i^Y \le T_i + K].
\]

If the log ends before \(K\), the sample is censored rather than automatically negative. The implementation must preserve \(\delta_i\) and the observation window.

### 3.3 Three distinct learning targets

The project must not collapse the following quantities.

#### A. Outcome prediction

\[
F_\theta(\tau_{0:T}) \approx P(Y=1\mid \tau_{0:T}).
\]

This asks whether the final outcome is predictable from the observed trajectory. It is useful for calibration and ranking but does not identify which actions caused the outcome.

#### B. Predictive contribution

Define a prefix predictor

\[
V_\theta(h_t) = \mathbb E_\theta[Y\mid h_t].
\]

A telescoping predictive increment is

\[
r_t^{\text{pred}}
= V_\theta(h_{t+1})-V_\theta(h_t).
\]

Then

\[
\sum_{t=0}^{T}r_t^{\text{pred}}
=V_\theta(h_{T+1})-V_\theta(h_0).
\]

This resembles return redistribution and is valuable for temporal localization. It remains predictive: an event can change the prediction without being causal.

#### C. Interventional action credit

For a specified future policy \(\pi_f\), reference action \(a^{\text{ref}}\), and history \(h_t\), define

\[
Q^{\pi_f}(h_t,a)
=
\mathbb E\left[
U \mid do(A_t=a), H_t=h_t,
A_{t+1:T}\sim\pi_f
\right].
\]

The action credit is

\[
C_t^{\pi_f}(a;a^{\text{ref}})
=
Q^{\pi_f}(h_t,a)-Q^{\pi_f}(h_t,a^{\text{ref}}).
\]

The future-policy clause is mandatory. “Replace today’s action with no-op” is ambiguous unless future behavior is specified. The default evaluation should use one of two estimands:

- **Frozen continuation:** future actions are fixed to the original recorded actions.
- **Policy-reactive continuation:** future actions are resampled from the same policy using the counterfactually changed history.

Report both when feasible. Frozen continuation isolates the local action effect; policy-reactive continuation measures total effect under adaptive downstream behavior.

### 3.4 Policy objective

For policy \(\pi\), true utility is

\[
J_U(\pi)=\mathbb E_{\tau\sim p^\pi}[U(\tau)].
\]

The observable behavioral proxy is

\[
J_Y(\pi)=\mathbb E_{\tau\sim p^\pi}[Y(\tau)].
\]

A learned reward model produces

\[
J_{\hat R}(\pi)=\mathbb E_{\tau\sim p^\pi}[\hat R_\theta(\tau)].
\]

Reward hacking is visible when optimization increases \(J_{\hat R}\) while \(J_U\) plateaus or declines. The controlled environments must define \(U\) separately from \(Y\).

### 3.5 Identification regimes

#### Regime A: sequential ignorability

Assume

\[
A_t \perp (U(a_{0:T})) \mid H_t.
\]

The behavior policy depends only on observed history. Exact or estimated propensities can support OPE and policy learning.

#### Regime B: randomized exploration

Some actions are randomly exposed with known probabilities. This is the strongest real-data bridge; KuaiRand supplies a form of randomized exposure in a non-language domain.

#### Regime C: hidden confounding

The behavior policy uses hidden state:

\[
A_t\sim\mu(a_t\mid H_t,Z_t),
\]

but the released learner sees only \(H_t\). Point identification generally requires additional assumptions, instruments, proxies, or sensitivity analysis. In this regime the project should report bounds or degradation, not pretend exact propensities solve the problem.

---

## 4. Research questions and hypotheses

### RQ1 — Can delayed future feedback improve reward modeling?

**Hypothesis H1:** A model trained on future multi-turn user behavior will improve calibration and ranking of response quality relative to an immediate-feedback-only model, particularly for corrections, incomplete answers, and superficially positive continuations.

### RQ2 — Does outcome prediction imply correct temporal credit?

**Hypothesis H2:** Models with similar trajectory AUC/Brier score will differ substantially in per-step interventional-credit correlation.

### RQ3 — Does joint training improve credit recovery?

**Hypothesis H3:** Joint outcome, redistribution, and intervention-value supervision will outperform outcome-only prediction and RUDDER-style redistribution on oracle credit.

### RQ4 — Does uncertainty predict reward error under shift?

**Hypothesis H4:** Deep ensembles or bootstrap heads will have higher error-detection AUROC than a single model’s softmax entropy when evaluated on unseen structural worlds, policy shifts, or user segments.

### RQ5 — Can pessimistic optimization delay Goodhart failure?

**Hypothesis H5:** Optimizing a lower-confidence reward

\[
\hat R_{\text{LCB}}=\bar R-\lambda\sigma_R
\]

will achieve lower peak proxy reward but higher true utility at the selected checkpoint and a smaller hacking gap.

### RQ6 — Do real logs and structural worlds support the same representations?

**Hypothesis H6:** A model pretrained on authentic conversation trajectories and calibrated on structural worlds will transfer better than a model trained only on synthetic text, but causal credit must still be learned from intervention-rich data.

---

## 5. Data strategy

### 5.1 Data-source roles

| Source | Domain | Main role | Causal status | Recommended use |
|---|---|---|---|---|
| WildChat-1M | real human–LLM conversations | primary authentic-language source | observational; unknown behavior propensities | construct delayed future-user feedback labels and train text encoders |
| WildFB / WildReward | curated in-the-wild feedback | immediate/near-turn reward baseline and teacher labels | observational | bootstrap feedback taxonomy; compare immediate vs delayed outcome modeling |
| LMSYS-Chat-1M | real multi-model conversations | replication and model-shift test | observational, gated | secondary replication after pipeline stabilizes |
| KuaiRand | real sequential recommendation logs | randomized-exposure bridge | partial randomization | test delayed behavioral credit/OPE on real long sequences |
| Structural world suite | synthetic SCMs | exact intervention and utility evaluation | fully known | primary causal-credit and overoptimization benchmark |

### 5.2 Primary dataset: WildChat-1M

WildChat contains one million user–ChatGPT conversations and more than 2.5 million interaction turns, collected with opt-in and released under ODC-BY. It includes timestamps and rich conversation metadata.

**Why use it**

- authentic user language;
- multi-turn corrections, clarifications, repetition, abandonment, and continuation;
- large enough for train/validation/test splits by time and conversation;
- compatible with the WildFeedback and WildReward research line.

**Limitations**

- many conversations are short;
- assistant actions come from proprietary models with unknown propensities;
- user follow-up is a noisy proxy, not ground-truth utility;
- absence of a follow-up can mean success, abandonment, or external interruption;
- causal action effects are not identifiable from the log alone;
- safety and privacy filtering are required.

**Initial inclusion criteria**

- at least three assistant turns for trajectory-credit experiments;
- English-only for v0.1, with language expansion later;
- no moderation flags in the initial safe subset;
- no obvious PII after source redaction plus local filtering;
- valid timestamp ordering and role alternation;
- response and future user message under configurable token limits.

**Splits**

1. Use conversation-level grouping; never split turns from one conversation across partitions.
2. Prefer chronological splits if timestamps permit:
   - train: earliest 80%;
   - validation: next 10%;
   - test: latest 10%.
3. If a stable user identifier is available and permitted, group by anonymized user before time splitting. Otherwise explicitly state that user leakage cannot be ruled out.
4. Create a model-shift split by assistant-model metadata if available.

### 5.3 WildFB / WildReward

WildReward derives 186K high-quality feedback instances from WildChat and trains an ordinal reward model using five feedback levels. Use it as:

- an immediate-feedback baseline;
- a source of feedback category definitions;
- a teacher for weak-label bootstrapping;
- a comparison point for calibration and cross-sample consistency.

Do not simply reproduce WildReward. The differentiator is future multi-turn outcome construction, temporal credit, intervention-grounded evaluation, and optimization under uncertainty.

### 5.4 LMSYS-Chat-1M

LMSYS-Chat-1M contains one million real-world conversations involving multiple LLMs and is gated behind a dataset agreement.

Use it only after the WildChat pipeline is stable. Its purposes are:

- replication across a different collection interface;
- evaluation across model identities;
- robustness to a shorter average conversation length;
- testing whether labelers overfit WildChat-specific language.

Do not make it a critical-path dependency because access is gated and many trajectories are short.

### 5.5 KuaiRand

KuaiRand is a real sequential recommendation dataset containing millions of interactions, rich user/item features, 12 feedback signals, and randomized video exposures inserted into production feeds.

Use it as an optional but valuable bridge:

- define actions as exposed items or item clusters;
- define delayed outcomes such as next-session return, sustained watch time, or future positive actions;
- use randomized exposures for more defensible intervention evaluation;
- test OPE and policy learning outside language while preserving the delayed behavioral structure.

The method should operate on a generic event encoder so that text and recommendation events share the same trajectory-level API.

### 5.6 Data licensing and derived artifacts

Every adapter must emit a `SourceManifest` containing:

```yaml
source_name: WildChat-1M
source_version: <commit-or-release>
source_license: ODC-BY-1.0
source_url: <canonical dataset page>
derivative_license: <chosen compatible license>
redistribute_raw_text: false
required_attribution: true
pii_filter_version: <hash>
labeler_version: <hash>
```

Prefer publishing:

- processing code;
- row IDs or hashes;
- derived labels when license-compatible;
- aggregate statistics;
- small sanitized examples.

Avoid republishing source text unless clearly permitted.

---

## 6. Delayed outcome construction for conversational logs

### 6.1 Unit of analysis

For assistant turn \(t\), define:

- context \(H_t\): all messages through user request \(u_t\);
- action \(A_t\): assistant response \(r_t\);
- future window \(W_{t,K}\): the next \(K\) user and assistant turns;
- delayed outcome \(Y_{t,K}\): a label computed only from future observations after \(A_t\).

Use multiple horizons, for example \(K\in\{1,2,4,\text{end}\}\). This converts “delay” into an experimental variable.

### 6.2 Feedback event taxonomy

Create a future-user event classifier with mutually non-exclusive labels:

- `explicit_positive`: thanks, approval, confirmation of success;
- `explicit_negative`: direct rejection or complaint;
- `correction`: user states that content is wrong;
- `constraint_restatement`: user repeats an unmet requirement;
- `request_repetition`: semantically repeats the original request;
- `clarification_due_to_failure`: asks what the assistant should already have made clear;
- `productive_followup`: advances the task without correcting the answer;
- `task_completion_signal`: says the task worked or is done;
- `abandonment_proxy`: conversation ends after a likely unresolved response;
- `neutral_continuation`;
- `unsafe_escalation` or policy-risk signal.

The labeler should use the prior user request, assistant response, and future user message. For a stricter anti-leakage variant, train a response-quality model from the future-user message but evaluate whether it can score the response without future text at inference.

### 6.3 Outcome definitions

#### Ordinal future satisfaction proxy

Let \(s_j\in\{-2,-1,0,1,2\}\) be the score for future event \(j\). A horizon-weighted label is

\[
Y_{t,K}^{\text{ord}}
=
\operatorname{clip}
\left(
\sum_{j=1}^{K} \alpha^{j-1}s_{t+j},
-2,2
\right).
\]

Use \(\alpha\in[0.7,1]\) and report sensitivity.

#### Binary failure-within-horizon

\[
Y_{t,K}^{\text{fail}}
=
\mathbf 1[
\exists j\le K:
\text{correction}_j\lor
\text{constraint-restatement}_j\lor
\text{explicit-negative}_j
].
\]

#### Task-progress proxy

\[
Y_{t,K}^{\text{progress}}
=
\mathbf 1[
\text{productive-followup or completion occurs before failure}
].
\]

#### Conversation-level outcome

For a full conversation, define a terminal ordinal outcome from the last observed user feedback and unresolved-request classifier. Mark cases with insufficient evidence as `unknown`, not neutral.

### 6.4 Censoring and ambiguity

A conversation ending is not automatically negative. Implement a three-way status:

- `observed_positive_or_negative`;
- `right_censored`;
- `uninformative/ambiguous`.

For fixed-horizon tasks, only include samples with enough future turns, or train a survival/competing-risk objective. A simple first implementation may use complete cases and then add inverse-probability-of-censoring weights:

\[
w_i^{\text{IPCW}}
=
\frac{\delta_i}{\hat G(\tilde T_i\mid H_i)},
\]

where \(\hat G\) estimates the probability of remaining observed.

### 6.5 Label generation pipeline

1. Deterministic regex/high-precision rules for explicit feedback.
2. Embedding similarity for repeated requests and constraint restatements.
3. LLM classifier for ambiguous feedback categories.
4. Agreement filtering between rule and LLM signals.
5. Human annotation of a stratified sample.
6. Calibrate class probabilities using the human sample.
7. Store the complete label provenance, including prompt, model, temperature, and version.

### 6.6 Human validation set

Create at least 1,000 manually labeled examples, stratified by:

- outcome class;
- delay horizon;
- conversation length;
- domain;
- labeler confidence;
- assistant model;
- explicit versus implicit feedback.

Use two annotators on at least 20% and adjudicate disagreements. Report Cohen’s kappa or Krippendorff’s alpha. The public paper should clearly separate labeler agreement from downstream model performance.

### 6.7 Leakage controls

- The reward model input for scoring action \(A_t\) must not include future user turns.
- The label-generation model may use future user turns, but its output must be frozen before train/test modeling.
- Split before generating model-assisted labels if the labeler itself is fine-tuned.
- Remove exact duplicate conversations and near-duplicate prompts across splits.
- Include trivial baselines using conversation length, response length, and termination to expose leakage.

---

## 7. Controlled structural environment suite

The suite should contain distinct causal structures, not merely parameter variations of one model. Each world implements a shared API but different state-transition equations.

### 7.1 Common structural causal model

For each episode:

\[
Z_0=f_0(\epsilon_0),
\]

\[
A_t=f_\mu(H_t,Z_t,\epsilon_t^A),
\]

\[
X_t=f_X(Z_t,A_t,\epsilon_t^X),
\]

\[
Z_{t+1}=f_Z(Z_t,A_t,X_t,\epsilon_t^Z),
\]

\[
Y=f_Y(Z_{0:T},A_{0:T},X_{0:T},\epsilon^Y),
\qquad
U=f_U(Z_{0:T},A_{0:T},X_{0:T}).
\]

Every exogenous noise variable must be explicitly seeded. Interventions replace the structural equation for \(A_t\) while keeping shared exogenous noise fixed for paired variance reduction.

### 7.2 World A: fatigue and habit

Purpose: reproduce the useful portion of the original lifecycle-messaging idea.

Latent state:

- topic interest \(\theta\);
- base responsiveness \(\beta\);
- fatigue sensitivity \(\lambda\);
- habit \(h_t\);
- fatigue \(f_t\);
- category habituation \(c_t[a]\).

Dynamics:

\[
f_{t+1}=(1-\delta_f)f_t+\lambda\mathbf 1[a_t\ne\text{none}],
\]

\[
c_{t+1}[a]=(1-\delta_c)c_t[a]+\mathbf 1[a_t=a],
\]

\[
p(e_t=1)=
\sigma(b_0+w_\theta\theta[a_t]-w_ff_t-w_cc_t[a_t]+\beta),
\]

\[
h_{t+1}=(1-\delta_h)h_t+\gamma e_t.
\]

Behavioral proxy:

\[
Y=\mathbf 1[h_T>\kappa].
\]

True utility:

\[
U=h_T-\eta_f\sum_t f_t-\eta_u\mathbf 1[\text{unsubscribe}].
\]

This separation allows a policy to raise retention while creating excessive fatigue.

### 7.3 World B: hidden intent and exogenous state shifts

Purpose: prevent methods from assuming all outcome changes are caused by actions.

Latent intent follows a Markov process:

\[
z_{t+1}\sim P(z_{t+1}\mid z_t,\epsilon_t^{\text{exo}}).
\]

Action success depends on intent-action match:

\[
p(x_t=1)=\sigma(\theta_{z_t,a_t}+b_t).
\]

The terminal outcome is driven by cumulative matched progress and an exogenous shock:

\[
Y=\mathbf 1\left[
\sum_t x_t+\omega\epsilon^{\text{shock}}>\kappa
\right].
\]

The logging policy may observe a noisy privileged signal of \(z_t\) unavailable to the learner, creating hidden confounding.

### 7.4 World C: delayed conversion with competing causes

Purpose: model an outcome that can occur long after an action and may be caused by multiple earlier interventions.

Each action creates a latent conversion impulse with random delay:

\[
d_t\sim p_D(\cdot\mid a_t,z_t),
\qquad
q_t\sim p_Q(\cdot\mid a_t,z_t).
\]

The hazard at future time \(s\) is

\[
\lambda_s
=
\lambda_0(s)+
\sum_{t<s}q_tK(s-t-d_t),
\]

where \(K\) is a delay kernel. Conversion time is sampled from the resulting hazard. Actions can interfere or saturate:

\[
q_t^{\text{effective}}
=
q_t\exp\left(-\rho\sum_{j<t}\mathbf 1[a_j\ne\text{none}]\right).
\]

This world tests whether a method can assign credit across long and variable delays.

### 7.5 World D: proxy–utility divergence

Purpose: create a genuine Goodhart setting.

State includes:

- useful progress \(g_t\);
- engagement/arousal \(e_t\);
- trust \(q_t\);
- dependency or annoyance \(d_t\).

Actions include helpful, neutral, urgent, flattering, fear-inducing, and no-op variants. For v0.1 these may be discrete rather than generated text.

Example dynamics:

\[
g_{t+1}=g_t+\Delta_g(a_t,z_t)+\epsilon_t^g,
\]

\[
e_{t+1}=\rho_e e_t+\Delta_e(a_t,z_t)+\epsilon_t^e,
\]

\[
q_{t+1}=\rho_q q_t+\Delta_q(a_t,z_t),
\]

\[
d_{t+1}=\rho_d d_t+\Delta_d(a_t,z_t).
\]

Observed proxy:

\[
Y=\alpha_e e_T+\alpha_r\mathbf 1[\text{return}].
\]

True utility:

\[
U=\beta_g g_T+\beta_q q_T-\beta_d d_T-\beta_f\sum_t\text{interruption-cost}_t.
\]

Some actions should reliably raise \(Y\) while lowering \(U\). Otherwise the overoptimization experiment is not meaningful.

### 7.6 Difficulty factors

Each world exposes orthogonal switches:

```yaml
horizon: [8, 32, 128]
outcome_stochasticity: [low, medium, high]
observability: [oracle, noisy, partial]
logging:
  regime: [clean, randomized, hidden_confounding]
  support: [broad, narrow]
outcome_delay:
  type: [fixed, geometric, heavy_tail]
proxy_utility_gap: [zero, moderate, severe]
nonstationarity: [off, segment_shift, temporal_shift]
```

### 7.7 Behavior policies

Include:

- random/exploratory policy;
- sensible heuristic;
- high-proxy but low-utility policy;
- segment-aware policy;
- privileged-state logging policy;
- mixture policy with version changes over time.

Log exact action probabilities only when the policy is defined on released observations. When privileged state is used, separately record:

- `propensity_full = μ(a|h,z)` for oracle analysis;
- `propensity_observed = P(a|h)` if computable;
- a flag that hidden confounding is present.

### 7.8 Oracle counterfactual evaluator

For history \(h_t\), action \(a\), reference \(a_0\), future policy \(\pi_f\), and \(M\) paired rollouts:

\[
\hat C_t^{\pi_f}(a;a_0)
=
\frac{1}{M}\sum_{m=1}^{M}
\left[
U^{(m)}(do(A_t=a),\pi_f)
-
U^{(m)}(do(A_t=a_0),\pi_f)
\right].
\]

Use common random numbers for both terms. Increase \(M\) until the Monte Carlo standard error is below a configured threshold. Store the standard error with each oracle label.

---

## 8. Canonical data schema

### 8.1 Event record

```python
@dataclass(frozen=True)
class Event:
    trajectory_id: str
    event_id: str
    event_time: datetime
    step_index: int
    event_type: Literal[
        "observation", "action", "user_response", "system_event", "outcome"
    ]
    payload: dict[str, Any]
    source: str
    source_row_id: str | None
    policy_id: str | None
    policy_version: str | None
    schema_version: str
```

### 8.2 Trajectory record

```python
@dataclass
class Trajectory:
    trajectory_id: str
    entity_key_hash: str | None
    events: list[Event]
    start_time: datetime
    end_time: datetime | None
    behavior_policy_id: str | None
    observation_regime: str
    censoring_status: str
    metadata: dict[str, Any]
```

### 8.3 Training example

```python
@dataclass
class DelayedOutcomeExample:
    trajectory_id: str
    prefix_end_step: int
    observations: Any
    actions: Any
    responses: Any
    terminal_outcome: float | int | None
    outcome_type: str
    outcome_observed_at: datetime | None
    censored: bool
    behavior_logprobs: list[float] | None
    propensity_quality: Literal["exact", "estimated", "unknown", "confounded"]
    sample_weight: float
    provenance: dict[str, Any]
```

### 8.4 Oracle credit example

```python
@dataclass
class OracleCreditExample:
    trajectory_id: str
    step_index: int
    action: int
    reference_action: int
    future_policy_id: str
    continuation_mode: Literal["frozen", "policy_reactive"]
    credit_utility: float
    credit_proxy: float
    monte_carlo_se: float
```

Use Parquet for processed data, Arrow for batch interchange, and JSON/YAML only for small manifests and configuration.

---

## 9. Baseline methods

### 9.1 Trivial and diagnostic baselines

These detect label leakage and task degeneracy.

- outcome prior by dataset/domain;
- conversation length only;
- assistant response length only;
- last-turn feedback only;
- uniform credit \(Y/(T+1)\);
- all credit to final action;
- recency-decayed credit;
- attention weights as attribution, explicitly labeled non-causal.

### 9.2 Outcome-model baselines

- logistic regression on handcrafted trajectory statistics;
- GRU/LSTM outcome classifier;
- causal Transformer outcome classifier;
- pretrained text encoder plus temporal pooling;
- ordinal regression matching WildReward-style labels.

For binary outcomes:

\[
\mathcal L_{\text{BCE}}
=-y\log p-(1-y)\log(1-p).
\]

For ordinal labels, use cumulative link/CORAL-style losses with ordered thresholds.

### 9.3 Return redistribution baselines

#### RUDDER-style predictor differences

Train a sequence model to predict terminal return from prefixes, then redistribute:

\[
r_t^{\text{RUDDER}}=
\hat V(h_{t+1})-\hat V(h_t).
\]

#### Likelihood or contribution decomposition

Include one modern return-decomposition baseline if implementation cost is reasonable. It should remain secondary to a correct RUDDER implementation.

### 9.4 Offline RL baselines

- behavior cloning;
- filtered BC as a naive outcome-conditioned baseline;
- recurrent IQL;
- recurrent CQL;
- Decision Transformer;
- model-based planning with learned dynamics/outcome model.

IQL core objectives:

\[
\mathcal L_V(\psi)
=
\mathbb E_{(s,a)\sim\mathcal D}
\left[
L_2^\tau(Q_\phi(s,a)-V_\psi(s))
\right],
\]

where \(L_2^\tau(u)=|\tau-\mathbf 1[u<0]|u^2\).

\[
\mathcal L_Q(\phi)
=
\mathbb E
\left[
\left(r+\gamma V_{\bar\psi}(s')-Q_\phi(s,a)\right)^2
\right].
\]

Policy extraction uses advantage-weighted cloning:

\[
\mathcal L_\pi
=-\mathbb E_{(s,a)\sim\mathcal D}
\left[
\exp(\beta(Q-V))\log\pi(a\mid s)
\right].
\]

### 9.5 Off-policy evaluation baselines

For clean/randomized settings:

- direct method;
- trajectory importance sampling;
- per-decision importance sampling;
- weighted importance sampling;
- doubly robust OPE;
- fitted Q evaluation.

Sequential doubly robust estimator:

\[
\hat V_{DR}
=
\hat V(s_0)
+
\sum_{t=0}^{T}
\rho_{0:t}
\left[
 r_t+\gamma\hat V(s_{t+1})-\hat Q(s_t,a_t)
\right],
\]

with

\[
\rho_{0:t}
=
\prod_{j=0}^{t}
\frac{\pi(a_j\mid s_j)}{\mu(a_j\mid s_j)}.
\]

Clip or normalize ratios and report effective sample size. Do not use importance-weighted point estimates in hidden-confounding settings without caveats.

### 9.6 Naive outcome-labeled DPO

If included, label it explicitly as a failure baseline. Pairing successful and unsuccessful trajectories does not imply that every action in the successful trajectory is preferable.

The ordinary DPO objective is

\[
-\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right).
\]

Using terminal labels to create stepwise preferences is intentionally suspect and should be evaluated as such.

---

## 10. Proposed method: Delayed Outcome Credit Model (DOCM)

### 10.1 Design goals

DOCM should:

- operate on variable-length multimodal event sequences;
- separate outcome prediction from causal credit;
- use intervention labels when available without requiring them for every sample;
- support censored outcomes;
- quantify epistemic uncertainty;
- produce rewards that can be consumed by offline or delayed-online RL;
- expose failure under distribution shift rather than hiding it.

### 10.2 Representation

Encode each step into

\[
e_t = E_o(o_t)+E_a(a_t)+E_x(x_t)+E_\Delta(\Delta t_t)+E_m(m_t),
\]

where \(E_m\) contains source/domain and observability metadata. For text, use one of:

- frozen small language-model embeddings;
- LoRA-tuned 0.5B–3B causal LM;
- sentence encoder for the MVP.

Pass step embeddings through a causal Transformer:

\[
h_0,\ldots,h_T=\operatorname{CausalTransformer}(e_0,\ldots,e_T).
\]

### 10.3 Heads

#### Outcome head

\[
\hat p_Y = \sigma(w_Y^\top h_T).
\]

For ordinal outcomes, predict ordered cumulative logits.

#### Prefix value head

\[
\hat V_t = \sigma(w_V^\top h_t).
\]

Redistributed predictive reward:

\[
\hat r_t^{pred}=\hat V_{t+1}-\hat V_t.
\]

#### Action-value head

For discrete actions:

\[
\hat Q_t(a)=f_Q(h_t,E_a(a)).
\]

For language responses, encode a candidate response \(r\) and score

\[
\hat Q_t(r)=f_Q(h_t,E_{text}(r)).
\]

Estimated credit:

\[
\hat C_t(a;a_0)=\hat Q_t(a)-\hat Q_t(a_0).
\]

#### Hazard head

For delayed event occurrence:

\[
\hat \lambda_t=
P(T^Y=t\mid T^Y\ge t,h_t).
\]

The discrete survival likelihood is

\[
\mathcal L_{surv}
=-\sum_i\left[
\delta_i\log\hat\lambda_{T_i}
+
\sum_{t<T_i}\log(1-\hat\lambda_t)
\right].
\]

#### Uncertainty head

Preferred MVP: train \(K=5\) bootstrap ensemble members. For prediction \(g_k\):

\[
\bar g=\frac1K\sum_k g_k,
\qquad
\sigma_g^2=\frac1{K-1}\sum_k(g_k-\bar g)^2.
\]

Alternative stretch: shared encoder with independent bootstrap heads; evidential methods should not replace ensembles until validated.

### 10.4 Loss function

Use source-dependent masks:

\[
\mathcal L
=
\lambda_Y\mathcal L_{outcome}
+
\lambda_V\mathcal L_{prefix}
+
\lambda_{tel}\mathcal L_{telescoping}
+
\lambda_C\mathcal L_{credit}
+
\lambda_Q\mathcal L_{Q-consistency}
+
\lambda_{surv}\mathcal L_{survival}
+
\lambda_{cal}\mathcal L_{calibration}.
\]

#### Outcome loss

Binary cross-entropy, ordinal loss, or negative log likelihood depending on outcome type.

#### Prefix loss

Train prefix values against the terminal outcome with time-dependent weighting:

\[
\mathcal L_{prefix}
=
\sum_t w_t\operatorname{BCE}(\hat V_t,Y).
\]

To avoid the final prefix dominating, normalize \(\sum_t w_t=1\) and compare uniform versus delay-aware weights.

#### Telescoping consistency

\[
\mathcal L_{tel}
=
\left(
\hat V_0+
\sum_{t=0}^{T}\hat r_t^{pred}
-
\hat V_{T+1}
\right)^2.
\]

If \(\hat r_t\) is produced by a separate head, use

\[
\left(
\hat V_0+
\sum_t\hat r_t-
\hat p_Y
\right)^2.
\]

#### Oracle credit loss

On structural-world or randomized intervention samples:

\[
\mathcal L_{credit}
=
\sum_t
\frac{(
\hat C_t-C_t^{oracle}
)^2}{\sigma_{oracle,t}^2+\epsilon}.
\]

#### Pairwise action ranking

For actions \(a^+,a^-\) with oracle value difference:

\[
\mathcal L_{rank}
=-\log\sigma(\hat Q(h,a^+)-\hat Q(h,a^-)).
\]

#### Bellman or continuation consistency

For simulator data:

\[
\mathcal L_{Q-consistency}
=
\left[
\hat Q(h_t,a_t)-
\left(r_t^{true}+\gamma\mathbb E_{a'\sim\pi_f}\hat Q(h_{t+1},a')\right)
\right]^2.
\]

Because real logs may have only terminal outcomes, this term is masked there.

### 10.5 Multi-source training

A batch may combine:

- real conversation samples: outcome and prefix losses;
- WildFB: immediate ordinal feedback loss;
- structural world trajectories: all losses including oracle credit;
- KuaiRand randomized samples: outcome, propensity-aware, and partial intervention losses.

Use source-balanced sampling to prevent the largest dataset from dominating. Record per-source gradient norms and validation metrics.

### 10.6 Domain adaptation

Do not assume simulator numeric states and real text share identical representations. Use:

- source-specific input adapters;
- shared temporal backbone;
- source-specific outcome heads where label semantics differ;
- a shared credit head only when action representations are comparable;
- optional domain-adversarial regularization as a stretch, not MVP.

The strongest initial design may train the architecture separately in each domain while sharing method and evaluation, rather than forcing one universal checkpoint.

### 10.7 Pessimistic reward

For policy learning, define

\[
\hat r_t^{LCB}
=
\bar r_t-\lambda\sigma_{r,t},
\]

or trajectory-level

\[
\hat R^{LCB}(\tau)
=
\bar R(\tau)-\lambda\sigma_R(\tau)-\eta d(\tau,\mathcal D),
\]

where \(d(\tau,\mathcal D)\) is an out-of-distribution score such as nearest-neighbor distance or density-ratio estimate. Tune \(\lambda\) and \(\eta\) on a held-out family of structural worlds, not the test world.

### 10.8 Method variants for ablation

- `DOCM-outcome`: terminal head only.
- `DOCM-prefix`: outcome + prefix values.
- `DOCM-credit`: add oracle credit supervision.
- `DOCM-ensemble`: add uncertainty.
- `DOCM-LCB`: use pessimistic policy objective.
- `DOCM-survival`: add censoring-aware hazard objective.

---

## 11. Policy-learning experiments

### 11.1 Discrete controlled-policy track

Train policies over small action spaces using:

- BC;
- recurrent IQL;
- CQL;
- PPO/GRPO against the learned reward;
- DOCM-LCB optimization.

Use the same policy architecture for fair comparison where possible.

### 11.2 LLM-native reranking track

A practical LLM-native first step is candidate reranking rather than full free-form generation.

For each real or synthetic context:

1. sample \(M\) candidate responses from a frozen policy model;
2. score candidates using the reward/credit model;
3. select with softmax or LCB ranking;
4. evaluate using held-out behavioral labels on real logs and true utility in a text-conditioned simulator or judge-assisted controlled set.

Selection policy:

\[
\pi(a\mid h)
\propto
\pi_0(a\mid h)
\exp\left(
\frac{\hat Q^{LCB}(h,a)}{\beta}
\right).
\]

This is more defensible than using a language model merely to emit one of four category tokens.

### 11.3 Full post-training track

After reranking works, use PPO, GRPO, RLOO, or online DPO against the learned reward. Constrain policy drift with KL:

\[
\max_\pi
\mathbb E_{a\sim\pi}[\hat R(a)]
-
\beta D_{KL}(\pi\|\pi_{ref}).
\]

For delayed group outcomes, plain GRPO may have high variance. Compare:

- terminal raw outcome;
- redistributed DOCM reward;
- value-baseline advantage;
- group-relative advantage;
- LCB reward;
- larger rollout groups.

### 11.4 Stopping and checkpoint selection

Never select checkpoints by maximum proxy reward alone. Report:

- proxy-optimal checkpoint;
- validation-outcome-optimal checkpoint;
- true-utility-optimal checkpoint in simulation;
- uncertainty-constrained checkpoint;
- KL-matched comparisons.

Define hacking gap:

\[
G_{hack}
=
\max_k J_U(\pi_k)
-
J_U(\pi_{k^*_{proxy}}),
\]

where

\[
k^*_{proxy}=\arg\max_k J_{\hat R}(\pi_k).
\]

Lower is better.

---

## 12. Experiment matrix

### Experiment E0 — Pipeline sanity

**Goal:** prove schemas, labels, models, and evaluation are wired correctly.

- tiny deterministic world;
- known additive action effects;
- exact outcome;
- model must recover credit correlation >0.99;
- one-command run under five minutes.

### Experiment E1 — Real-log delayed outcome prediction

**Datasets:** WildChat primary; WildFB immediate baseline.

**Tasks:** predict future correction, repetition, productive continuation, and ordinal outcome at horizons 1, 2, 4, and end.

**Baselines:** priors, length-only, immediate-feedback model, GRU, Transformer, DOCM variants.

**Metrics:**

- AUROC and AUPRC;
- macro-F1 for event classes;
- Brier score;
- expected calibration error;
- ordinal MAE / quadratic weighted kappa;
- subgroup metrics by domain, length, and model identity.

**Key test:** whether future-window training improves held-out label prediction beyond immediate user reaction.

### Experiment E2 — Oracle credit recovery

**Worlds:** A–D.

**Metrics:**

- Spearman and Pearson correlation with oracle credit;
- sign accuracy: \(\operatorname{sign}(\hat C)=\operatorname{sign}(C)\);
- top-action accuracy;
- calibration of predicted credit intervals;
- normalized RMSE;
- performance by temporal distance to outcome.

**Key test:** similar outcome AUC but different credit accuracy.

### Experiment E3 — Offline policy improvement

Generate logged datasets with broad/narrow support and clean/hidden logging.

**Metrics:**

- true utility \(J_U\);
- behavioral proxy \(J_Y\);
- regret to oracle policy;
- unsubscribe/negative-event rate where applicable;
- OPE error;
- effective sample size;
- policy divergence from behavior.

### Experiment E4 — Structural generalization

Train on three world families and test on the fourth, plus parameter shifts within families.

**Metrics:**

- outcome degradation;
- credit degradation;
- uncertainty–error correlation;
- error-detection AUROC using uncertainty;
- policy utility under shift.

### Experiment E5 — Reward overoptimization

Optimize policies for increasing update budgets against a fixed learned reward model.

Plot against optimization step:

- learned proxy reward;
- observed behavioral proxy;
- true utility;
- uncertainty;
- KL to reference;
- fraction of out-of-support actions;
- qualitative action distribution.

Compare single RM, ensemble mean, ensemble LCB, KL regularization, and periodic RM refresh.

### Experiment E6 — KuaiRand bridge

Define a delayed next-session or future-window outcome using randomized exposures.

Compare:

- outcome prediction;
- credit estimates on randomized positions;
- OPE accuracy;
- policy ranking stability.

This experiment is optional for the MVP but valuable before paper submission.

### Experiment E7 — LLM reranking/post-training

Use a 0.5B–3B open model with LoRA.

Compare:

- SFT/base policy;
- immediate-feedback RM;
- delayed-outcome RM;
- DOCM mean reward;
- DOCM LCB reward.

Report compute, candidate count, tokens, wall-clock, and memory.

---

## 13. Metrics and statistical protocol

### 13.1 Outcome metrics

- negative log likelihood;
- Brier score;
- AUROC/AUPRC;
- ECE and reliability diagrams;
- calibration slope/intercept;
- ordinal ranking accuracy.

### 13.2 Credit metrics

- Spearman correlation;
- Pearson correlation;
- Kendall’s tau;
- sign accuracy;
- action-ranking regret;
- temporal localization error;
- interval coverage.

### 13.3 Policy metrics

- true utility;
- proxy outcome;
- regret;
- constraint violations;
- action frequency/volume;
- behavior-policy KL;
- OPE absolute and relative error.

### 13.4 Overoptimization metrics

- hacking gap;
- update budget to divergence;
- utility at proxy-optimal checkpoint;
- maximum true utility achieved;
- area between normalized proxy and utility curves;
- uncertainty at divergence.

### 13.5 Statistical tests

- at least five seeds for small-model experiments; three only for expensive LLM runs;
- paired evaluation seeds across policies;
- bootstrap confidence intervals over trajectories or users;
- hierarchical bootstrap if multiple turns per conversation are evaluated;
- correct for multiple primary comparisons or predeclare one primary metric;
- report effect sizes, not only p-values.

---

## 14. Implementation plan

### 14.1 Technology choices

- Python 3.11+
- PyTorch
- Hugging Face Transformers and Datasets
- PyArrow/Parquet
- DuckDB for local analytics
- Hydra or Pydantic settings for config
- Gymnasium-style controlled environments
- `d3rlpy` or minimal custom offline-RL baselines
- TRL for the first LLM post-training adapter
- `verl` integration deferred to Phase 2 or the final Phase 1 demonstration
- pytest + Hypothesis for property tests
- Weights & Biases or MLflow optional, with local JSONL as the reproducible source of truth

### 14.2 Phase 1 package layout

```text
src/longfeedback/
├── schema/
│   ├── event.py
│   ├── trajectory.py
│   └── outcome.py
├── data/
│   ├── wildchat.py
│   ├── wildfb.py
│   ├── lmsys.py
│   ├── kuairand.py
│   ├── filters.py
│   └── splits.py
├── outcomes/
│   ├── taxonomy.py
│   ├── rules.py
│   ├── llm_labeler.py
│   ├── censoring.py
│   └── validation.py
├── worlds/
│   ├── base.py
│   ├── fatigue_habit.py
│   ├── hidden_intent.py
│   ├── delayed_conversion.py
│   └── proxy_utility.py
├── models/
│   ├── encoders.py
│   ├── outcome.py
│   ├── docm.py
│   └── ensemble.py
├── credit/
│   ├── oracle.py
│   ├── rudder.py
│   ├── attributions.py
│   └── metrics.py
├── policies/
│   ├── behavior.py
│   ├── bc.py
│   ├── iql.py
│   ├── cql.py
│   └── reranker.py
├── ope/
│   ├── importance_sampling.py
│   ├── doubly_robust.py
│   └── diagnostics.py
└── evaluation/
    ├── outcomes.py
    ├── policies.py
    ├── overoptimization.py
    └── reporting.py
```

### 14.3 Commands

```bash
# Download or validate source access
longfeedback data prepare wildchat --config configs/data/wildchat.yaml

# Generate labels
longfeedback outcomes label --source wildchat --horizon 4

# Build controlled logs and oracle credits
longfeedback worlds generate --world fatigue_habit --episodes 100000
longfeedback worlds credit --world fatigue_habit --samples 20000 --mc-rollouts 128

# Train models
longfeedback train outcome --config configs/models/transformer.yaml
longfeedback train docm --config configs/models/docm.yaml

# Train/evaluate a policy
longfeedback policy train --algorithm iql --reward docm_lcb
longfeedback evaluate policy --checkpoint <path> --world-suite all

# Produce report
longfeedback report phase1 --experiment-group paper_v0
```

### 14.4 Configuration example

```yaml
experiment:
  name: docm_worldA_seed0
  seed: 0

data:
  sources:
    - name: structural/fatigue_habit
      weight: 0.7
    - name: wildchat/delayed_feedback_h4
      weight: 0.3
  max_steps: 64
model:
  encoder: causal_transformer
  d_model: 384
  layers: 6
  heads: 6
  dropout: 0.1
  ensemble_members: 5
loss:
  outcome: 1.0
  prefix: 0.5
  telescoping: 0.2
  credit: 1.0
  survival: 0.2
optimizer:
  name: adamw
  lr: 3.0e-4
  weight_decay: 0.01
training:
  batch_size: 256
  steps: 100000
  grad_clip: 1.0
  eval_every: 1000
```

---

## 15. Testing plan

### 15.1 Unit tests

- event ordering and timezone normalization;
- trajectory prefix construction;
- censoring labels;
- propensity normalization;
- reward telescoping identity;
- survival likelihood on hand-computed cases;
- action-value difference equals credit output;
- ensemble mean/variance calculation;
- seed determinism.

### 15.2 Property-based tests

- no future event appears in a model input prefix;
- sum of behavior probabilities equals one;
- intervention evaluator changes only the intervened structural equation;
- paired rollouts reuse exogenous noise;
- deterministic worlds produce zero Monte Carlo variance;
- no-op has zero local effect when all action mechanisms are disabled;
- stronger fatigue penalty cannot increase utility for identical trajectory.

### 15.3 Integration tests

- raw dataset row → sanitized event → labeled trajectory → training batch;
- generated world → oracle credits → DOCM training → metric report;
- reward checkpoint → policy training → true utility evaluation;
- interrupted experiment resumes with identical results.

### 15.4 Reproducibility tests

- frozen small fixture datasets in repository;
- expected metric ranges, not exact floating-point values;
- environment/config hashes embedded in checkpoints;
- CI CPU smoke test;
- optional nightly GPU test.

---

## 16. Eight-week execution plan

### Week 1 — schemas and deterministic world

- implement canonical event/trajectory schema;
- implement World A in deterministic mode;
- implement intervention evaluator;
- implement trivial baselines and outcome GRU;
- verify oracle credit analytically.

**Exit:** E0 passes.

### Week 2 — stochastic worlds and RUDDER

- add stochasticity and partial observability;
- implement World B;
- implement RUDDER-style redistribution;
- implement credit metrics;
- produce first outcome-accuracy-versus-credit plot.

**Exit:** evidence that outcome accuracy and credit accuracy are distinct.

### Week 3 — DOCM MVP

- causal Transformer encoder;
- outcome, prefix, and action-value heads;
- joint loss;
- train on Worlds A/B;
- compare with RUDDER and outcome-only models.

**Exit:** Gate A.

### Week 4 — WildChat pipeline

- source adapter and license manifest;
- safe filtering;
- future-feedback taxonomy;
- high-precision rules plus LLM labeling;
- 1,000-example human validation plan;
- train real-log outcome baselines.

**Exit:** E1 is above trivial baselines.

### Week 5 — Worlds C/D and uncertainty

- implement delayed-conversion and proxy-utility worlds;
- ensemble training;
- structural leave-one-world-out tests;
- uncertainty/error plots.

### Week 6 — policy optimization

- BC/IQL policy baselines;
- policy optimization against learned reward;
- LCB variant;
- overoptimization curves.

**Exit:** Gate B.

### Week 7 — LLM reranking

- integrate 0.5B–3B model or frozen text encoder;
- sample candidate responses;
- delayed-outcome reranking evaluation;
- compute profiling.

### Week 8 — report and hardening

- rerun primary experiments;
- bootstrap confidence intervals;
- ablations;
- README narrative;
- data/model cards;
- reproducibility command;
- identify Phase 2 abstractions.

---

## 17. Risk register and mitigations

### Risk 1: real conversational outcomes are too noisy

**Signals:** AUPRC near base rate; labeler disagreement; length baselines dominate.

**Mitigation:** narrow to high-precision outcome categories; treat missing feedback as censored; use WildFB immediate labels as auxiliary supervision; avoid claiming general satisfaction.

### Risk 2: DOCM only wins because it has more parameters

**Mitigation:** capacity-matched baselines; shared encoder with different heads; report parameter counts and training compute; ablate each loss.

### Risk 3: simulator conclusions depend on one causal graph

**Mitigation:** four structural families; leave-one-family-out evaluation; distinguish structural robustness from parameter sensitivity.

### Risk 4: causal language overreaches real logs

**Mitigation:** reserve “causal credit” for controlled/randomized tracks; use “predictive contribution” for WildChat; maintain a formal assumptions document.

### Risk 5: policy optimizer trivially exploits obvious simulator bugs

**Mitigation:** treat each exploit as a test failure unless intentionally part of World D; add invariant tests; inspect action/state distributions; run oracle policy sanity checks.

### Risk 6: LLM component looks decorative

**Mitigation:** make language understanding central to the reward/credit model or candidate reranker; do not use an LLM merely as a four-class policy.

### Risk 7: project scope expands into infrastructure too early

**Mitigation:** no distributed services, streaming, or general plugin system before Gate B.

### Risk 8: hidden confounding invalidates OPE

**Mitigation:** explicitly label identification regime; use randomized track; add sensitivity analysis/bounds; do not publish point estimates as causal in confounded settings.

---

## 18. Primary ablations

1. outcome-only vs prefix vs oracle-credit supervision;
2. single model vs ensemble;
3. mean reward vs LCB reward;
4. immediate labels vs delayed labels vs both;
5. fixed horizon vs survival objective;
6. full history vs handcrafted features;
7. clean logging vs randomized logging vs hidden confounding;
8. same-family parameter shift vs unseen structural family;
9. frozen continuation vs policy-reactive counterfactual credit;
10. proxy equals utility vs proxy/utility divergence.

Predeclare one primary ablation table to avoid a sprawling report.

---

## 19. Recommended figures

1. **Problem decomposition:** delay, stochasticity, observability, confounding, proxy error.
2. **Two-track design:** authentic logs versus structural worlds.
3. **Outcome accuracy does not imply credit accuracy:** scatter across models.
4. **Credit recovery by temporal distance:** DOCM versus baselines.
5. **Leave-one-world-out uncertainty:** uncertainty versus absolute error.
6. **Overoptimization curves:** learned proxy, behavioral proxy, and true utility.
7. **Real-log calibration:** immediate versus delayed outcome model.
8. **System architecture:** model outputs feeding policy learning and Phase 2 backfill.

---

## 20. Reference papers and resources

### Interaction logs and in-the-wild feedback

- Zhao et al., **WildChat: 1M ChatGPT Interaction Logs in the Wild**, arXiv:2405.01470.
- Shi et al., **WildFeedback: Aligning LLMs With In-situ User Interactions and Feedback**, arXiv:2408.15549.
- Peng et al., **WildReward: Learning Reward Models from In-the-Wild Human Interactions**, arXiv:2602.08829 / ACL 2026.
- Zheng et al., **LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset**, arXiv:2309.11998.
- Gao et al., **KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos**, arXiv:2208.08696.

### Delayed credit assignment

- Arjona-Medina et al., **RUDDER: Return Decomposition for Delayed Rewards**, arXiv:1806.07857.
- Harutyunyan et al., **Hindsight Credit Assignment**, arXiv:1912.02503.
- Hung et al., **Optimizing Agent Behavior over Long Time Scales by Transporting Value**, arXiv:1810.06721.
- Chelu et al., **Forethought and Hindsight in Credit Assignment**, arXiv:2010.13685.

### Offline RL and sequence policies

- Kostrikov et al., **Offline Reinforcement Learning with Implicit Q-Learning**, arXiv:2110.06169.
- Kumar et al., **Conservative Q-Learning for Offline Reinforcement Learning**, arXiv:2006.04779.
- Chen et al., **Decision Transformer: Reinforcement Learning via Sequence Modeling**, arXiv:2106.01345.
- Jiang and Li, **Doubly Robust Off-policy Value Evaluation for Reinforcement Learning**, arXiv:1511.03722.
- Farajtabar et al., **More Robust Doubly Robust Off-policy Evaluation**, arXiv:1802.03493.

### LLM post-training

- Rafailov et al., **Direct Preference Optimization**, arXiv:2305.18290.
- Schulman et al., **Proximal Policy Optimization Algorithms**, arXiv:1707.06347.
- Shao et al., **DeepSeekMath** (introducing GRPO), arXiv:2402.03300.
- Hu et al., **OpenRLHF**, arXiv:2405.11143.

### Asynchronous and stale-policy learning

- Espeholt et al., **IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures**, arXiv:1802.01561.
- Fu et al., **AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning**, arXiv:2505.24298.

### Reward overoptimization and proxy risk

- Gao, Schulman, and Hilton, **Scaling Laws for Reward Model Overoptimization**, ICML 2023.
- Skalse et al., **Defining and Characterizing Reward Gaming**, NeurIPS 2022.
- OpenAI, **Expanding on Sycophancy**, 2025, as an applied motivation rather than a formal benchmark.

---

## 21. Immediate implementation checklist

- [ ] Create repository and schema package.
- [ ] Implement deterministic World A.
- [ ] Write exact intervention tests.
- [ ] Implement outcome GRU and RUDDER baseline.
- [ ] Produce outcome-versus-credit sanity plot.
- [ ] Implement DOCM outcome/prefix/Q heads.
- [ ] Add ensemble wrapper.
- [ ] Build WildChat adapter without redistributing raw text.
- [ ] Define and freeze feedback taxonomy v0.1.
- [ ] Label and manually review the first 200 examples.
- [ ] Implement BC and recurrent IQL.
- [ ] Run first overoptimization experiment.
- [ ] Decide at Gate A whether to continue, simplify, or reposition.


---

# Phase 2 Design: Infrastructure for Delayed-Outcome RL

> **Document status:** implementation design v0.1  
> **Phase objective:** extract a reusable system for training policies from outcomes that arrive after rollout completion.  
> **Working package name:** `longfeedback`  
> **First integration target:** TRL for accessibility; `verl` for scalable LLM RL; OpenRLHF optional.

---

## 1. Executive summary

Standard LLM RL pipelines assume a rollout can be scored immediately or within the same training job. Even when a custom reward function performs slow I/O, the reward is generally expected to return before the batch is optimized. A real behavioral outcome may arrive minutes, days, or weeks later, after:

- the rollout worker has terminated;
- the policy has changed several times;
- the trajectory has accumulated additional actions;
- the observation window has been censored;
- an earlier provisional label has been revised;
- the user has been exposed to actions from multiple policy versions.

Phase 2 builds the missing lifecycle layer between agent interaction and the RL trainer.

The core abstraction is:

```text
rollout events
    → versioned trajectory
    → sealed outcome window
    → pending outcome
    → observed/censored/revised outcome
    → reward + credit computation
    → immutable training sample
    → offline or stale-policy-aware update
```

The infrastructure should not attempt to replace `verl`, TRL, OpenRLHF, Ray, vLLM, or a production event platform. It should provide the **delayed reward control plane and data contracts** those systems do not natively define.

---

## 2. Requirements derived from Phase 1

Phase 1 should determine the minimum requirements. The expected set is:

1. Store complete policy provenance and action log probabilities at rollout time.
2. Assemble events into trajectories using event time, not only processing time.
3. Support multiple outcome definitions and windows for one trajectory.
4. Represent right censoring and ambiguous outcomes explicitly.
5. Attach provisional, final, and revised outcomes idempotently.
6. Version reward models, credit models, labelers, and outcome definitions.
7. Recompute or backfill rewards without mutating raw trajectories.
8. Produce trainer-neutral batches containing terminal reward, per-step rewards, behavior log probabilities, and staleness metadata.
9. Track whether a sample is on-policy, stale, offline, or unidentifiable due to missing propensity/confounding.
10. Measure proxy reward separately from observed behavioral outcome and true utility where available.
11. Recover safely from duplicate events, out-of-order events, and interrupted jobs.
12. Provide local-mode implementations usable by one researcher.

---

## 3. Non-goals

The first infrastructure release will not:

- implement a general-purpose Kafka replacement;
- provide a hosted multi-tenant SaaS;
- guarantee exactly-once delivery across arbitrary external systems;
- resolve real-world user identity;
- infer causal effects automatically;
- train every RL algorithm;
- hide the difference between on-policy and delayed off-policy data;
- automatically decide when a behavioral outcome is ethically safe to optimize;
- support unrestricted sensitive data.

The package provides contracts and reference components. Production adopters can replace storage and orchestration adapters.

---

## 4. Relationship to existing frameworks

### 4.1 TRL

TRL supports custom reward functions in `GRPOTrainer`, including asynchronous coroutine rewards. This is useful for slow remote model calls. The missing abstraction for this project is a reward that is **not yet observable during the training step**, must be persisted, and may arrive after the model version has changed.

### 4.2 `verl`

`verl` supports custom reward functions, model-based rewards, multi-turn rollouts, and synchronous/asynchronous reward-loop implementations. The project should integrate with those hooks rather than fork the trainer. The added layer is persistent trajectory/outcome lifecycle, reward revision, long-delay staleness metadata, and backfilled replay.

### 4.3 OpenRLHF

OpenRLHF provides scalable PPO, DPO, reward-model training, Ray scheduling, vLLM, and DeepSpeed. It is a useful optional adapter, but adding three integrations before one is reliable would dilute the project.

### 4.4 AReaL and IMPALA

AReaL and IMPALA address asynchronous actors and learners, with policy staleness and off-policy correction. They are relevant but solve a shorter operational delay: data are still generated as RL rollouts whose rewards are available for learning. LongFeedback adds **outcome-time delay and reward finalization**, which can span multiple learner updates or processes.

### 4.5 Design decision

Build a trainer-neutral `ResolvedTrajectoryBatch`. Implement adapters in this order:

1. offline PyTorch reference trainer;
2. TRL;
3. `verl`;
4. OpenRLHF only after external demand.

---

## 5. Architectural principles

### 5.1 Raw events are immutable

Never overwrite the source event log. Corrections are new records linked to earlier records.

### 5.2 Derived artifacts are versioned

A reward is not a property permanently attached to a trajectory. It is the output of:

\[
R = f(\tau,\text{outcome-definition},\text{reward-model-version},\text{credit-version}).
\]

Changing any input produces a new reward artifact.

### 5.3 Event time and processing time are separate

Every event stores:

- `event_time`: when the interaction happened;
- `ingested_at`: when the system received it;
- `processed_at`: when a resolver used it.

Outcome windows and trajectory ordering use event time. Watermarks determine when a window is sufficiently complete.

### 5.4 Idempotency is mandatory

All externally supplied events and outcome observations require stable idempotency keys. Replaying an ingestion job must not duplicate actions or rewards.

### 5.5 Censoring is first-class

“Not observed yet,” “observation window ended,” and “observed negative” are different states.

### 5.6 Training samples are immutable snapshots

A trainer consumes a snapshot with fixed:

- trajectory version;
- outcome version;
- reward version;
- behavior policy information;
- model inputs.

If the outcome changes later, create a new sample and deprecate the old one; do not silently mutate completed runs.

### 5.7 Identification metadata travels with the sample

Every sample states whether propensities are exact, estimated, missing, or confounded. Algorithms may reject incompatible samples.

---

## 6. Logical architecture

```text
┌──────────────────────┐
│ Agent / Rollout      │
│ TRL / verl / custom  │
└──────────┬───────────┘
           │ events + policy provenance
           ▼
┌──────────────────────┐
│ Event Ingestor       │
│ validation/idempotency│
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Event Store          │
│ append-only          │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Trajectory Assembler │
│ ordering + windows   │
└─────┬───────────┬────┘
      │           │
      │           ▼
      │   ┌──────────────────┐
      │   │ Outcome Resolver │◄── product/user/system events
      │   │ pending/finalized│
      │   └─────────┬────────┘
      │             ▼
      │   ┌──────────────────┐
      └──►│ Reward Backfiller│◄── RM / credit model registry
          └─────────┬────────┘
                    ▼
          ┌──────────────────┐
          │ Training Sample  │
          │ Store / Replay   │
          └─────────┬────────┘
                    ▼
          ┌──────────────────┐
          │ Trainer Adapters │
          │ offline/TRL/verl │
          └─────────┬────────┘
                    ▼
          ┌──────────────────┐
          │ Evaluation &     │
          │ Monitoring       │
          └──────────────────┘
```

### 6.1 Local reference deployment

For research and CI:

- Parquet files for immutable events and samples;
- SQLite or DuckDB for indexes and job state;
- local filesystem or S3-compatible object store for payloads/checkpoints;
- Python worker processes;
- no mandatory message broker.

### 6.2 Scalable deployment adapters

Optional adapters:

- Kafka/PubSub for event ingress;
- Postgres for metadata and state transitions;
- S3/GCS/HDFS for payloads;
- Ray for backfill and training jobs;
- Redis only for ephemeral locks/caches, never source of truth.

---

## 7. Core domain model

### 7.1 Identifiers

Use globally unique, opaque IDs:

- `trajectory_id`
- `episode_id`
- `event_id`
- `action_id`
- `outcome_observation_id`
- `outcome_resolution_id`
- `reward_artifact_id`
- `training_sample_id`
- `policy_version_id`
- `reward_model_version_id`
- `outcome_definition_id`

Identifiers should not contain raw user IDs or sensitive metadata.

### 7.2 Trajectory state machine

```text
OPEN
  │ seal condition met
  ▼
SEALED_PENDING_OUTCOME
  ├── outcome observed ─────────► OUTCOME_OBSERVED
  ├── provisional signal ───────► OUTCOME_PROVISIONAL
  ├── window expires ───────────► CENSORED
  └── invalid data ─────────────► INVALID

OUTCOME_PROVISIONAL
  ├── finalized ────────────────► OUTCOME_OBSERVED
  ├── revised provisional ──────► OUTCOME_PROVISIONAL
  └── expires ──────────────────► CENSORED

OUTCOME_OBSERVED
  ├── reward computed ──────────► REWARD_RESOLVED
  └── correction received ──────► OUTCOME_REVISED

OUTCOME_REVISED
  └── recompute reward ─────────► REWARD_RESOLVED

REWARD_RESOLVED
  ├── sample materialized ──────► TRAINING_READY
  ├── reward superseded ────────► REWARD_SUPERSEDED
  └── policy/legal hold ────────► QUARANTINED

TRAINING_READY
  ├── consumed ─────────────────► CONSUMED
  ├── superseded ───────────────► REWARD_SUPERSEDED
  └── quarantined ──────────────► QUARANTINED
```

State transitions must be persisted atomically with an audit record.

### 7.3 Outcome definition

An `OutcomeDefinition` is configuration plus executable resolver logic.

```python
@dataclass(frozen=True)
class OutcomeDefinition:
    outcome_definition_id: str
    name: str
    version: str
    anchor_event: str
    window_start: timedelta
    window_end: timedelta
    type: Literal["binary", "continuous", "ordinal", "survival", "vector"]
    finalization_delay: timedelta
    censoring_policy: str
    resolver_path: str
    utility_semantics: str
    proxy_semantics: str
```

Examples:

- future correction within four conversation turns;
- task completion within 24 hours;
- return within 28 days;
- unsubscribe within seven days;
- vector outcome `[progress, return, complaint, fatigue]`.

### 7.4 Outcome observation vs outcome resolution

An observation is a raw fact, such as “user returned at timestamp.” A resolution is the interpretation under an outcome definition.

```python
@dataclass(frozen=True)
class OutcomeObservation:
    outcome_observation_id: str
    entity_key_hash: str
    event_time: datetime
    type: str
    value: Any
    source: str
    idempotency_key: str
    ingested_at: datetime

@dataclass(frozen=True)
class OutcomeResolution:
    outcome_resolution_id: str
    trajectory_id: str
    outcome_definition_id: str
    status: Literal["provisional", "final", "censored", "ambiguous"]
    value: Any | None
    observed_at: datetime | None
    resolved_at: datetime
    evidence_ids: tuple[str, ...]
    resolver_version: str
    supersedes_id: str | None
```

### 7.5 Reward artifact

```python
@dataclass(frozen=True)
class RewardArtifact:
    reward_artifact_id: str
    trajectory_id: str
    trajectory_version: int
    outcome_resolution_id: str
    reward_model_version_id: str | None
    credit_model_version_id: str | None
    terminal_reward: float | None
    per_step_rewards: tuple[float, ...] | None
    per_step_uncertainty: tuple[float, ...] | None
    proxy_reward: float | None
    utility: float | None
    created_at: datetime
    supersedes_id: str | None
    config_hash: str
```

### 7.6 Policy provenance

At action generation time, record:

```python
@dataclass(frozen=True)
class PolicyProvenance:
    policy_version_id: str
    model_name: str
    model_revision: str
    checkpoint_hash: str
    tokenizer_revision: str
    decoding_config_hash: str
    adapter_revision: str | None
    reference_policy_version_id: str | None
    generated_at: datetime
```

For each selected action/token sequence, store behavior log probability:

\[
\log\mu(a_t\mid h_t)
=
\sum_{j=1}^{L_t}\log\mu(y_{t,j}\mid h_t,y_{t,<j}).
\]

Optionally store token-level log probabilities for PPO/V-trace-compatible updates.

---

## 8. Storage schema

### 8.1 Event table

| Column | Type | Notes |
|---|---|---|
| event_id | string | primary key |
| trajectory_id | string | partition/index key |
| entity_key_hash | string nullable | pseudonymous join key |
| event_time | timestamp UTC | semantic ordering |
| ingested_at | timestamp UTC | arrival time |
| step_index | int | may be assigned after assembly |
| event_type | enum | observation/action/response/outcome/system |
| payload_uri | string nullable | large payload location |
| payload_json | JSON nullable | small payload |
| source | string | adapter/source |
| source_row_id | string nullable | lineage |
| idempotency_key | string | unique by source |
| schema_version | string | migration support |

Partition by date/source; index by trajectory and event time.

### 8.2 Action table

| Column | Type | Notes |
|---|---|---|
| action_id | string | primary key |
| event_id | string | source event |
| trajectory_id | string | |
| policy_version_id | string | exact actor |
| action_text_uri | string nullable | encrypted/sanitized as needed |
| action_struct | JSON nullable | tool/action metadata |
| behavior_logprob | float | sequence log probability |
| token_logprobs_uri | string nullable | optional Arrow array |
| reference_logprob | float nullable | KL computation |
| sampling_seed | int nullable | reproducibility |
| decoding_config_hash | string | |

### 8.3 Outcome and reward tables

Keep observations, resolutions, and reward artifacts separate. Use append-only version rows and `supersedes_id` links. Build a view selecting the latest valid artifact.

### 8.4 Training sample manifest

The manifest should reference immutable payloads:

```json
{
  "training_sample_id": "...",
  "trajectory_id": "...",
  "trajectory_version": 3,
  "reward_artifact_id": "...",
  "policy_version_id": "...",
  "sample_type": "delayed_online",
  "propensity_quality": "exact",
  "policy_staleness_steps": 420,
  "payload_uri": "s3://.../sample.arrow",
  "created_at": "...",
  "schema_version": "1.0"
}
```

---

## 9. Trajectory assembly

### 9.1 Boundary strategies

A trajectory may be defined by:

- explicit episode ID;
- conversation ID;
- session inactivity timeout;
- fixed horizon after anchor event;
- user lifecycle window;
- task/workflow ID.

The boundary strategy is versioned. Different outcome definitions may use different views of the same raw events.

### 9.2 Event ordering

Sort by:

1. `event_time`;
2. source sequence number if available;
3. `ingested_at`;
4. stable event ID tie-breaker.

Flag impossible role/order patterns rather than silently repairing them.

### 9.3 Late events and watermarks

Define a watermark \(W(t)\) such that events with event time before \(W\) are considered sufficiently complete. A trajectory can be provisionally sealed before the watermark and finalized afterward.

Example:

```yaml
assembly:
  inactivity_timeout: 30m
  allowed_lateness: 24h
  provisional_seal_after: 45m
  final_seal_after: 24h
```

Late events create a new trajectory version and may trigger outcome/reward supersession.

### 9.4 Entity joins

Real deployments may need to join a trajectory to later outcomes using a pseudonymous entity key. The package should accept externally generated hashes and never prescribe cross-device identity resolution.

Support one-to-many joins:

- one trajectory may have multiple outcome definitions;
- one outcome observation may resolve multiple trajectories;
- attribution policy determines eligibility.

### 9.5 Attribution windows

For outcome at time \(t_Y\), candidate actions satisfy

\[
t_Y-L_{max}\le t_a\le t_Y-L_{min}.
\]

The infrastructure does not assume last-touch attribution. It stores all eligible trajectories and delegates credit to the chosen model/definition.

---

## 10. Outcome resolution and censoring

### 10.1 Resolver interface

```python
class OutcomeResolver(Protocol):
    definition: OutcomeDefinition

    def resolve(
        self,
        trajectory: TrajectorySnapshot,
        observations: Sequence[OutcomeObservation],
        as_of: datetime,
    ) -> OutcomeResolution:
        ...
```

Resolvers must be deterministic given inputs and version. External model calls should be cached and represented as evidence artifacts.

### 10.2 Fixed-window binary resolver

For event set \(E\), window \([t_0,t_1]\):

\[
Y=\mathbf 1[\exists e\in E:t_0\le t_e\le t_1].
\]

Status is:

- final positive when event occurs;
- final negative only after \(t_1+\text{allowed lateness}\);
- pending before then;
- censored if source observation coverage ended before \(t_1\).

### 10.3 Continuous/aggregate resolver

Examples: total future usage, total successful tool completions, number of corrections.

Include aggregation semantics:

- sum;
- max;
- first occurrence;
- discounted sum;
- competing risks.

### 10.4 Multi-objective outcome vector

A safe default is to preserve components:

\[
y=[y_{progress},y_{return},y_{complaint},y_{fatigue},y_{cost}].
\]

Scalarization is versioned:

\[
R_w=w^\top y.
\]

Do not discard vector outcomes after scalarization. This enables retrospective safety analysis.

### 10.5 Censoring-aware sampling

Options:

1. train only on finalized samples;
2. include censored samples in a survival objective;
3. apply IPCW;
4. use positive-unlabeled methods when only positives are reliably observed.

Every trainer run declares which policy it uses.

---

## 11. Reward and credit computation

### 11.1 Reward provider interface

```python
class RewardProvider(Protocol):
    version_id: str

    def score(
        self,
        trajectories: Sequence[TrajectorySnapshot],
        outcomes: Sequence[OutcomeResolution],
    ) -> Sequence[RewardArtifact]:
        ...
```

Implementations:

- terminal observed outcome;
- deterministic outcome scalarizer;
- learned trajectory RM;
- DOCM per-step credit;
- RUDDER redistribution;
- ensemble mean/LCB;
- hybrid observed outcome plus learned shaping.

### 11.2 Reward revision

A reward may change because:

- the outcome was revised;
- a new RM version is deployed;
- credit redistribution changes;
- scalarization weights change;
- censoring status becomes final.

The system creates a new artifact. Training runs record the exact artifact IDs used.

### 11.3 Backfill job

```python
@dataclass
class BackfillSpec:
    trajectory_query: str
    outcome_definition_id: str
    reward_provider_version_id: str
    credit_provider_version_id: str | None
    as_of: datetime
    overwrite_policy: Literal["never", "supersede"]
    output_dataset_name: str
```

Backfill is embarrassingly parallel by trajectory. It must support checkpointing and deterministic retry.

### 11.4 Per-step shaping constraints

Reward redistribution can alter the optimal policy unless it is carefully defined. A potential-based shaping signal has the form

\[
F(s_t,s_{t+1})=\gamma\Phi(s_{t+1})-\Phi(s_t).
\]

For learned delayed credit, do not automatically claim policy invariance. Store whether a reward is:

- observed terminal;
- potential-based shaping;
- predictive redistribution;
- estimated causal credit;
- heuristic shaping.

The trainer and report should surface this provenance.

---

## 12. Trainer-neutral batch contract

```python
@dataclass
class ResolvedTrajectoryBatch:
    trajectory_ids: list[str]
    observations: TensorTree
    actions: TensorTree
    attention_mask: torch.Tensor
    behavior_logprobs: torch.Tensor | None
    reference_logprobs: torch.Tensor | None
    terminal_rewards: torch.Tensor | None
    per_step_rewards: torch.Tensor | None
    reward_uncertainty: torch.Tensor | None
    outcome_vectors: torch.Tensor | None
    censored: torch.Tensor
    policy_staleness_steps: torch.Tensor
    policy_staleness_seconds: torch.Tensor
    propensity_quality: list[str]
    importance_weights: torch.Tensor | None
    metadata: list[dict[str, Any]]
```

A batch validator enforces algorithm requirements. Examples:

- PPO adapter requires behavior token log probabilities.
- DR OPE requires exact or accepted estimated propensities.
- censored samples require a compatible objective.
- hidden-confounded samples cannot enter naive importance sampling.

---

## 13. Learning modes

### 13.1 Mode A: finalized offline training

Wait until outcomes finalize, materialize a static dataset, and run BC/IQL/CQL/DT/reward-model training. This is the easiest and most statistically transparent mode.

Use when:

- outcomes take days/weeks;
- policy updates need not be continuous;
- behavior policies are heterogeneous;
- reproducibility is more important than freshness.

### 13.2 Mode B: periodic delayed batches

At a cadence, resolve all newly mature trajectories and update the model. Each batch may contain multiple behavior-policy versions.

Record delay:

\[
\Delta_i^{policy}
=k_{train}-k_{behavior,i},
\]

and wall-clock lag.

### 13.3 Mode C: asynchronous delayed online learning

Rollouts continue while outcomes arrive and learners update. This is the hardest mode because reward delay creates policy staleness beyond ordinary actor/learner lag.

The package should initially support it as an experimental adapter, not the default.

### 13.4 Mode D: learned proxy now, true outcome later

A provisional RM score is available immediately; the behavioral outcome arrives later.

Training can use:

- provisional shaping for rapid updates;
- later outcome to recalibrate the RM;
- importance-weighted replay or correction;
- monitoring of proxy/outcome divergence.

Keep the two rewards separate; do not overwrite provisional scores with final outcomes.

---

## 14. Policy staleness and off-policy correction

### 14.1 Importance ratios

For trajectory generated by behavior policy \(\mu\) and current policy \(\pi\):

\[
\rho_{0:T}
=
\prod_{t=0}^{T}
\frac{\pi(a_t\mid h_t)}{\mu(a_t\mid h_t)}.
\]

For language actions, sequence ratios can explode because each response contains many tokens. Prefer token-level clipped ratios or action-level normalized log-ratio diagnostics.

### 14.2 PPO-style ratio

\[
r_t(\theta)
=
\exp\left(
\log\pi_\theta(a_t\mid h_t)-
\log\mu(a_t\mid h_t)
\right).
\]

Clipped objective:

\[
L^{clip}
=
\mathbb E
\left[
\min(r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)
\right].
\]

Classic PPO assumes relatively fresh samples. Long delays require stricter acceptance or explicit stale-data methods.

### 14.3 V-trace-style correction

For stale trajectories, clipped importance weights:

\[
\rho_t=\min(\bar\rho,\pi(a_t|s_t)/\mu(a_t|s_t)),
\qquad
c_t=\min(\bar c,\pi(a_t|s_t)/\mu(a_t|s_t)).
\]

V-trace target:

\[
v_s
=V(s)+
\sum_{t=s}^{s+n-1}
\gamma^{t-s}
\left(\prod_{i=s}^{t-1}c_i\right)
\rho_t
\left(r_t+\gamma V(s_{t+1})-V(s_t)\right).
\]

This is a candidate for short-to-moderate staleness, not a guarantee for arbitrarily old behavioral data.

### 14.4 Staleness gates

Samples can be:

- accepted on-policy;
- accepted with correction;
- routed to offline RL;
- used only for reward-model training;
- rejected.

Example policy:

```yaml
staleness:
  on_policy_max_updates: 2
  corrected_max_updates: 20
  offline_after_updates: 20
  max_kl_to_current: 0.5
  min_effective_sample_size: 0.1
```

### 14.5 Effective sample size

For weights \(w_i\):

\[
ESS=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
\]

Log normalized ESS and reject updates below a configured threshold.

### 14.6 Recommended first release behavior

- Use delayed outcomes for RM/credit-model updates regardless of actor staleness.
- Use finalized trajectories for offline policy learning.
- Permit online PPO/GRPO only when the policy lag is small and log probabilities are present.
- Route long-delay samples to IQL/CQL/advantage-weighted replay rather than pretending they are on-policy.

---

## 15. Delayed GRPO considerations

GRPO commonly groups multiple completions for the same prompt and normalizes rewards within the group. Delayed behavioral outcomes create three problems:

1. group members may mature at different times;
2. some outcomes may be censored;
3. policy versions may differ if completions were generated asynchronously.

### 15.1 Group finalization policy

Options:

- wait for all group outcomes;
- train when a minimum fraction mature;
- impute provisional RM scores for unresolved members;
- abandon incomplete groups after expiry.

Each option changes bias and latency.

### 15.2 Masked group normalization

For observed rewards in group \(G\):

\[
A_i=
\frac{R_i-\bar R_G}{s_G+\epsilon},
\qquad i\in G_{observed}.
\]

Require at least two distinct observed rewards; otherwise skip or use a value baseline. Never label censored outcomes as zero merely to complete the group.

### 15.3 Hybrid delayed advantage

Use final outcome plus learned credit/value baseline:

\[
\hat A_{i,t}
=
\alpha A_i^{group}
+(1-\alpha)
\left(
\hat Q(h_{i,t},a_{i,t})-\hat V(h_{i,t})
\right).
\]

This is an experimental algorithm and should live outside the core infrastructure until validated.

---

## 16. TRL integration

### 16.1 Adapter strategy

Do not modify `GRPOTrainer` initially. Provide two integration paths.

#### Path A: offline resolved dataset

Materialize a Hugging Face Dataset with:

- prompt/context;
- completion/action;
- terminal/per-step reward;
- behavior log probability;
- sample weight;
- reward metadata.

Use a custom trainer or compatible offline objective.

#### Path B: pending reward wrapper

During generation:

1. write trajectory and policy provenance;
2. return a placeholder job ID rather than a training reward;
3. suspend that sample from the active optimization batch;
4. later read resolved samples from replay.

Because stock GRPO expects rewards for current completions, the delayed replay learner may be a separate loop using TRL model utilities rather than a thin reward function.

### 16.2 Proposed package

```text
integrations/trl/
├── recorder.py
├── resolved_dataset.py
├── delayed_replay_trainer.py
├── reward_provider.py
└── examples/
```

### 16.3 Example

```python
recorder = TrajectoryRecorder(store=store, policy_registry=registry)

completion = model.generate(**inputs)
handle = recorder.record_generation(
    prompt=prompt,
    completion=completion,
    token_logprobs=token_logprobs,
    outcome_definition="future_success_24h:v1",
)

# Later, possibly in another process
resolved = replay_loader.load_ready(
    reward_version="docm-ensemble-lcb:v3",
    max_policy_staleness=20,
)
trainer.train_on_resolved(resolved)
```

---

## 17. `verl` integration

### 17.1 Integration points

Use `verl` for rollout and scalable training. LongFeedback provides:

- a rollout recorder callback;
- trajectory/outcome IDs in `extra_info`;
- a remote or local resolved-reward service;
- a replay source for mature trajectories;
- a reward manager that understands terminal/per-step backfilled rewards;
- policy-version and staleness metadata.

### 17.2 Two execution modes

#### Synchronous experiment mode

The structural environment can resolve outcomes at the end of a simulated long-horizon episode. This uses normal `verl` reward hooks and validates algorithms.

#### Persistent delayed mode

Rollouts are stored and training is decoupled. A separate process materializes mature samples and launches replay-based updates. This may require a custom data source rather than only a custom reward function.

### 17.3 Adapter layout

```text
integrations/verl/
├── trajectory_callback.py
├── reward_manager.py
├── mature_sample_source.py
├── config/
│   ├── delayed_grpo.yaml
│   └── delayed_ppo.yaml
└── examples/
```

### 17.4 Version compatibility

Pin a tested `verl` commit/release in each LongFeedback release. Use a small compatibility layer and CI smoke test rather than importing deep private internals across the package.

---

## 18. Model registry

Track:

- policy model;
- reference policy;
- outcome labeler;
- reward model;
- credit model;
- value model;
- behavior-policy estimator;
- censoring model.

Each version record contains:

```yaml
model_version_id: docm:0.3.1+sha.abc123
artifact_uri: s3://...
code_commit: abc123
training_data_manifest: ...
config_hash: ...
framework_versions: ...
metrics_uri: ...
created_at: ...
parent_version: ...
```

The registry can be a simple table plus object storage in v0.1. MLflow integration is optional.

---

## 19. Monitoring and evaluation

### 19.1 Data pipeline metrics

- events ingested per source;
- duplicate rate;
- invalid-event rate;
- late-event distribution;
- trajectory assembly lag;
- pending/final/censored outcome counts;
- reward-backfill throughput;
- supersession rate;
- quarantine count.

### 19.2 Statistical metrics

- outcome base rate by policy version;
- censoring rate;
- reward distribution;
- RM calibration;
- uncertainty distribution;
- policy staleness;
- importance-ratio quantiles;
- ESS;
- OOD score;
- subgroup performance.

### 19.3 Optimization safety metrics

Maintain separate time series for:

- learned reward;
- observed behavioral outcome;
- true utility where available;
- negative outcome components;
- KL to reference;
- action/response length;
- uncertainty;
- support distance;
- reward-model version.

Alert conditions might include:

```yaml
alerts:
  proxy_up_true_outcome_down:
    proxy_delta_min: 0.05
    outcome_delta_max: -0.01
  uncertainty_spike:
    p95_relative_increase: 0.5
  low_effective_sample_size:
    threshold: 0.1
  censoring_shift:
    relative_increase: 0.25
```

### 19.4 Evaluation snapshots

Every training checkpoint should reference an immutable evaluation snapshot:

- world configs and seeds;
- real-log test manifest;
- reward models;
- outcome definitions;
- policy evaluator version.

---

## 20. Reliability and consistency

### 20.1 Delivery semantics

The system can provide effectively-once processing through:

- at-least-once event delivery;
- unique idempotency keys;
- transactional insert-or-ignore;
- deterministic resolvers;
- immutable versioned outputs.

Do not claim universal exactly-once semantics.

### 20.2 Concurrency control

Use optimistic locking on trajectory/resolution version. A worker writes only if the current version matches what it read. Conflicts trigger retry.

### 20.3 Failure recovery

Every job records:

- input query/manifest;
- cursor or partition progress;
- output artifact IDs;
- code/config hash;
- retry count;
- terminal status.

Backfill outputs are committed by partition and published atomically through a final manifest.

### 20.4 Data validation

Use schema validation at ingress and materialization. Quarantine rather than drop invalid records. Store a reason code.

### 20.5 Revisions

A revision graph should be acyclic. Enforce that `supersedes_id` references an earlier artifact for the same trajectory/definition.

---

## 21. API design

### 21.1 High-level client

```python
from longfeedback import Client

client = Client.from_config("longfeedback.yaml")

trajectory = client.trajectories.start(
    entity_key_hash=user_hash,
    policy_version_id=policy_version,
    outcome_definitions=["return_28d:v1", "complaint_7d:v2"],
)

trajectory.record_observation(observation)
trajectory.record_action(
    action=completion,
    behavior_logprob=logp,
    token_logprobs=token_logps,
)
trajectory.record_response(user_event)
trajectory.seal()
```

### 21.2 Outcome ingestion

```python
client.outcomes.observe(
    entity_key_hash=user_hash,
    event_time=return_timestamp,
    type="user_return",
    value=1,
    idempotency_key=source_event_id,
)

client.outcomes.resolve_ready(as_of=datetime.now(tz=UTC))
```

### 21.3 Reward backfill

```python
job = client.rewards.backfill(
    trajectory_query="state = 'OUTCOME_OBSERVED'",
    outcome_definition="return_28d:v1",
    provider="docm_lcb:v3",
    output_dataset="return28_docm_v3",
)
job.run()
```

### 21.4 Training loader

```python
loader = client.training.loader(
    dataset="return28_docm_v3",
    algorithm="iql",
    filters={
        "propensity_quality": ["exact", "estimated"],
        "max_policy_staleness_steps": 1000,
        "exclude_censored": True,
    },
)

for batch in loader:
    trainer.update(batch)
```

### 21.5 CLI

```bash
longfeedback ingest events --source ./events.parquet
longfeedback assemble trajectories --as-of 2026-07-09T00:00:00Z
longfeedback resolve outcomes --definition return_28d:v1
longfeedback backfill rewards --provider docm_lcb:v3
longfeedback materialize dataset --name ready_v1
longfeedback train --adapter trl --config configs/train/delayed_replay.yaml
longfeedback audit sample <training_sample_id>
```

---

## 22. Configuration

```yaml
storage:
  backend: local
  event_uri: ./data/events
  artifact_uri: ./data/artifacts
  metadata_db: ./data/longfeedback.sqlite

assembly:
  boundary: conversation_id
  allowed_lateness: 24h
  inactivity_timeout: 30m

outcomes:
  definitions:
    - configs/outcomes/future_correction_h4.yaml
    - configs/outcomes/return_28d.yaml

reward:
  provider: docm_lcb
  version: v3
  lcb_lambda: 1.0

training:
  adapter: trl
  mode: delayed_replay
  staleness:
    on_policy_max_updates: 2
    corrected_max_updates: 20
    offline_after_updates: 20
  sample_filters:
    include_censored: false
    accepted_propensity_quality: [exact, estimated]

privacy:
  raw_text_storage: encrypted
  public_export_raw_text: false
  pii_filter_version: v2
```

---

## 23. Testing strategy

### 23.1 Unit tests

- idempotent event insertion;
- event-time ordering;
- state transition validation;
- fixed-window resolver boundary cases;
- censoring versus negative distinction;
- reward supersession;
- importance ratio calculations;
- ESS;
- config and schema migrations.

### 23.2 Property tests

- replaying the same ingestion sequence yields identical database state;
- event permutation within allowed arrival order yields the same finalized trajectory;
- no training sample references mutable/latest aliases;
- a reward revision never changes the old artifact;
- every consumed sample is auditable to raw event IDs;
- exact deterministic resolver is stable across process restarts.

### 23.3 Integration tests

1. Generate structural-world rollout.
2. Record it through the client.
3. Seal trajectory.
4. Delay outcome observation.
5. Resolve and backfill reward.
6. Materialize batch.
7. Train one update.
8. Audit the checkpoint’s sample lineage.

Run this test for local, TRL, and `verl` adapters.

### 23.4 Fault-injection tests

- duplicate events;
- out-of-order events;
- resolver crash after partial writes;
- object-store timeout;
- database lock conflict;
- late outcome correction;
- missing policy checkpoint;
- incompatible schema version;
- partial group outcomes for GRPO.

### 23.5 Statistical regression tests

- delayed replay with zero delay matches synchronous training within tolerance;
- deterministic reward backfill matches direct environment reward;
- stale-policy rejection thresholds produce expected sample counts;
- V-trace/reference implementation agrees on a fixed fixture.

---

## 24. Performance targets

### Local mode

- ingest ≥50K small events/sec from Parquet on a modern workstation;
- assemble ≥10K trajectories/sec for simple boundaries;
- resolve ≥10K deterministic outcomes/sec;
- backfill learned rewards limited mainly by model inference;
- load Arrow batches without Python object-per-event overhead.

### Scalable mode

Targets should be defined after Phase 1 profiling. Avoid premature claims. A reasonable initial stress test is:

- 10M events;
- 1M trajectories;
- 100K outcomes maturing per batch;
- restartable backfill across 100 partitions.

### Cost controls

- cache text embeddings and RM outputs by content/version hash;
- use batched model inference;
- separate lightweight outcome resolution from GPU reward scoring;
- compact old event partitions without deleting lineage;
- support metadata-only dry runs.

---

## 25. Security and privacy

### 25.1 Data minimization

Store only fields required for training and audit. Permit payload references so sensitive text can live in a more protected store.

### 25.2 Pseudonymization

The system accepts pseudonymous entity keys. Salting/hashing should occur upstream. Do not provide reversible identity utilities.

### 25.3 Access boundaries

Separate permissions for:

- raw event payloads;
- sanitized trajectories;
- derived labels;
- model checkpoints;
- public exports.

### 25.4 Deletion and retention

Append-only research lineage conflicts with deletion requirements. Support tombstone records and a compaction job that removes protected payloads while retaining non-identifying aggregate lineage when legally permitted.

### 25.5 Reward safety

Outcome definitions require a human-readable `utility_semantics` and `proxy_semantics`. The CLI should warn when optimizing a proxy without a configured negative-outcome monitor.

---

## 26. Observability and audit interface

A single sample should be inspectable:

```text
training sample
  ├── reward artifact
  │     ├── reward model version
  │     ├── uncertainty
  │     └── outcome resolution
  │            └── raw outcome observations
  ├── trajectory snapshot
  │     └── raw event IDs
  ├── behavior policy version
  ├── behavior log probabilities
  └── trainer run/checkpoint
```

Command:

```bash
longfeedback audit sample <id> --format markdown
```

The output must redact sensitive payloads by default.

---

## 27. Phased implementation roadmap

### Infrastructure v0.1 — local lifecycle core

- Pydantic/dataclass schemas;
- Parquet event store;
- SQLite metadata store;
- trajectory assembler;
- fixed-window outcome resolver;
- reward artifacts and supersession;
- offline batch materializer;
- audit CLI.

### Infrastructure v0.2 — Phase 1 integration

- DOCM reward provider;
- censoring/survival metadata;
- structural-world ingestion;
- policy provenance and log probabilities;
- offline IQL/BC reference trainer;
- statistical monitors.

### Infrastructure v0.3 — TRL adapter

- generation recorder;
- mature replay dataset;
- delayed replay trainer example;
- async remote RM scoring where reward is computable immediately;
- distinction between slow reward and truly delayed reward documented.

### Infrastructure v0.4 — `verl` adapter

- rollout callback;
- mature sample source;
- reward manager;
- synchronous structural environment example;
- persistent delayed replay example;
- compatibility CI.

### Infrastructure v0.5 — scalable adapters

- Postgres metadata backend;
- S3 object backend;
- Ray backfill;
- optional Kafka ingestion;
- partition-level recovery.

### v1.0 — stable contracts

- schema migration policy;
- backward compatibility tests;
- plugin documentation;
- security/privacy guide;
- benchmark examples;
- external contributor feedback.

---

## 28. Acceptance criteria

Phase 2 is ready for public release when an external user can follow one tutorial and:

1. generate or load trajectories;
2. record exact policy provenance;
3. attach an outcome after a simulated delay;
4. distinguish pending, censored, and negative outcomes;
5. backfill a terminal and per-step reward;
6. train a policy through at least one adapter;
7. reproduce an overoptimization curve;
8. trace a checkpoint back to source events;
9. rerun the pipeline idempotently;
10. revise an outcome and produce a superseding reward without corrupting the original run.

---

## 29. Reference papers and systems

### RL algorithms and correction

- Schulman et al., **Proximal Policy Optimization Algorithms**, arXiv:1707.06347.
- Schulman et al., **High-Dimensional Continuous Control Using Generalized Advantage Estimation**, arXiv:1506.02438.
- Espeholt et al., **IMPALA**, arXiv:1802.01561.
- Jiang and Li, **Doubly Robust Off-policy Value Evaluation for Reinforcement Learning**, arXiv:1511.03722.
- Kostrikov et al., **Implicit Q-Learning**, arXiv:2110.06169.
- Kumar et al., **Conservative Q-Learning**, arXiv:2006.04779.
- Shao et al., **DeepSeekMath / GRPO**, arXiv:2402.03300.

### Asynchronous LLM RL systems

- Fu et al., **AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning**, arXiv:2505.24298.
- `verl` / HybridFlow documentation and repository.
- Hugging Face TRL `GRPOTrainer` documentation.
- Hu et al., **OpenRLHF**, arXiv:2405.11143.

### Dataset and offline-RL infrastructure

- Farama Foundation, **Minari**, standard API and format for offline RL datasets.
- Fu et al., **D4RL: Datasets for Deep Data-Driven Reinforcement Learning**.
- Saito et al., **Open Bandit Dataset and Pipeline**, NeurIPS 2020.

### Delayed reward and reward modeling

- Arjona-Medina et al., **RUDDER**, arXiv:1806.07857.
- Peng et al., **WildReward**, arXiv:2602.08829.
- Gao, Schulman, and Hilton, **Scaling Laws for Reward Model Overoptimization**, ICML 2023.

---

## 30. Immediate Phase 2 checklist

Do not start this checklist until Phase 1 Gate B.

- [ ] Inventory repeated Phase 1 trajectory/outcome code.
- [ ] Freeze schema v0.1.
- [ ] Implement append-only local event store.
- [ ] Implement trajectory state machine.
- [ ] Implement fixed-window resolver and censoring.
- [ ] Implement reward artifact versioning and supersession.
- [ ] Implement immutable training sample manifest.
- [ ] Add policy provenance and token log probabilities.
- [ ] Build offline reference loader/trainer.
- [ ] Add end-to-end delayed outcome test.
- [ ] Add sample audit CLI.
- [ ] Build TRL adapter.
- [ ] Build `verl` adapter only after the TRL path is stable.

