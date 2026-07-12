# Roadmap and gates

The v0.1 release checklist (two structural worlds, canonical schema, outcome
Transformer, RUDDER baseline, oracle credit evaluator, a policy-learning
experiment, a conversational dataset adapter, and reproducibility commands) is
complete.

## E0 — deterministic vertical slice (done)

Schema → World A → exact/paired oracle → diagnostic baselines → report.

## Gate A — Worlds A/B and DOCM MVP (done)

Stochastic World A with oracle/noisy/partial observability, World B with hidden
Markov intent and confounded logging, adaptive paired Monte Carlo oracle credit,
and three capacity-matched DOCM variants. `make gate-a` emits a machine-readable
`gate_a_decision` block with the three criteria:

1. at least one regime shows similar outcome AUROC across variants but a
   materially different oracle credit recovery;
2. the oracle intervention evaluator is stable under repeated seeds;
3. a Q-greedy policy from the credit head beats behavior cloning on true
   utility without raising the proxy while lowering utility.

## Gate B — robust research result (passed)

Worlds C (delayed conversion) and D (proxy-utility divergence) join A/B;
`make gate-b` trains capacity-matched variants plus a bootstrap ensemble per
family and evaluates uncertainty under parameter and logging-policy shift.
Credit supervision beats outcome-only and prefix training in all four
families, uncertainty flags credit error under every shift, and the real-log
outcome task (`make e1` on the prepared LMSYS snapshot) is learnable above
trivial baselines. `longfeedback report build` renders the paper-style v0.2
report from the artifacts.

Two items originally deferred as "remaining Gate B work before v0.3" have
since been closed (2026-07-10), after the project had already moved on to
v0.3 without them — see `docs/data_governance.md` and below for what changed:

- **WildChat adapter** (`longfeedback data prepare wildchat`,
  `src/longfeedback/data/wildchat.py`) is done. WildChat-1M is ungated and
  fetched directly from HuggingFace (no sign-in, unlike LMSYS).
  `make e1-wildchat` reruns E1 on it as the *primary* source (LMSYS is the
  secondary replication, per `docs/data_governance.md`); real result on a
  15,091-conversation local snapshot: `best_informed_auroc` 0.731 vs. trivial
  0.605 (margin **+0.126** — larger than LMSYS's +0.059). E1's "learnable
  above trivial" finding replicates on the intended primary source, with a
  healthier margin.
- **Leave-one-family-out transfer** is done
  (`leave_one_family_out_transfer` in `src/longfeedback/experiments/gate_b.py`).
  Model weights cannot transfer across families (each world has a different
  observation space by construction), so this tests something narrower and
  honest instead: does a threshold on *standardized* ensemble uncertainty,
  fit by pooling three families' (uncertainty, high-error) pairs, still flag
  high-error predictions in the fourth, held-out family? Real result:
  **3 of 4 families transfer** (World A: balanced accuracy 0.83 vs. chance
  0.50; World B: 0.95; World D: 0.62) — **World C does not** (falls to
  exactly 0.50, chance, despite a *perfect* 1.0 self-calibrated score within
  its own distribution). This is reported honestly as
  `gate_b_decision.transfer` and does **not** retroactively flip Gate B's
  already-established "pass" (which still rests on the original four
  criteria) — it's a stretch extension, not a redefinition.

Still open: human validation of the E1 rule labels (needs human annotators,
not something this can close) — see Backlog.

## v0.3 (current)

### E5 — reward overoptimization (done)

REINFORCE against single/ensemble-mean/ensemble-LCB/KL-regularized learned
rewards in World D, under broad and narrow logging support. `make e5` result:
the Goodhart effect is clearly demonstrated in both regimes (hacking gap up to
~10 utility points); the KL-to-behavior-clone penalty materially mitigates it,
but ensemble-LCB pessimism does not — hypothesis H5 is recorded as
`refuted_in_this_environment`, not tuned away. (The multi-seed protocol below
later refined this: the LCB refutation is seed-robust under narrow support but
seed 0 understated LCB under broad support, where it does robustly help.)

Next: LLM-native reranking (E7) — design proposal below, not started (needs a
user decision on API budget/provider before implementation, unlike E5/E6).

### E6 — randomized bridge (done)

`data/kuairand-data/` holds the real **KuaiRand-Pure** release, fetched
directly from Zenodo (<https://zenodo.org/records/10439422>, CC BY-SA 4.0, no
gated form — checksum `0820331067a3784d9691136f772b35a7` verified). The real
CSV header matched the ADR-009 guess almost exactly (`is_click`, `is_like`,
`is_follow`, `is_comment`, `is_forward`, `is_hate`, `long_view`,
`play_time_ms`, `duration_ms` all present as documented), plus extra columns
(`hourmin`, `profile_stay_time`, `comment_stay_time`, `is_profile_enter`,
`is_rand`, `tab`) the adapter picked up automatically since it passes through
whatever the header has (ADR-009). `is_rand` confirms the file-based
`logging_policy` tagging is exactly right: `log_random_*` is 100% `is_rand=1`,
`log_standard_*` is 100% `is_rand=0`.

`make data-kuairand` (200k rows/file) then `make e6` produces a real result:
`src/longfeedback/experiments/e6.py` compares each video's engagement rate
under the confounded `standard` log against its true population rate measured
directly on the genuinely-randomized `random` log (no model fit — both are
direct empirical rates over disjoint logging populations, so there is no
train/eval leakage to guard, ADR-008). Result: `confounding_bias_detected` —
the confounded-log rate is miscalibrated (mean absolute bias ≈0.20 on a
0–1 rate; RMSE 0.25 vs. **0.08 for a trivial constant-mean baseline** — the
confounded per-video rate is *worse* than just guessing the population mean),
while still carrying weak rank information (Spearman ≈0.26). Verdict:
`hypothesis_h6_confounded_log_bias = "biased_but_rank_useful"`. This is the
concrete demonstration the bridge exists to produce: a production-confounded
log (structurally the same kind of source as WildChat/LMSYS/E1) looks
informative but gives badly uncalibrated absolute value estimates, and only
genuine randomization fixes that.

**This is a demonstration of the problem, not yet a test of the proposed
method.** No model was trained above — both rates are raw log averages. A
separate `e6_feature_adjustment` block tests a narrower, real claim: does a
model conditioning on real confounders (user + video content features,
`RidgeBaseline`, trained on the confounded log) do better than naive
log-averaging, checked against genuinely random exposures? Note this is *not*
the project's core sequential credit-assignment claim (Gate A/B) — KuaiRand
impressions are single-step, so there's no delayed-credit problem here, only
confounding adjustment. Result on the real snapshot: the feature-conditioned
model has the best Brier score of the three (0.1375 vs. 0.1378 trivial vs.
0.1406 naive per-video rate) and the best AUROC (0.571 vs. 0.500 vs. 0.555).
`hypothesis_h6b_feature_adjustment_helps = "supported"` — modest margins, but
correctly signed: conditioning on confounders beats both naive log-averaging
and a trivial guess, validated against real randomization. What remains
untested on real data is the project's actual headline claim (credit
assignment across delayed, multi-step outcomes) — that still only has
synthetic-world evidence (Gate A/B), because real data never provides ground
truth credit to check against.

### Multi-seed statistical protocol (done)

The design doc §13.5 protocol is implemented as the `multiseed` experiment
(`make multiseed`, `src/longfeedback/experiments/multiseed.py`): it reruns
the unmodified Gate B and E5 pipelines across five seeds and reports the
predeclared primary metrics with percentile-bootstrap 95% CIs over seeds
(paired within seed; effect sizes in native units — see the contract section
in `docs/scientific_contract.md`). Real result (2026-07-11, seeds 0–4, all
ten per-seed runs individually passing, `artifacts/multiseed/`):

- **Gate B credit recovery is robust in all four families**, stronger than
  the single-seed claim (which needed only 3/4): per-family credit margin of
  `docm_credit` over the best baseline has CI entirely above zero — World A
  +0.81 [+0.65, +0.95], World B +0.12 [+0.06, +0.19], World C +0.18
  [+0.06, +0.30], World D +0.83 [+0.81, +0.84]. Uncertainty-flags-error
  under shift is likewise robust in all four families (error-detection AUROC
  CIs all above 0.71).
- **The leave-one-family-out transfer finding replicates**: Worlds A/B/D
  transfer stably across seeds (0.83/0.94/0.63 mean balanced accuracy, tight
  CIs), while World C sits at exactly chance (0.50) in 4 of 5 seeds (mean
  0.60, CI [0.50, 0.80] — includes chance). The single-seed "World C does
  not transfer" result was not a seed artifact.
- **E5's Goodhart effect is robust in both regimes**: hacking gap of the
  `single` reward is +9.3 [+8.7, +9.9] utility points under narrow support
  and +5.6 [+3.8, +7.7] under broad. **KL mitigation is robust in both**
  (narrow: gap reduction +7.8 [+7.5, +8.2]).
- **The multi-seed evidence refines seed 0's H5 refutation**: LCB pessimism
  is a robust mitigation under *broad* support (gap reduction +2.6
  [+0.8, +4.1], utility gain +3.2 [+0.9, +5.5]) but not under narrow support
  (−0.0 [−0.8, +0.9], CI spans zero) — precisely the regime where the
  RM-error channel it targets is most active. So the honest multi-seed
  verdict is: LCB helps where reward-model error is mild and fails where it
  matters most; the single-seed refutation stands for the narrow regime and
  was itself partly a seed artifact for the broad one (seed 0 is the only
  seed with a negative broad-regime LCB delta).

### E7 — LLM-native reranking (design proposal, not started)

Unlike E5/E6, this item has no established pattern to extend from in this
repo and its implementation spends real money against an external API, so it
stops at a design proposal pending confirmation rather than code.

**Recommended framing:** swap E5's learned reward-model scorer for an LLM
judge and reuse the rest of the E5/World-D harness unchanged. Concretely,
prompt an LLM with the trajectory prefix and a small set of candidate next
actions, have it score or rank them, and plug that score into the existing
`RewardScorer` interface (`src/longfeedback/experiments/e5.py`). This buys
three results for one new component: (1) agreement between the LLM judge and
the `rules-v2` proxy labels on real logs (extends E1's predictive-only
framing), (2) whether an LLM-judge reward Goodharts the same way a trained
DOCM reward does when optimized against with REINFORCE (extends E5's
hacking-gap methodology directly), and (3) a cost/latency comparison against
the trained encoder.

Alternatives considered and set aside for now: LLM-as-policy (replacing the
softmax policy itself with an LLM action-chooser) and LLM reranking of
KuaiRand candidate items (blends with E6, but is a bigger rewrite of the E6
harness than a one-component swap). Both are more speculative than the
recommended framing, so they are lower priority unless it turns out to be
uninteresting.

**Blocked on a user decision, not on code:** which model/provider and a cost
budget for the API calls (a full E5-style sweep is hundreds of scored
trajectories per checkpoint × several checkpoints × two logging regimes —
this is not a few-dollar smoke test), and a sign-off on the judge prompt/
rubric since it fixes what "reward" means for this experiment's claims.
Implementation should wait for that; nothing about it needs a slow
data-download turnaround the way E6 does, so once scoped it should move fast.

### E8 — real multi-step credit via randomized session steps (phase 1 done: null)

The core credit-assignment claim has only synthetic evidence (Gate A/B);
real logs never reveal per-step counterfactuals. E8 attacks the strongest
available substitute: KuaiRand's randomly-inserted videos are genuine
mid-session randomized interventions inside real user trajectories
(verified on the local snapshot: 65,974 sessions mix randomized and
standard steps; 322,736 of the 1,186,059 randomized steps have >=3
subsequent same-session impressions). Phase 1 (power gate) asks whether a
single randomized step causally moves *later-session* survival at all;
phase 2 (gated on phase 1) grades observationally-trained credit models
against those randomized effects on held-out users. Contract frozen before
any effect computation — see "E8 acceptance contract" in
`docs/scientific_contract.md` and ADR-012.

Real phase-1 result (2026-07-12): all 2,622,668 impressions produced
1,239,180 sessions and 1,186,059 randomized steps. The frozen base-rate rule
moved the primary horizon from k=5 (survival rate 0.156) to k=3 (0.272) before
examining effects. The duration slope was +0.00054 survival probability per SD
of log-duration, 95% user-cluster bootstrap CI [-0.00020, +0.00134]. The CI
includes zero and the estimate is far below the predeclared 0.005 relevance
threshold; the 80%-power MDE was only 0.00110. Verdict:
`refuted_at_this_granularity`; duration-based phase 2 is blocked and will not
be tuned back into significance.

The null is credible, not a wiring artifact: an unregistered manipulation
check (reported for validity only, never part of the frozen gate) shows the
same randomized durations strongly move *immediate* behavior at the same
steps (slope of log play-time on z log-duration ≈ +0.057; long_view
≈ +0.006). A single mid-session content perturbation measurably changes the
step itself yet leaves three-step session survival untouched. Read with E6
(the same platform's confounded log misestimates per-video value by ≈0.20
mean absolute bias), this sharpens the project's story: observational logs
can look strongly informative while carrying essentially zero delayed
interventional effect — per-step credit inferred observationally from such
data would be mostly confounding.

### E9 — HeartSteps randomized longitudinal benchmark (done)

The adapter pins public CC BY 4.0 HeartSteps V1 revision `3016391...`, removes
declared travel periods, reconstructs 7,608 decisions from 37 participants,
and preserves the known 0.6 suggestion probability. A source-semantic audit
caught that `is.randomized` encodes assignment rather than eligibility;
`avail` is the randomized-decision indicator and `send` is treatment, matching
the published analysis. The observed treatment rate is 0.593 over 6,122
available decisions.

The 30-minute positive control reproduces the paper's direction and scale:
+0.109 log steps, approximately +11.5%, 95% user-cluster CI [-0.050, +0.259]
on the log scale (the paper reported +14%, p=0.06). The cross-fitted distal
Week-6 excursion effect is +65 average daily steps, CI [-196, +358]; its early
period estimate is larger (+466) but also imprecise, CI [-39, +1063]. E9
therefore validates the estimator pipeline against a real repeated-randomized
positive control but does **not** authorize model grading on distal credit.
The next evidentiary step is E10, a powered conversational micro-randomized
trial with an objective delayed task outcome.

## Backlog

- Human validation of the E1 rule labels: ~1,000 stratified examples, two
  annotators on ≥20%, agreement statistics (design doc §6.6).
- CI housekeeping: bump `actions/checkout` and `setup-uv` off the deprecated
  Node 20 runtime.

## Gate C — infrastructure extraction

Extract event stores, outcome resolution, reward backfill, trainer-neutral batches,
and TRL/`verl` adapters only when at least two experiments use the same lifecycle
contract. Do not generalize a component used by one script.
