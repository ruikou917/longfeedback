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
`refuted_in_this_environment`, not tuned away.

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

## Backlog

- Human validation of the E1 rule labels: ~1,000 stratified examples, two
  annotators on ≥20%, agreement statistics (design doc §6.6).
- Multi-seed statistical protocol: ≥5 seeds with bootstrap confidence
  intervals for the primary Gate B and E5 tables (design doc §13.5).
- CI housekeeping: bump `actions/checkout` and `setup-uv` off the deprecated
  Node 20 runtime.

## Gate C — infrastructure extraction

Extract event stores, outcome resolution, reward backfill, trainer-neutral batches,
and TRL/`verl` adapters only when at least two experiments use the same lifecycle
contract. Do not generalize a component used by one script.
