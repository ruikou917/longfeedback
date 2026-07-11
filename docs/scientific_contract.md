# Scientific contract

LongFeedback keeps four targets separate in code, configuration, artifacts, and
claims.

| Quantity | Symbol | Available from | Permitted interpretation |
|---|---:|---|---|
| Behavioral outcome | `Y` | logs and worlds | Observed proxy; not user welfare |
| True utility | `U` | controlled worlds only | Evaluation objective defined by the SCM |
| Predictive contribution | `V(h[t+1]) - V(h[t])` | any outcome-labeled sequence | Change in predicted outcome; not causal |
| Interventional credit | `C(t, a; a_ref, pi_future)` | controlled/randomized regimes | Causal only for the stated intervention and continuation |

The future-policy clause is part of the interventional estimand. The oracle API defines:

- **Frozen continuation:** replace the selected action and replay recorded future
  actions with paired exogenous noise. Every oracle example stores a hash of that
  exact continuation.
- **Policy-reactive continuation:** replace the action and resample future actions
  from a named policy on the changed state history.

The E0 command intentionally uses frozen continuation only. Policy-reactive behavior
is tested at the API level and enters the experiment matrix once stochastic policy
continuations receive their own uncertainty and acceptance criteria.

Real conversational logs may train outcome and prefix-value models. They do not
provide causal-credit labels unless a defensible randomized or identified design is
present. Conversation termination is not automatically a negative label; pending,
censored, ambiguous, and observed outcomes remain distinct.

## E0 acceptance contract

- The full run completes on CPU in less than five minutes.
- Exact deterministic and paired Monte Carlo utility credit correlate above 0.99.
- A nonzero one-step oracle credit matches an independently derived closed form.
- Deterministic oracle Monte Carlo standard error is zero within floating tolerance.
- Prefix features contain no future actions.
- RUDDER-style increments telescope to the predicted final-minus-initial value.
- Identical configurations and seeds produce identical scientific metrics.

For finite diagnostic plots, correlation with a constant predictor is displayed as
`0.0`; this is explicitly marked as an undefined-correlation convention, not a
statistical estimate.

## Gate A acceptance contract

- Every DOCM variant shares one architecture and parameter count; comparisons
  are capacity-matched by construction, and the metrics report asserts it.
- Oracle credit labels use paired common-random-number rollouts; rollout counts
  escalate until the Monte Carlo standard error meets the configured threshold
  or the rollout cap, and the achieved maximum is reported.
- Oracle labels are re-estimated on a subsample under a shifted seed; the two
  estimates must correlate above the configured stability threshold.
- The primary credit target is interventional **utility** credit with frozen
  continuation. Scores for the outcome-only and prefix variants against that
  target are associations between a predictive quantity and a causal one; the
  metrics file carries this caveat explicitly.
- World B's privileged intent signal may drive logging policies but must never
  enter released features or payloads; property tests enforce this, and
  confounded regimes are flagged `propensity_quality: confounded`.
- The policy sanity check reports true utility and behavioral proxy separately
  for the behavior policy, a behavior clone, and the Q-greedy policy; a proxy
  gain paired with a utility loss fails the gate.
- Identical configurations and seeds produce identical scientific metrics.

## Gate B acceptance contract

- Capacity matching is asserted within each world family (observation
  dimensions differ across families, so cross-family parameter counts do not).
- Credit supervision must beat both the outcome-only and prefix variants on
  held-out credit Spearman in at least three of the four structural families.
- Ensemble uncertainty is the between-member standard deviation of a bootstrap
  ensemble; under at least one distribution shift it must correlate with
  absolute credit error and detect high-error predictions above chance.
- Distribution shifts are declared per family (parameter shift or
  logging-policy shift).
- Leave-one-family-out transfer (`gate_b_decision.transfer`) is reported
  honestly per family but never gates Gate B's `pass`, which still rests
  solely on the original four criteria above — it is a stretch extension
  added after Gate B was already recorded as passed, not a redefinition of
  what passing means. It can only test whether a *threshold on standardized
  ensemble uncertainty* fit on three families generalizes to the fourth
  (Youden's-J-optimal threshold, balanced accuracy against that family's own
  median-split high/low error labels) — never raw model weights, which
  cannot transfer across families with different observation spaces by
  construction. A family failing to transfer does not fail Gate B; it is a
  genuine, reportable limit on how family-general the uncertainty-error
  relationship is.
- The real-log criterion (E1) supports predictive claims only, on either
  conversational source. WildChat-1M is the primary source (ungated,
  ODC-BY); LMSYS-Chat-1M is the secondary replication source (gated). Labels
  are deterministic rule proxies (versioned, currently `rules-v2`), frozen
  before modeling, computed from future user turns that never enter model
  inputs; trivial length baselines are always reported alongside. Results
  from the two sources are reported side by side and never averaged or
  merged into one number — they are separate replications, not one enlarged
  sample.

## E5 acceptance contract (overoptimization)

- Two failure channels are always reported separately and never conflated:
  **RM exploitation** (learned reward above the observed proxy off the logged
  distribution) and **proxy misalignment** (observed proxy up while true
  utility falls). Ensemble pessimism targets only the former.
- Checkpoints are never selected by proxy reward alone; every checkpoint
  reports learned reward, observed proxy, and true utility side by side, plus
  ensemble uncertainty, KL to the behavior clone, and the bait-action rate.
- The hypothesis-H5 verdict about LCB pessimism is recorded explicitly and may
  be `supported`, `refuted_in_this_environment`, or
  `not_testable_rm_error_channel_inactive` (when the learned reward never
  overestimates the observed proxy, pessimism has nothing to mitigate). A
  negative verdict does not fail the experiment; failing to demonstrate the
  Goodhart effect or any effective mitigation does.
- Logging-support regimes (broad/narrow) are declared in configuration; the
  optimizer interacts with the real world, and only the reward is learned.

## E6 acceptance contract (randomized bridge)

- E6 is the first real (non-simulated) data source licensed for causal claims,
  not just predictive ones: `log_random_*` rows carry a genuinely uniform
  exposure policy, so their logging propensity is known exactly rather than
  estimated. `log_standard_*` rows remain production-confounded, exactly like
  WildChat/LMSYS, and must never be treated as randomized.
- Every trajectory's `logging_policy` metadata field (`"random"` or
  `"standard"`) is set from the source file it came from, never inferred, and
  every metric and claim reports the two populations separately. A model
  trained on `standard` rows is evaluated for off-policy accuracy against
  `random` rows; a claim that mixes the two without saying so is a contract
  violation.
- The engagement outcome is a versioned, deterministic rule over immediate
  interaction signals (`kuairand-rules-v1`), frozen before modeling, exactly
  as E1's `rules-v2` labels are for text — E6 does not claim delayed or
  cross-session outcomes (ADR-008).
- Because the adapter accepts whatever columns the real CSV header provides
  beyond the minimal user/item/time identifiers (ADR-009), the prepared
  dataset's `stats.json` always lists the columns actually consumed, so a
  silent schema drift from the documented KuaiRand release is visible rather
  than assumed away.
- The primary E6 metric is per-video engagement-rate bias: `standard`-log rate
  minus `random`-log rate for videos with enough exposures in both (no model
  is fit; both are direct empirical rates, so there is no train/eval leakage
  to guard against). `confounding_bias_detected` uses **mean absolute** bias,
  not signed mean bias, because a confounded log can be badly miscalibrated
  in both directions at once (over-targeted videos inflated, under-targeted
  videos deflated) with the signed average canceling to near zero. The
  confounded rate is always compared against a trivial constant-mean baseline
  on RMSE, exactly as E1 compares against a trivial length baseline.
- E6 additionally reports a **feature-adjustment** comparison
  (`e6_feature_adjustment`), which is scoped explicitly as a *different,
  narrower* claim than the project's core sequential credit-assignment thesis
  (Gate A/B): KuaiRand impressions are single-step, so there is no delayed
  credit problem to test, only whether a model conditioning on real
  confounders (user/video content features, `RidgeBaseline`, trained on the
  confounded log) predicts better on genuinely random exposures than (a) a
  naive per-video rate and (b) a trivial constant. Only `user_features_pure`
  and `video_features_basic_pure` are used as inputs;
  `video_features_statistic_pure` is never used because its columns
  (`like_cnt`, `play_cnt`, etc.) are aggregated engagement outcomes and would
  leak the label. The primary metric is Brier score on the held-out `random`
  log; `hypothesis_h6b_feature_adjustment_helps` is `supported` only when the
  feature-conditioned model beats **both** baselines by
  `feature_adjustment_improvement_margin`, `partially_supported_...` when it
  beats only the trivial baseline, and `refuted_in_this_environment`
  otherwise — reported honestly regardless of which one the real run lands
  on.
