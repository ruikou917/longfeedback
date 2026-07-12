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

## Multi-seed statistical protocol contract (design doc 13.5)

- The `multiseed` experiment reruns the unmodified Gate B and E5 pipelines
  once per seed (five seeds in `configs/experiments/multiseed.yaml`; fewer
  seeds still run but the decision block records
  `protocol_seed_count_met: false`) and never alters what a single-seed run
  computes — each per-seed run keeps its own auditable artifact directory,
  manifest, and scientific-metrics hash.
- Primary metrics are predeclared in the module and echoed in
  `metric_conventions`: Gate B's per-family credit margin (`docm_credit`
  credit Spearman minus the best baseline variant) and E5's per-regime
  hacking gap for the `single` reward plus the paired KL/LCB mitigation
  deltas. All other table entries are secondary and carry no robustness
  verdict of their own.
- Comparisons are paired within a seed (variants of one run share evaluation
  seeds); the percentile bootstrap resamples per-seed paired differences with
  a fixed resampling seed, so the reported intervals are reproducible.
  Effect sizes are reported in native units (Spearman points, utility
  points), never bare p-values.
- A "robust" verdict requires the bootstrap CI to exclude zero in the
  beneficial direction **and** the across-seed mean to meet the same
  threshold the single-seed decision used; the per-family/per-regime rows are
  aggregated with the same `min_winning_families` / any-regime rules as the
  single-seed decisions rather than testing each row in isolation (the
  multiple-comparisons stance is declared in `metric_conventions`).
- Identical configurations and seeds produce identical scientific metrics,
  including the bootstrap intervals.

## E8 acceptance contract (real multi-step credit via randomized session steps)

E8 is the project's attempt to test the core multi-step credit-assignment
claim against *real* interventions: KuaiRand's randomly-inserted videos are
genuine mid-session randomized steps embedded in real user trajectories, so
their causal effect on *later-session* behavior is identifiable without a
model. This contract was frozen before any treatment effect was computed;
only outcome base rates (marginals, not effects) informed the horizon rule.

- **Trajectory definition (ADR-012).** A session is one user's impressions,
  merged across the random and standard logs, sorted by `time_ms`, split at
  gaps > 30 minutes. A randomized step is a row with `is_rand = 1` (already
  verified to match the source-file `logging_policy` tagging exactly).
- **Delayed outcome.** For step `t` at horizon `k`:
  `Y_k(t) = 1` iff the user consumes at least `k` further impressions in the
  same session after `t`. This is pure future behavior — it contains no
  engagement signal from step `t` itself — and it is defined for **every**
  randomized step; conditioning on having subsequent steps would condition
  on the outcome and is a contract violation. The **primary horizon is
  k = 5**, adjusted deterministically (before any effect computation) to the
  nearest of {1, 3, 5, 10} whose base rate over randomized steps lies in
  [0.2, 0.8], as recorded in the prepared dataset's `stats.json`.
- **Primary causal quantity (phase 1, the power gate).** The OLS slope of
  `Y_k` on the z-scored `log1p(duration_ms)` of the randomly-inserted video,
  over all randomized steps. Because the video is assigned uniformly at
  random, this slope is an unconfounded intent-to-treat effect: does the
  content of a single mid-session step causally change how long the session
  survives? Inference is a percentile bootstrap clustered over **users**
  (the design doc's hierarchical bootstrap; impressions within a user are
  never treated as independent), with a fixed resampling seed.
  `power_gate_pass` requires the CI to exclude zero **and**
  |slope| ≥ 0.005 (half a percentage point of survival per standard
  deviation of log-duration — smaller effects are statistically detectable
  here but too weak to grade credit models against).
- **Null protocol.** If the gate fails, E8 reports the estimated slope, its
  CI, and the minimum detectable effect at ~80% power (2.8 × bootstrap SE),
  and records `hypothesis_h8_delayed_step_effect` as
  `refuted_at_this_granularity` — an honest, well-powered null about this
  platform/granularity, not a failure to be tuned away. Phase 2 then must
  not use duration as its grading axis.
- **Secondary, no verdicts of their own:** per-duration-quintile survival
  tables with CIs, the same slope at the non-primary horizons, and
  video-type group contrasts. All secondary rows are reported in full.
- **Phase 2 (model grading, gated on phase 1).** Capacity-matched sequence
  models trained on **observational data only** (standard-log sessions, and
  never any `is_rand = 1` step's assignment mechanism) produce per-step
  credit estimates; these are graded against group-level randomized effects
  on **held-out users'** randomized steps, against attribution baselines
  (uniform, last-touch, immediate-engagement). The claim graded is exactly
  Gate A/B's, transplanted: does the model's credit agree with interventional
  ground truth — here real, not simulated?
- **Scope honesty.** KuaiRand sessions are recommendation trajectories, not
  conversations; a positive E8 validates the method's credit estimates
  against real randomized interventions in *a* real sequential domain, not
  in dialogue. Per-step ground truth for every observational step remains
  unattainable in principle; E8's grading is group-level at randomized steps
  only, and every claim must say so.

## E9 acceptance contract (HeartSteps randomized longitudinal benchmark)

- The target is a group-level causal excursion effect under the known 0.6/0.4
  experimental policy, never an individual counterfactual.
- `avail` defines eligible randomized decisions and `send` defines treatment.
  The source field `is.randomized` is not an inclusion indicator; an observed
  treatment rate far from 0.6 is a hard preprocessing failure.
- The 30-minute log-step effect is a positive-control reproduction. The distal
  outcome is average daily Jawbone steps over the final 35 decision points;
  all earlier eligible decisions receive that delayed participant outcome.
- Nuisance outcome regressions are cross-fitted by participant. Inference
  resamples participants, never decision rows. Travel periods declared in the
  source are excluded following the published preprocessing.
- Distal model grading is authorized only if a predeclared distal excursion
  effect excludes zero. A proximal reproduction alone validates the analysis
  machinery but does not support the project's delayed-credit model claim.

## E10-HS-Day acceptance contract (frozen before effect computation)

- **Trajectory and outcome.** One trajectory is five consecutive scheduled
  HeartSteps decisions for one participant pseudo-day, using the published
  decision-number ordering. Days with other than five recorded decisions are
  excluded before treatment/outcome analysis. The only model outcome is the
  terminal daily activity score: the mean of `log1p(jbsteps30)` over all five
  decisions. Component window outcomes are never model inputs or intermediate
  rewards.
- **Intervention and estimand.** At every available decision, suggestion
  delivery was randomized with known probability 0.6. Credit is the
  intent-to-treat effect of sending versus not sending a suggestion on the
  terminal daily score, under the trial's randomized future continuation.
  Effects are group-level, never individual counterfactuals.
- **Predeclared groups.** Decision position 1--5 crossed with home/work versus
  other context (10 groups). Position is order within the reconstructed day,
  not the source's occasionally missing `decision_slot` field.
- **Inference.** Point estimates are randomized differences in means.
  Percentile bootstrap resamples participants with all their days and
  decisions intact. The global CI is 95%; group discovery uses simultaneous
  Bonferroni 95% familywise coverage across the 10 fixed groups.
- **Phase-1 signal gate.** A relevant detected effect has absolute magnitude
  at least 0.01 terminal-score units and a CI excluding zero. Precision is
  adequate only when the median group bootstrap SE is at most 0.15. Model
  grading is authorized when precision is adequate and either the global
  effect is relevant/detected or at least two fixed groups are
  relevant/detected. If the gate fails, no model is trained and this outcome
  is not redefined.
- **Phase 2.** If authorized, capacity-matched outcome-only, prefix/RUDDER,
  and credit-supervised models are evaluated with participant-level
  cross-fitting. The primary comparison uses a held-out randomized causal
  loss or equivalently precision-weighted held-out group effects; exact
  calibration, seed robustness, and matched-outcome thresholds must be
  committed before any model result is computed.
- **Scope.** A pass supports a domain-general real randomized multi-step
  behavioral claim. It does not establish conversational transfer, long-term
  welfare, or exact unit-level credit.

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
