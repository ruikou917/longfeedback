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
  logging-policy shift); cross-family transfer remains an open item and must
  not be claimed.
- The real-log criterion (E1) supports predictive claims only. Its labels are
  deterministic rule proxies (versioned `rules-v1`), frozen before modeling,
  computed from future user turns that never enter model inputs; trivial
  length baselines are always reported alongside.
