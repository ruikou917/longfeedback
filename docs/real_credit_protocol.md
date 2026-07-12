# Real-data causal-credit program

## Claim boundary

Real logs never reveal an individual's paired counterfactual outcome. The
real-data target is therefore a randomization-identified conditional average
effect under an explicit stochastic continuation policy, with uncertainty:

`C_t(h) = E[Y(A_t=a, A_{>t}~pi_0) - Y(A_t=a_ref, A_{>t}~pi_0) | H_t=h]`.

This is group-level interventional credit, not exact per-trajectory credit.
Observational adjustment, human ratings, LLM judges, and semi-synthetic labels
may triangulate it but cannot replace randomized identification.

## Evidence ladder

1. **E8, KuaiRand power gate.** Test whether a randomized content attribute
   moves a later within-session outcome enough to grade models. The frozen
   duration axis produced a well-powered null, so model grading on that axis
   is prohibited.
2. **E9, HeartSteps external benchmark.** The proximal randomized effect
   reproduced at the published scale, validating the pipeline; the distal
   effect remained imprecise, so model grading was not authorized.
   **E9b** (below) grades the model ladder at the proximal horizon, where
   the randomized effect is a replicated published finding.
3. **E10, controlled conversational MRT.** Randomize a small, interpretable
   response-strategy action at eligible turns in real multi-turn tasks with an
   objective delayed terminal score. Log the candidate set, availability,
   exact propensity, assignment, rendered-response hash, and continuation
   policy at every decision.
4. **E11, confirmatory policy trial.** Freeze reference, outcome-only, and
   credit-supervised policies; randomize new participants to policies; test the
   predeclared delayed task outcome. No policy may be selected on the
   confirmatory cohort.

## E9 requirements

- Pin the public HeartSteps revision and preserve CC BY 4.0 attribution.
- Use only available randomized decision points; suggestion probability is
  0.6 and no-suggestion probability is 0.4.
- Reconstruct complete user trajectories and keep users, not rows, inside one
  split or cross-fitting fold.
- Report the established 30-minute proximal effect as a positive-control
  reproduction before analyzing the Week-6 distal outcome.
- Estimate time-varying distal effects with an auditable, cross-fitted
  estimator and user-cluster uncertainty. Compare against naive outcome-only
  and prefix-attribution baselines only after estimator checks pass.
- Treat the 37-participant sample as an external methodological benchmark,
  not a powered high-dimensional personalization study.

## E9b — proximal-horizon model grading (predeclared 2026-07-12, before any new estimate)

E9 validated the estimator pipeline but its distal gate (Week-6 CI excluding
zero) failed, so no model was graded. E9b is the predeclared follow-on that
grades the capacity-matched ladder at the horizon where HeartSteps has
replicated randomized ground truth: the 30-minute proximal effect and its
published early-study concentration. Everything below is fixed before the
gate estimate, any model training, or any grading number is computed.

- **Estimand.** The randomization-identified proximal excursion effect
  `tau(h) = E[log1p(steps30) | A=1, h] - E[log1p(steps30) | A=0, h]` at
  available randomized decisions, under the known 0.6/0.4 propensities.
  Following the repository credit convention (Gate A), the model target at a
  logged decision is the credit of the logged action versus the reference
  (no-suggestion) action, so randomized targets are informative only at
  suggestion-sent decisions; reference decisions carry zero logged credit by
  definition and are excluded from grading.
- **Gate (grading authorization).** Compute the covariate-adjusted,
  participant-cross-fitted orthogonal-score estimate of the average proximal
  effect over all available randomized decisions, and the same estimate
  restricted to the first study-day quartile, each with 95% user-cluster
  bootstrap CIs. Grading is authorized if either CI excludes zero (the
  published effect concentrates early, so the quartile-1 arm is the
  scientifically motivated disjunct). If neither excludes zero, E9b stops and
  is reported as a null; no model is trained.
- **Episodes.** One episode per participant pseudo-day: 5 consecutive
  decision slots ordered by decision number (days with a different slot count
  are dropped; descriptively 1,517 of 1,525). Per-slot observations:
  availability flag, log1p prior 30-minute steps, home/work flag, scaled
  study day, slot index. Action: suggestion sent (reference 0); non-eligible
  slots are forced to the reference action with availability 0. Response
  channel: log1p proximal 30-minute steps (post-action logger output).
  Delayed episode outcome: binary, day total of `steps_until_next_decision`
  above the training-fold pooled median (threshold never fit on held-out
  participants).
- **Ladder.** The existing capacity-matched DOCM variants `docm_outcome`,
  `docm_prefix`, `docm_credit` with identical architecture (d_model 64,
  2 layers, 4 heads) and identical training budget; parameter counts must
  match exactly.
- **Credit supervision (training folds only).** At eligible suggestion-sent
  decisions, the doubly-robust pseudo-outcome
  `xi = (Y - (1-p)*mu1 - p*mu0) / p` with arm regressions `mu` cross-fitted
  across training participants only (same ridge features as E9). `xi` is
  centered at `tau(h)` under randomization; held-out participants never
  contribute to any supervision signal.
- **Cross-evaluation.** Three participant-level folds by the E9 SHA-256 user
  hash; each fold is held out once with models trained on the other two;
  grading pools held-out predictions across folds.
- **Grading (held-out, eligible suggestion-sent decisions).**
  - Precondition: matched outcome quality, |AUROC(docm_credit) −
    AUROC(docm_outcome)| ≤ 0.05 on the day outcome.
  - Primary: per-decision squared error against held-out `xi` after a single
    affine calibration per variant fitted on training-fold `xi` (paired
    comparison; the shared `xi` noise cancels in expectation, so the paired
    difference estimates the difference in squared distance to `tau`).
    Decision: mean paired gap (docm_outcome error − docm_credit error) > 0
    with 95% user-cluster bootstrap CI excluding zero.
  - Secondary (descriptive, no gating): study-day-quartile profile of
    bin-mean predicted credit against held-out orthogonal-score bin
    estimates; attenuation tracking (sign of Spearman between predicted
    credit and study day compared with the randomized time trend, reported
    only if the held-out trend CI excludes zero); `docm_prefix` reported
    alongside.
- **Robustness.** Five training seeds (0–4) with fixed folds. The claim is
  supported only if the seed-0 primary CI excludes zero, at least 4 of 5
  seeds have a positive gap, and no seed's CI excludes zero in the negative
  direction.
- **Scope.** 37 participants; a mobile-health methodological benchmark. A
  pass supports the core claim at the estimation level on real randomized
  multi-step trajectories at the proximal horizon; it does not transfer the
  claim to the conversational domain (E10) or to the decision level (E11).

## E10 design constraints

- Randomize among safe strategy categories, never arbitrary generated strings:
  direct answer versus guided hint, clarification versus best effort,
  concise versus explanatory, verification prompt versus no prompt, or
  checkpoint summary versus ordinary continuation.
- Use approximately 6–12 eligible decision points per trajectory and keep
  assignment probabilities bounded away from zero and one.
- Primary outcome: objectively scored task success, held-out quiz performance,
  or transfer accuracy observed only after the multi-turn interaction.
- Engagement, length, satisfaction, and return are secondary behavioral
  proxies; none is called welfare.
- Run a technical pilot, then a variance pilot, then simulation-based power
  analysis before the confirmatory sample. Obtain the appropriate ethics/IRB
  determination and informed consent before recruitment.

## E10 operational plan (blocked on a user decision, like E7)

E10 spends real money on recruitment and API calls, so implementation waits
for explicit sign-off on the items below; everything else about it is
specified here so the decision is about resources, not design.

- **Stage ladder.** (1) Technical pilot, ~20 conversations, own-team or free
  participants: verifies logging of candidate set, availability, propensity,
  assignment, rendered-response hash, and continuation policy at every
  decision point. (2) Variance pilot, ~100 paid conversations: estimates
  outcome variance and intra-participant clustering, no hypothesis tests.
  (3) Simulation-based power analysis from the pilot variance. (4)
  Confirmatory sample at the powered size (plausibly 500–1,000 conversations
  at 6–12 eligible decisions each; the pilot decides).
- **Cost envelope.** Participant payments (Prolific-class platform) plus LLM
  API costs, roughly $2k–6k total depending on the powered sample size;
  refined after the variance pilot.
- **User decisions required before stage 2:** budget ceiling; which single
  strategy dimension to randomize (from the safe categories above); the
  objective delayed outcome task (scored task success, held-out quiz, or
  transfer accuracy); model/provider for the assistant; ethics determination
  and consent text.
- **Frozen before the confirmatory stage:** the primary outcome, the
  estimator, the horizon(s), and the model-grading criterion — same
  contract-first discipline as E8/E9; pilot data never enters confirmatory
  analysis.

## E11 requirements

- E11 runs only after E10 supports the credit claim at the estimation level;
  it tests the decision-level claim (does credit-informed training improve
  the delayed outcome?), which estimation-level agreement does not imply.
- Reference, outcome-only, and credit-supervised policies are trained on E10
  data and **frozen** — weights, prompts, and decoding parameters hashed —
  before any confirmatory participant is recruited.
- New participants are randomized to policy arms with logged assignment;
  nobody analyzing outcomes selects or tunes policies on the confirmatory
  cohort, and interim looks follow a predeclared spending rule or do not
  happen.
- One predeclared primary outcome (the same objective delayed task outcome
  as E10), powered from E10's measured variance; behavioral engagement
  metrics are secondary and are never promoted after the fact.
- A registered analysis plan (estimator, covariates, exclusion rules) is
  committed to the repository before recruitment; deviations are reported as
  deviations.

## Claim strength by rung

What may honestly be claimed as each rung completes: after E9 (done), the
estimator pipeline is validated against a real repeated-randomized positive
control, but the core claim is still unsupported on real data. After a
successful E10, the claim "credit-supervised models recover randomized
per-decision effects in real conversations better than outcome-only/prefix
baselines" is supported at the estimation level. After a successful E11, the
decision-level claim "credit supervision improves delayed outcomes" is
supported. A null at any rung is reported as a null and bounds the claim
rather than being re-run away.

## Success criterion

The core claim is supported on real data only if (1) a randomized benchmark
has a detectable delayed action effect, (2) at matched outcome-prediction
quality the credit-supervised model has lower precision-weighted error against
held-out randomized effects than outcome-only and prefix baselines, and (3) a
fresh randomized policy trial improves the delayed objective outcome. Nulls
remain nulls; changing an axis after seeing its effect requires a new,
explicitly exploratory experiment.
