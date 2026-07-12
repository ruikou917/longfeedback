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

## Success criterion

The core claim is supported on real data only if (1) a randomized benchmark
has a detectable delayed action effect, (2) at matched outcome-prediction
quality the credit-supervised model has lower precision-weighted error against
held-out randomized effects than outcome-only and prefix baselines, and (3) a
fresh randomized policy trial improves the delayed objective outcome. Nulls
remain nulls; changing an axis after seeing its effect requires a new,
explicitly exploratory experiment.
