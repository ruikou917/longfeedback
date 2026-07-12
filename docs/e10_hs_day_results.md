# E10-HS-Day results

Run date: 2026-07-12. Public HeartSteps V1 data, repository revision recorded
in the prepared-data manifest. Phase-1 contract commit: `5642afd`. Phase-2
contract commit: `9b2961d`. First real Phase-2 implementation/run commit:
`49ed092`.

## Phase 1: randomized terminal-signal gate

- 1,517 complete five-decision participant-days, 37 participants, and 6,104
  eligible randomized decisions.
- Send-versus-no-send effect on the terminal daily activity score: `+0.08586`
  (participant-cluster bootstrap 95% CI `+0.00931` to `+0.16296`).
- Median fixed-group bootstrap SE: `0.11500`, below the frozen `0.15`
  precision threshold.
- Decision: **pass**; Phase 2 authorized. No individual position-by-context
  group survived the frozen simultaneous familywise intervals.

## Phase 2: capacity-matched model comparison

All three variants had 68,548 parameters. Evaluation used three held-out
participant folds, five training seeds, and 3,617 held-out randomized sent
decisions. The scientific metrics hash is
`f06d314db89288d1b05b77262fd43b6d32c6b7279aa6d22de4887963d2de891d`;
it was independently recomputed from the artifact and matched.

For the frozen seed-0 primary run:

| Comparison | Paired MSE gap | Participant-bootstrap 95% CI | Decision |
|---|---:|---:|---|
| outcome-only minus credit | +0.6092 | +0.0868 to +1.2579 | pass |
| prefix/RUDDER minus credit | +0.3258 | -0.0244 to +0.7507 | fail |

Credit supervision reduced held-out randomized-target MSE by 11.05% versus
outcome-only and 6.23% versus prefix/RUDDER in seed 0. Both paired gaps were
positive in all five seeds, and no seed had a confidence interval excluding
zero in the adverse direction. The frozen robustness condition therefore
passed for both comparisons.

The terminal-score RMSEs in seed 0 were 1.1879 for credit-supervised and
1.2590 for outcome-only. The credit model was better, but the frozen
matched-quality rule used the absolute difference; `0.0711` exceeded its
`0.05` tolerance, so that precondition failed.

## Frozen verdict

**Phase 2 failed and the confirmatory core real-data claim is not supported
under the predeclared contract.** Two criteria failed: the seed-0
prefix/RUDDER confidence interval narrowly included zero, and the absolute
terminal-RMSE difference exceeded tolerance. Neither threshold may be changed
after observing these results.

The run is strong exploratory evidence: every seed favored explicit credit
supervision against both baselines, and the outcome-only primary comparison
was significant. It is suitable to report as a transparent near-null or
pilot result, but not as the planned confirmatory support for the full core
claim. A confirmatory claim now requires a genuinely untouched randomized
dataset or a new prospective sample with the revised contract frozen before
access; another analysis of these same 37 participants can only be labeled
exploratory or sensitivity analysis.
