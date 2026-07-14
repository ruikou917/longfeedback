# LinkedIn new-member 28-day onboarding log audit

Status: standalone implementation specification draft v0.1  
Date: 2026-07-14  
Owner: LongFeedback project  
Scope: historical onboarding email decisions and day-28 member activity

This document is written for a separate implementation/audit agent that will receive access
to real internal logs. It defines the scientific boundary, minimum source contract,
canonical schemas, privacy constraints, audit computations, decision gates, artifacts,
tests, and execution order for determining whether historical onboarding logs can support
causal credit-target construction.

This audit is **independent of ALFWorld**. It requires no ALFWorld installation, model,
trajectory, result, or artifact. ALFWorld tests whether LongFeedback works when credit can be
measured by replay. This audit asks whether particular onboarding-email contrasts are
supported by the production logging process. Neither result substitutes for the other.

## 1. Objective and non-objectives

### 1.1 Objective

For each proposed action contrast, determine whether the logs adequately record:

1. **eligibility/availability**: whether each action could actually have been chosen;
2. **candidate actions**: the complete choice set presented to the historical policy;
3. **propensity**: the probability or assignment mechanism that selected the logged action;
4. **overlap/positivity**: whether comparable histories contain both sides of the contrast;
5. **assignment-feature completeness**: whether all important drivers of action assignment
   and the day-28 outcome appear to be recorded before the decision; and
6. **censoring/outcome observability**: whether day-28 outcomes are observable under a
   defensible missingness design.

The unit of the verdict is an explicit contrast in an explicit target population, for
example:

```text
eligible send_any_email versus eligible no_email
on days 1-7
among members onboarded under policy version P
with future decisions following the recorded production continuation policy
```

The audit does not issue a single blanket verdict such as “the dataset has overlap.” One
email family may be supportable while another is not.

### 1.2 Non-objectives

This task must not:

- train LongFeedback or any credit-assignment model;
- estimate email treatment effects;
- compare day-28 activity rates by action;
- learn or deploy an onboarding policy;
- use clicks or opens as causal proof;
- infer that an estimated propensity removes hidden confounding;
- claim individual-level causal credit;
- upload raw or derived member-level data to an external service;
- depend on ALFWorld; or
- select the easiest action contrast after examining outcome effects.

The outcome may be read only to audit existence, timing, censoring, type, and missingness.
Action-specific outcome means, treatment effects, and model results are prohibited until the
audit contract and eligible contrasts are frozen.

## 2. Scientific boundary

The later modeling target, not estimated in this audit, is a conditional average excursion
effect under a named continuation policy:

```text
C_t(h,a;a_ref,pi_0)
  = E[Y_28(do(A_t=a), A_{>t}~pi_0)
      - Y_28(do(A_t=a_ref), A_{>t}~pi_0) | H_t=h]
```

`H_t` contains only information available before decision `t`. The default reference is
`NO_EMAIL`, but it is a legitimate reference only at decision points where an email could
have been sent. A structurally ineligible row is not an untreated control.

The permitted interpretation depends on the audit result:

| Verdict | Meaning | Permitted next step |
|---|---|---|
| `randomized_ready` | Exact stochastic assignment and adequate overlap are verified | Randomization-identified credit targets |
| `observational_ready` | Estimated/reconstructed assignment, adequate overlap, and a defensible measured-confounding argument | Observational doubly robust targets with explicit assumptions |
| `restricted_ready` | Only a restricted cohort, period, action grouping, or day range has support | Model only the frozen restricted estimand |
| `predictive_only` | Outcome prediction is possible, but causal credit is not identified | Predictive contribution only; no causal labels |
| `invalid` | Data integrity or temporal ordering is unreliable | Repair logging/extraction before modeling |

“Doubly robust” protects against specified nuisance-model errors under identification
assumptions. It does not repair unmeasured confounding or absent action support. The project’s
[causal assumptions](./causal_assumptions.md) and
[real-data causal-credit protocol](./real_credit_protocol.md) remain controlling.

## 3. Instructions for the executing agent

The executing agent must follow these rules before accessing real logs:

1. Treat every source as read-only.
2. Do not use network tools, web search, external APIs, external LLMs, hosted notebooks, or
   telemetry-bearing experiment trackers while real data is in scope.
3. Do not print raw rows, member IDs, email addresses, names, free text, subject lines,
   template bodies, URLs, or rare attribute combinations to the terminal or report.
4. Do not call `head()`, `sample()`, or equivalent row-display operations on sensitive
   tables. Inspect schemas, types, null counts, hashes, and aggregates instead.
5. Use a project-secret HMAC for member join keys. An ordinary unsalted hash of a stable
   member ID is not sufficient.
6. Keep raw and row-level derivative artifacts in an approved access-controlled,
   Git-ignored location. Never stage or commit them.
7. Suppress report cells smaller than the configured privacy threshold; default `k=20`.
8. Replace email content with approved template/family/version identifiers before canonical
   data leave the secure source boundary.
9. Record every transformation, exclusion, and source revision in a manifest.
10. If a field or business rule is unknown, mark it `unknown`; do not infer a favorable
    answer from observed frequencies.
11. Continue safe audit work when possible, but a missing fact must remain visible in the
    final readiness matrix.
12. Do not compute outcome-by-action comparisons even if requested by an intermediate
    notebook cell. That is a later, separately authorized task.

The existing [data-governance checklist](./data_governance.md) applies in addition to these
rules.

## 4. Decisions to freeze before outcome analysis

The audit agent should obtain or document the absence of answers to the following. Missing
answers do not authorize guessing.

### 4.1 Cohort and endpoint

- What event defines onboarding time zero?
- Which onboarding cohorts and calendar dates are in scope?
- Is the terminal outcome status on day 28, activity during days 22-28, a cumulative
  threshold across days 1-28, or another exact rule?
- What timezone defines onboarding days?
- How are members with account closure, deletion, fraud removal, delayed logging, or missing
  day-28 data handled?
- Can a member enter the cohort more than once?

### 4.2 Decision point

- Is one decision point a day, a scheduled email opportunity, or each send-system call?
- Can multiple onboarding emails be considered or sent in one day?
- At what timestamp is eligibility evaluated?
- What action does the system take when no candidate is available?
- Is treatment the policy assignment/send attempt (recommended intent-to-treat definition),
  successful delivery, or another event? Do not use open/click as treatment.
- Are manual sends, retries, transactional emails, and operational notices excluded?

### 4.3 Action taxonomy

- Is the primary contrast `SEND_ANY` versus `NO_EMAIL`?
- Which email families are scientifically meaningful and operationally stable?
- Are template revisions equivalent actions or separate versions?
- Are channel, cadence, language, and content bundled into one action or separate factors?
- Which action is the reference?

Begin with the coarsest useful binary action. A multiclass audit is authorized only after
the binary send/no-send contrast is understood.

### 4.4 Historical assignment

- Which service, rule engine, experiment, model, or person selected the action?
- Was a candidate set materialized before selection?
- Were assignment probabilities logged?
- Were randomization units member, decision, campaign, or cohort?
- Which policy/model versions operated during the requested period?
- Were there deterministic suppression, priority, quota, or budget rules?
- Did assignment use features or scores that are absent from the extract?
- Could downstream systems override or cancel the selected action?

### 4.5 Concurrent interventions

- Which in-app nudges, push notifications, SMS messages, ads, sales contacts, or product
  experiments affected the same members?
- Were these assigned using variables related to both email selection and day-28 activity?
- Are their assignments and timing recorded?

These facts determine whether “email action” is a well-defined intervention or one component
of an inseparable onboarding policy bundle.

## 5. Minimum source inventory

The audit should accept multiple source tables through an explicit column-mapping
configuration. Do not require one pre-joined file.

| Source concept | Minimum fields | Why required |
|---|---|---|
| Member/cohort | stable member key, onboarding timestamp, cohort attributes | trajectory construction and splitting |
| Decision log | decision ID, member key, decision timestamp, policy/version | unique ordered decision points |
| Eligibility | decision ID, eligible flag, reason/version | distinguishes reference actions from structural unavailability |
| Candidate set | decision ID, action IDs considered | defines the action support at that state |
| Assignment | selected action, selected probability or full probability vector when logged | treatment and propensity |
| Pre-action features | feature snapshot or reconstructable event history with timestamps | confounding and propensity audit |
| Delivery/reaction | attempted, delivered, bounced, opened, clicked, later visit timestamps | post-action state construction only |
| Other interventions | timestamped action/channel identifiers | concurrent-treatment audit |
| Day-28 endpoint | member key, outcome cutoff, outcome value, observation/censoring status | endpoint observability audit |
| Policy documentation | rule/model revision, required inputs, experiment metadata | propensity and hidden-feature verification |

If the historical candidate set was not logged, the agent may audit a deterministic
reconstruction only when the exact versioned eligibility and generation rules plus all their
inputs are available. Mark reconstructed sets separately from exact logged sets.

`NO_EMAIL` may be an implicit residual action rather than a materialized candidate. It may be
added as `RECONSTRUCTED_EXACT` only when documentation proves when it was available and how
its probability was computed (for example, one minus the sum of email-action probabilities).
Otherwise its candidate/propensity status is `UNKNOWN`.

## 6. Canonical data contracts

Use Parquet for row-level secure artifacts and JSON/YAML for small manifests and aggregate
reports. Pydantic boundary models should reject unknown enum values unless the value is
explicitly mapped to `UNKNOWN` with a counted warning.

### 6.1 `members.parquet`

```text
member_key_hmac
onboarding_timestamp_utc
cohort_week
cohort_policy_family
outcome_cutoff_timestamp_utc
outcome_observed
censoring_reason
day28_outcome
source_revision
```

`day28_outcome` must not be joined into the outcome-blind audit frames until endpoint
existence and censoring are checked. Reports may include overall outcome prevalence only
after contrast selection is frozen; they may not stratify it by action during this task.

### 6.2 `decisions.parquet`

One row per true policy decision:

```text
member_key_hmac
decision_id_hmac
decision_timestamp_utc
onboarding_day_index
within_day_slot
policy_id
policy_revision
experiment_id_hmac
experiment_arm_code
eligible
eligibility_reason_code
eligibility_rule_revision
candidate_set_id
selected_action_id
selected_action_probability_logged
assignment_source
manual_override
delivery_attempted
delivery_status_code
state_snapshot_timestamp_utc
source_revision
```

`assignment_source` is one of:

```text
RANDOMIZED_EXACT
POLICY_STOCHASTIC_EXACT
POLICY_DETERMINISTIC
RECONSTRUCTED_EXACT
ESTIMATED
UNKNOWN
MANUAL
```

An assignment can have an exact propensity of one and still fail overlap.

### 6.3 `candidate_actions.parquet`

One row per available action at a decision:

```text
decision_id_hmac
action_id
action_family
template_revision
is_reference_action
candidate_source
candidate_probability_logged
candidate_generation_rule_revision
```

`candidate_source` is `LOGGED_EXACT`, `RECONSTRUCTED_EXACT`, or `UNKNOWN`. Do not fabricate a
candidate row merely because that action appeared for another member.

### 6.4 `pre_action_features.parquet`

One row per decision with approved, temporally valid feature columns:

```text
decision_id_hmac
feature_snapshot_timestamp_utc
baseline_member_features...
prior_product_activity_features...
prior_email_assignment_features...
prior_delivery_open_click_features...
prior_other_intervention_features...
calendar_and_cohort_features...
feature_schema_revision
```

No feature may use events at or after `decision_timestamp_utc`. For windows such as
“activity in prior 24 hours,” record the window endpoint and verify it is strictly
pre-decision.

### 6.5 `propensity_vectors.parquet`

One row per decision/action probability, preserving the source:

```text
decision_id_hmac
action_id
propensity
propensity_source
propensity_model_id
crossfit_fold
calibration_revision
```

Logged/reconstructed and estimated probabilities must never be silently mixed in one
quality category.

### 6.6 `contrast_readiness.parquet`

One row per proposed action contrast and target slice:

```text
contrast_id
target_action_id
reference_action_id
target_population_rule
day_range
policy_revisions_json
eligible_decisions
target_action_count
reference_action_count
candidate_set_coverage
propensity_quality
common_support_fraction
target_action_ess
reference_action_ess
largest_normalized_weight
assignment_feature_completeness
concurrent_intervention_status
censoring_status
verdict
allowed_claim
restriction_rule
failure_reasons_json
```

## 7. Suggested strict configuration

```yaml
name: linkedin_onboarding_28d_log_audit
seed: 0
output_dir: REQUIRED_SECURE_LOCAL_PATH

privacy:
  network_access_allowed: false
  print_raw_rows: false
  small_cell_threshold: 20
  member_key_method: hmac_sha256
  hmac_secret_env: REQUIRED_SECRET_ENV_NAME
  publishable_artifacts: aggregate_only

cohort:
  onboarding_start: REQUIRED
  onboarding_end: REQUIRED
  timezone: REQUIRED
  time_zero_definition: REQUIRED
  horizon_days: 28
  member_reentry_rule: REQUIRED

outcome:
  definition: REQUIRED
  value_type: binary
  cutoff_rule: REQUIRED
  censoring_rule: REQUIRED

decision:
  grain: REQUIRED
  eligible_definition: REQUIRED
  no_candidate_behavior: REQUIRED
  max_decisions_per_member: REQUIRED

actions:
  reference_action: NO_EMAIL
  primary_mapping:
    NO_EMAIL: [REQUIRED_SOURCE_VALUES]
    SEND_ANY: [REQUIRED_SOURCE_VALUES]
  secondary_mappings: []

propensity:
  source_priority: [logged, reconstructed, estimated]
  allow_estimation: true
  crossfit_folds: 5
  split_unit: member
  temporal_holdout: true
  probability_floor_for_diagnostics: 0.01
  strict_common_support_floor: 0.05

overlap:
  minimum_raw_count_per_arm: 1000
  minimum_members_per_arm: 500
  minimum_overall_ess_per_arm: 500
  minimum_stratum_ess_per_arm: 100
  minimum_common_support_fraction: 0.95
  maximum_largest_normalized_weight: 0.02
  required_strata: [day_band, cohort_month, policy_revision]

report:
  outcome_blind: true
  include_raw_examples: false
  include_aggregate_plots: true
```

All `REQUIRED` values must be resolved before a final readiness verdict. The agent may run a
schema/inventory pass with unresolved values but must label the result incomplete.

The numerical overlap thresholds are proposed screening defaults, not universal causal
laws. They may be changed after inspecting action/propensity distributions but before any
outcome-by-action analysis. Every change must create a new configuration revision and be
reported.

## 8. Audit A: source, schema, and chronology

### 8.1 Source manifest

For every source table, record:

- approved storage location alias, never a credential-bearing URI;
- table/snapshot/version;
- extraction query hash or file checksum;
- row count and schema hash;
- minimum and maximum timestamps;
- timezone assumptions;
- data owner and access classification;
- extraction date; and
- fields actually consumed.

Do not put SQL text containing sensitive literals into publishable artifacts.

### 8.2 Key integrity

Verify:

- member keys are nonmissing before HMAC conversion;
- decision IDs are unique or have a documented deterministic deduplication key;
- joins do not multiply decision rows;
- every candidate row maps to one decision;
- every outcome maps to one cohort entry under the reentry rule; and
- all exclusions have mutually exclusive reason codes.

Report counts before and after every join and exclusion.

### 8.3 Temporal integrity

Required ordering:

```text
onboarding_time
  <= pre_action_feature_window_end
  < decision_time
  <= delivery_attempt_time
  <= reaction_time
  < next_decision_time
  <= outcome_cutoff_time
```

Some events may be absent, but observed events may not violate ordering. Quantify clock
skew, late-arriving events, duplicate delivery attempts, and reaction events without a
corresponding decision. Do not “fix” timestamps by moving events across the decision boundary.

### 8.4 Endpoint observability

Report overall missingness and censoring by cohort time, geography bucket if approved,
platform, and policy revision. Do not stratify endpoint values by selected email action.
Determine whether censoring could depend on prior actions and state; if so, a later model
requires a separate censoring mechanism.

## 9. Audit B: eligibility and availability

Eligibility is a pre-action property. Audit the rule separately from whether an email was
ultimately sent.

Required checks:

1. Missing eligibility rate by source and policy revision.
2. Eligibility rule/version coverage over calendar time.
3. Selected email while `eligible=false`.
4. `NO_EMAIL` rows split into:
   - eligible but no email selected;
   - ineligible;
   - no candidate generated;
   - suppression/quota/budget;
   - send selected but delivery cancelled;
   - unknown.
5. Eligibility reason distribution with small-cell suppression.
6. Consistency between eligibility, suppression, candidate set, and action.
7. Whether eligibility was recomputed after observing post-action information.
8. Whether manual overrides bypassed normal eligibility.

The primary analytic population contains only decisions at which both actions in the
contrast were eligible candidates. Do not use ineligible `NO_EMAIL` rows as controls.

Hard failures:

- eligibility cannot be placed before selection in time;
- sent actions systematically occur outside recorded eligibility;
- the eligible/no-candidate/no-send states cannot be distinguished; or
- eligibility depended on unavailable inputs that cannot be reconstructed or declared.

These failures may be repaired by a new extraction; they must not be solved by model tuning.

## 10. Audit C: candidate-action sets

### 10.1 Canonicalization

Map source values to stable action families through a versioned table. Keep the source action
code privately for reconciliation, but expose only canonical IDs to downstream modeling.

When several source actions are coarsened into one family, its propensity is the sum of the
mutually exclusive source-action propensities that were present in that decision's candidate
set. Preserve the component list and test the sum; never fit or copy one component's
probability as the family propensity.

Recommended first taxonomy:

```text
NO_EMAIL
SEND_ANY_ONBOARDING_EMAIL
```

Possible later taxonomy, only if supported:

```text
NO_EMAIL
PROFILE_COMPLETION
CONNECTION_DISCOVERY
CONTENT_DISCOVERY
APP_INSTALL
OTHER_ONBOARDING
```

Transactional, security, legal, password-reset, and operational email should normally be
excluded rather than treated as interchangeable onboarding actions.

### 10.2 Required diagnostics

- Fraction of decisions with an exact logged candidate set.
- Fraction with exact deterministic reconstruction.
- Fraction unknown.
- Fraction where the selected action is absent from the candidate set.
- Candidate-set size distribution by policy revision and onboarding day.
- Pairwise co-availability matrix for action families.
- Action and template revision churn over time.
- Candidate generation changes near cohort boundaries.
- Manual actions that did not pass through the normal candidate system.

The selected action must belong to the recorded/reconstructed candidate set. Exceptions are
hard errors unless a documented downstream override expands the set and is incorporated as
part of the assignment mechanism.

For an implicit `NO_EMAIL`, “belongs to the set” means its availability and residual
probability were reconstructed exactly under the rule in Section 5.

### 10.3 Contrast formation

Generate candidate contrasts before reading action-specific outcomes. For each target action
`a` and reference `b`, restrict to decisions where both appear in the same verified candidate
set. Record the restriction as part of the estimand.

Do not claim support for `a versus b` merely because both occur somewhere in the dataset.

## 11. Audit D: propensity quality

### 11.1 Quality hierarchy

Assign the strongest justified category, never the most convenient one:

1. `RANDOMIZED_EXACT`: documented random assignment and exact probability.
2. `POLICY_STOCHASTIC_EXACT`: production stochastic policy and exact probability vector.
3. `RECONSTRUCTED_EXACT`: versioned deterministic/stochastic logic and all inputs reproduce
   the assignment probability exactly.
4. `ESTIMATED`: probability learned from observed pre-action history.
5. `POLICY_DETERMINISTIC`: probability is zero/one under the historical policy.
6. `MANUAL` or `UNKNOWN`: mechanism cannot be represented defensibly.

Exact does not imply randomized, unconfounded, or overlapping. A propensity of `1.0` is
exact but provides no counterfactual support.

### 11.2 Logged-propensity checks

When a full vector is logged:

- all probabilities are finite and in `[0,1]`;
- probabilities sum to one within `1e-6` after documented rounding;
- the selected action has positive probability;
- actions outside the candidate set have zero/absent probability;
- deterministic seeds or assignment records reproduce a sample where possible;
- observed assignment frequencies agree with logged probabilities in aggregate calibration
  bins and within major policy revisions; and
- experiment-arm probabilities match experiment documentation.

Calibration disagreement is a logging/reconstruction alarm, not permission to replace exact
probabilities automatically with an estimated model.

### 11.3 Reconstructed propensities

Reconstruction requires:

- exact policy/rule revision;
- complete pre-action input snapshot;
- candidate ordering and tie-breaking;
- random seed or probability calculation;
- quota/budget state if used;
- downstream overrides; and
- a reproducibility audit against logged selections.

If quota, priority, or randomization state is global and unavailable, do not label the result
exact.

### 11.4 Estimated propensities

Estimated propensity models are permitted only for an observational-readiness assessment.

Rules:

- split by member so all 28 days stay in one fold;
- include a temporal holdout or rolling policy-version check;
- use only pre-action features;
- include candidate-set masks so impossible actions receive no probability;
- fit multiclass probabilities when the action taxonomy is multiclass;
- calibrate using training/development folds only;
- save out-of-fold probabilities for every audited decision;
- report Brier/log loss, reliability curves, and calibration by action/version/day;
- report feature availability and missingness; and
- list known assignment inputs that are absent from the extract.

High action-prediction accuracy may signal deterministic assignment and poor overlap. Low
error does not prove no unmeasured confounding. Propensity prediction is a diagnostic and
nuisance-estimation task, not causal validation.

### 11.5 Post-action leakage denylist

The propensity model must reject:

- delivery success for the current decision;
- current-email open/click;
- post-send product activity;
- future email assignments;
- the day-28 outcome;
- fields backfilled after the decision; and
- aggregates whose windows extend past the decision timestamp.

Prior-email reactions may be used if their timestamps strictly precede the current decision.

## 12. Audit E: overlap and positivity

Run overlap diagnostics separately for every contrast, day range, policy revision, and
target population.

### 12.1 Basic support

Within decisions where both actions were candidates, report:

- raw counts and member counts per arm;
- action rate by onboarding day and early/middle/late bands;
- action rate by cohort month and policy revision;
- propensity quantiles (`0, 0.1%, 1%, 5%, 25%, 50%, 75%, 95%, 99%, 99.9%, 100%`);
- common-support fraction at floors `0.01`, `0.02`, and `0.05`;
- fraction with propensity above `0.95`, `0.98`, and `0.99`;
- candidate co-availability; and
- support under important predeclared state strata.

Do not create hundreds of post hoc state subgroups. Required strata should be limited to
known assignment dimensions and scientifically important effect modifiers.

### 12.2 Effective sample size

For inverse-propensity weights `w_i` in each arm:

```text
ESS = (sum_i w_i)^2 / sum_i w_i^2
```

Report ESS separately for target and reference actions, overall and by required stratum.
Also report:

- largest normalized weight `max_i w_i / sum_i w_i`;
- top 1% share of total weight;
- member-level weight concentration;
- weight distribution before any truncation; and
- sensitivity under predeclared truncation levels.

Do not hide poor overlap by reporting only truncated weights. Truncation changes the
bias/variance tradeoff and may effectively change the target population.

### 12.3 Conditional support

Marginal send/no-send counts can look excellent while conditional overlap is absent. Add:

- propensity-density overlap plots by arm;
- standardized covariate differences before and after weighting;
- classifier-based separability of actions from pre-action history;
- nearest-neighbor or representation-distance diagnostics within the verified candidate
  population; and
- policy-version-specific support maps.

These diagnostics identify unsupported regions; they do not prove exchangeability.

### 12.4 Proposed screening gates

A contrast is numerically eligible for `randomized_ready` or `observational_ready` only if:

1. each arm has at least 1,000 decisions and the configured minimum distinct members;
2. overall ESS is at least 500 per arm;
3. early/middle/late and each retained policy revision have ESS at least 100 per arm;
4. at least 95% of the target population has both action propensities at least `0.01`;
5. the strict-support sensitivity at `0.05` is reported;
6. no one member contributes more than 2% of normalized weight; and
7. no retained policy revision contains a deterministic arm for the proposed contrast.

These gates screen obvious failures. Passing them does not establish that the causal effect
is identified. If a threshold fails, the allowed responses are to restrict the target
population prospectively, coarsen the action taxonomy, declare `predictive_only`, or collect
new randomized data. Do not change thresholds after examining treatment effects.

## 13. Audit F: measured-confounding and intervention audit

This section is a structured domain review, not a statistical test.

Create `assignment_feature_inventory.parquet` with:

```text
assignment_driver
used_by_policy_revision
available_in_extract
snapshot_is_pre_action
missing_rate
related_to_day28_activity
status
owner_confirmation
```

At minimum review:

- member baseline activity and acquisition channel;
- locale/language and platform;
- account/profile completeness;
- prior sessions and product activity;
- prior emails, delivery, opens, and clicks;
- fatigue, suppression, unsubscribe, and notification preferences;
- experimentation assignments;
- marketing eligibility and legal restrictions;
- model scores used for targeting;
- quotas, budgets, campaign priorities, and send-time optimization;
- manual intervention;
- concurrent push/in-app/SMS/product treatments; and
- fraud, trust, account-quality, or risk filters.

Verdict rules:

- If an important common cause of action and day-28 activity was used by assignment but is
  unavailable, the contrast is not `observational_ready`.
- If the missing driver is recoverable from a versioned snapshot, rerun the extraction.
- If assignment was randomized independently of member potential outcomes within a verified
  eligible set, the driver inventory documents rather than replaces that randomization.
- If email is inseparable from a bundled campaign, redefine the action as the bundle or
  declare the email-only effect unsupported.

Optional negative-control and placebo checks may be proposed for the later causal-analysis
stage, but they do not rescue known missing confounders in this audit.

## 14. Censoring and missing data readiness

Day-28 activity may be missing because of deletion, privacy retention, delayed pipelines,
account restriction, or extraction boundaries. Audit:

- complete 28-day follow-up availability by onboarding cohort;
- administrative right-censoring near the extract end date;
- censoring timestamps and reason coverage;
- feature and reaction missingness by policy revision and platform;
- whether censoring can occur after treatment and depend on state;
- whether “no activity” is distinguishable from “activity unavailable”; and
- whether deletion requirements propagate to derivative artifacts.

Administrative censoring is handled by excluding cohorts without complete follow-up before
action/outcome analysis. Potentially informative censoring requires a later censoring model
and must be marked in `contrast_readiness.parquet`. Never convert missing outcomes to
inactive by default.

## 15. Readiness decision algorithm

For each contrast:

```text
if schema/key/chronology is unreliable:
    verdict = invalid
elif eligibility or candidate set is unknown:
    verdict = predictive_only
elif overlap gate fails:
    verdict = restricted_ready if a pre-outcome restriction passes
              else predictive_only
elif verified sequential randomization with exact propensities:
    verdict = randomized_ready
elif propensity is exact/estimated and all important assignment drivers are observed:
    verdict = observational_ready
else:
    verdict = predictive_only
```

`observational_ready` must carry this claim language:

> The logs support observationally adjusted, group-level excursion-effect estimation for
> the stated contrast under consistency, positivity, sequential exchangeability conditional
> on the recorded history, correct temporal ordering, and the stated censoring assumptions.

It must not be shortened to “the logs reveal causal credit.”

## 16. Required artifacts

Secure row-level artifacts:

```text
resolved_audit_config.yaml
source_manifest.json
members.parquet
decisions.parquet
candidate_actions.parquet
pre_action_features.parquet
propensity_vectors.parquet
fold_assignments.parquet
contrast_readiness.parquet
assignment_feature_inventory.parquet
exclusions.parquet
```

Aggregate safe artifacts:

```text
audit_report.md
audit_metrics.json
audit_manifest.json
schema_audit.json
chronology_audit.json
eligibility_audit.json
candidate_action_audit.json
propensity_audit.json
overlap_audit.json
censoring_audit.json
approved_contrasts.yaml
plots/action_rates_by_day.png
plots/propensity_overlap_<contrast>.png
plots/ess_by_day_<contrast>.png
plots/candidate_coavailability.png
plots/policy_revision_timeline.png
```

The aggregate report must suppress small cells and contain no raw IDs, free text, template
content, or member-level rows.

## 17. Required `audit_report.md` structure

1. **Executive verdict**: one paragraph, including whether any contrast is ready.
2. **Scope**: cohort dates, time zero, horizon, decision grain, and outcome definition.
3. **Source manifest summary**: revisions, row counts, and timestamp coverage.
4. **Data integrity**: joins, duplicates, exclusions, and chronology.
5. **Eligibility**: exact/reconstructed/unknown coverage and contradictions.
6. **Candidate sets**: source quality, selected-in-set rate, taxonomy, and co-availability.
7. **Propensity**: quality categories, exact checks, estimated-model diagnostics, and known
   missing assignment inputs.
8. **Overlap**: counts, propensity ranges, ESS, concentration, and support restrictions.
9. **Censoring**: follow-up completeness and required later adjustment.
10. **Concurrent interventions/confounding**: completed domain inventory.
11. **Contrast readiness matrix**.
12. **Allowed and prohibited claims**.
13. **Required logging repairs or randomized data collection**.
14. **Handoff artifacts and hashes**.

The contrast matrix is the central output:

| Contrast | Population | Candidate quality | Propensity quality | Support/ESS | Confounding status | Verdict | Restriction/next action |
|---|---|---|---|---|---|---|---|

Do not include an effect estimate column.

## 18. Handoff contract to credit-model training

A later target-generation/modeling agent may consume only:

- contrasts listed in `approved_contrasts.yaml`;
- the exact target-population restriction;
- verified candidate and eligibility masks;
- logged/reconstructed or cross-fitted out-of-fold propensities with quality labels;
- member-level cross-fitting assignments;
- pre-action features that passed leakage checks;
- censoring flags and required adjustment status; and
- the immutable audit manifest/hash.

The modeling agent must not broaden the cohort, add an email family, change the reference,
or relax the propensity/overlap restriction without rerunning this audit under a new
configuration.

The later longitudinal target provider will replace ALFWorld replay with recursive doubly
robust pseudo-targets. That implementation is out of scope here. This audit merely determines
where such targets can be attempted and what their identification label must be.

## 19. Proposed package layout

```text
src/longfeedback/audits/
    onboarding_logs.py          # audit orchestration
    source_inventory.py         # schema and revision manifest
    canonicalize_onboarding.py  # strict secure canonical tables
    eligibility.py              # availability audit
    candidate_actions.py        # taxonomy and candidate-set audit
    propensity.py               # exact/reconstructed/estimated checks
    overlap.py                  # support, ESS, and plots
    censoring.py                # endpoint observability
    readiness.py                # contrast verdict state machine
    privacy.py                  # HMAC, suppression, safe reporting
src/longfeedback/experiments/
    onboarding_log_audit.py     # config-driven entry point
configs/audits/
    onboarding_log_audit_template.yaml
tests/audits/
    fixtures/                   # synthetic data only
    test_onboarding_schema.py
    test_eligibility_audit.py
    test_candidate_action_audit.py
    test_propensity_audit.py
    test_overlap_audit.py
    test_privacy_guards.py
    test_readiness_decisions.py
```

Production source connectors may live in a separate internal package. The open LongFeedback
repository should contain interfaces, synthetic fixtures, audit logic, and aggregate report
templates—not internal table names, credentials, or raw member data.

## 20. Testing plan

### 20.1 Synthetic fixtures

Create small synthetic datasets for:

1. exact 50/50 randomized send/no-send with good overlap;
2. exact but deterministic assignment with no overlap;
3. estimated stochastic assignment with all confounders observed;
4. assignment depending on a declared missing hidden score;
5. selected action absent from candidate set;
6. ineligible no-send rows incorrectly offered as controls;
7. candidate reconstruction under two policy versions;
8. post-action click leaked into a propensity feature;
9. good marginal but failed day-specific overlap;
10. one member dominating inverse-propensity weight;
11. administrative censoring near extraction end; and
12. informative censoring with missing reasons.

### 20.2 Unit tests

- HMAC keys are stable under the same secret and differ across secrets.
- No raw key appears in any output artifact.
- Candidate probability vectors sum correctly.
- Coarsened action-family propensities equal the sum of their candidate components.
- Implicit `NO_EMAIL` is accepted only with an exact availability/probability reconstruction.
- Selected action must appear in candidate set.
- Ineligible rows cannot enter a causal contrast.
- Propensity quality categories cannot be silently upgraded.
- ESS and weight concentration match hand-computed examples.
- Cross-fitting keeps each member in one fold.
- Feature windows end before decision time.
- Outcome columns are inaccessible to outcome-blind audit functions.
- Small cells are suppressed in report tables and plots.
- Decision algorithm returns every expected verdict on fixtures.

### 20.3 Property tests

- Permuting row order does not change aggregate audit metrics.
- Duplicating a source join key is detected rather than multiplying rows silently.
- Adding an unsupported action cannot improve a contrast verdict.
- Moving a feature timestamp after the decision converts it to a leakage failure.
- Replacing all propensities by zero/one cannot pass overlap.
- Changing outcome values while preserving missingness does not change this audit’s verdict.
- All artifacts are deterministic for fixed inputs, configuration, and secret scope.

### 20.4 Integration test

Run the complete audit on synthetic tables and verify:

- only configured output paths are written;
- no network call is made;
- all required artifacts and hashes are emitted;
- the aggregate report contains no fixture member IDs or free text; and
- the readiness table matches the known synthetic truth.

## 21. Failure handling

| Failure | Required behavior |
|---|---|
| Source schema drift | stop canonicalization for that source; emit schema diff |
| Join multiplication | hard fail; do not deduplicate after the fact without a declared rule |
| Missing eligibility | mark affected contrasts predictive-only or request new extraction |
| Selected action absent from candidate set | hard data-integrity failure for affected revision |
| Invalid logged probability vector | quarantine revision and fail exact-propensity audit |
| Deterministic assignment | record exact propensity but fail overlap for unsupported contrast |
| Estimated propensity near zero | restrict/abstain; never hide with silent clipping |
| Missing assignment driver | observational causal readiness fails until recovered |
| Post-action leakage | remove feature, invalidate derived propensity artifacts, rerun |
| Missing day-28 outcome | distinguish censoring from inactivity; never fill with zero |
| Small privacy cell | suppress aggregate; do not print underlying rows |
| Network/external upload attempted | abort and report a governance violation |
| Outcome effect computed during audit | delete unauthorized artifact and restart from frozen outcome-blind contract |

## 22. Execution sequence

### Phase A: contract and secure setup

1. Resolve source owners, access boundary, output path, and HMAC secret mechanism.
2. Complete the decisions in Section 4 or mark them unknown.
3. Create the strict source-column mapping without printing raw rows.
4. Run the synthetic privacy/integration test.

Exit: the agent can inspect schemas and aggregates without exposing member data.

### Phase B: source and canonicalization audit

1. Produce source manifest and schema hashes.
2. Construct HMAC member/decision keys.
3. Canonicalize members, decisions, candidates, and pre-action features.
4. Run key, join, chronology, and endpoint-observability checks.

Exit: canonical row counts reconcile with every source and exclusion.

### Phase C: freeze outcome-blind contrasts

1. Canonicalize the primary binary action.
2. Identify exact candidate-set populations.
3. Define primary day range and policy revisions.
4. Freeze proposed contrast IDs and screening thresholds.

Exit: no contrast is chosen using day-28 effect estimates.

### Phase D: eligibility, propensity, and overlap

1. Complete eligibility and no-send decomposition.
2. Audit exact/reconstructed assignment mechanisms.
3. If authorized, fit member-cross-fitted estimated propensities.
4. Compute overlap, ESS, concentration, and conditional-support diagnostics.
5. Complete assignment-feature and concurrent-intervention inventory with data owners.
6. Audit censoring readiness.

Exit: every contrast has the evidence required by the readiness state machine.

### Phase E: review and handoff

1. Generate the contrast readiness matrix and report.
2. Have the data owner verify policy-version and assignment-feature statements.
3. Freeze `approved_contrasts.yaml` and manifest hashes.
4. Publish only aggregate/synthetic-safe artifacts.
5. Hand secure row-level artifacts to the later target-generation agent.

Exit: the project knows exactly which contrasts may receive randomized, observational,
restricted, or predictive-only treatment.

## 23. Minimum completion checklist

The audit is complete only when:

- [ ] source revisions and extraction hashes are recorded;
- [ ] member and decision keys are unique and pseudonymized securely;
- [ ] time zero, day-28 outcome, censoring, and decision grain are explicit;
- [ ] eligibility is known before action selection;
- [ ] eligible no-send is distinguishable from ineligible/no-candidate/cancelled send;
- [ ] selected actions belong to verified candidate sets;
- [ ] action taxonomy and template revisions are versioned;
- [ ] propensity quality is labeled per policy revision;
- [ ] exact propensities pass probability and calibration checks where applicable;
- [ ] estimated propensities are cross-fitted by member and contain no post-action features;
- [ ] overlap, ESS, and weight concentration are reported per contrast and key strata;
- [ ] known assignment drivers and concurrent interventions are inventoried;
- [ ] missing assignment drivers prevent observational-ready status;
- [ ] censoring is distinguished from inactivity;
- [ ] no outcome-by-action effect was computed;
- [ ] every contrast has a readiness verdict and allowed-claim string;
- [ ] raw and row-level artifacts remain secure and Git-ignored;
- [ ] aggregate reports satisfy small-cell suppression; and
- [ ] the manifest and approved-contrast hashes are frozen for handoff.

## 24. Reference material

- [LongFeedback causal assumptions](./causal_assumptions.md)
- [LongFeedback real-data causal-credit program](./real_credit_protocol.md)
- [LongFeedback scientific contract](./scientific_contract.md)
- [LongFeedback data governance](./data_governance.md)
- [Causal Inference: What If, Part III—time-varying treatments and g-methods](https://www.hsph.harvard.edu/miguel-hernan/wp-content/uploads/sites/1268/2024/04/hernanrobins_WhatIf_26apr24.pdf)
- [Doubly Robust Off-policy Value Evaluation for Reinforcement Learning](https://proceedings.mlr.press/v48/jiang16.html)
- [Double Reinforcement Learning for Efficient Off-policy Evaluation](https://jmlr.org/beta/papers/v21/19-827.html)
- [Learning optimal dynamic treatment regimes from longitudinal data](https://academic.oup.com/aje/article/193/12/1768/7693607)

The external references guide the later estimator design. They do not override the stricter
outcome-blind audit and claim boundaries specified here.

## 25. Suggested handoff prompt for the executing agent

Provide this document and the secure local source/config locations, then use a request like:

> Read `docs/linkedin_onboarding_log_audit_design.md` completely before accessing data.
> Implement and execute only the outcome-blind eligibility, candidate-action, propensity,
> overlap, assignment-feature, chronology, and censoring audit defined there. Treat all
> sources as read-only; make no network or external-service calls; never print or publish raw
> rows, IDs, free text, or small cells. Do not train a credit model, estimate an email effect,
> compare outcomes by action, or change a contrast after inspecting outcomes. Use the secure
> source mappings at `<SECURE_CONFIG_PATH>` and write row-level artifacts only to
> `<SECURE_OUTPUT_PATH>`. Mark missing business definitions or policy inputs as `unknown`
> rather than guessing. Finish by producing `audit_report.md`, `contrast_readiness.parquet`,
> `approved_contrasts.yaml`, all required audit JSON files, and an immutable manifest with
> hashes. State clearly which contrasts are `randomized_ready`, `observational_ready`,
> `restricted_ready`, `predictive_only`, or `invalid`, and why.

Do not paste credentials, secrets, raw table names containing sensitive identifiers, or
member-level data into the prompt. Supply them through the approved local configuration and
secret mechanisms.
