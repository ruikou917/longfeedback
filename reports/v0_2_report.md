# Learning Reward and Credit from Delayed Implicit Outcomes — v0.2 results

> Auto-generated from experiment artifacts; regenerate with `longfeedback report build`. All numbers are aggregate statistics.

## Abstract

We study whether delayed behavioral outcomes can supervise language-agent training. Across four structural world families with known causal ground truth, capacity-matched sequence models that predict the terminal outcome equally well differ drastically in how accurately their per-step signals recover true interventional credit; supervising an action-value head on paired counterfactual credit closes that gap. Bootstrap-ensemble uncertainty flags credit errors under parameter and logging-policy shift, and on real conversation logs a delayed future-feedback outcome is learnable well above trivial length baselines.

## 1. Claims and non-claims

Supported claims (structural worlds unless noted):

1. Terminal-outcome accuracy does not imply credit accuracy (E0/Gate A/Gate B).
2. Oracle-credit supervision improves credit recovery at fixed capacity (wins in 4/4 families).
3. Ensemble uncertainty correlates with credit error under distribution shift.
4. Real-log delayed feedback (next-turn pushback) is predictable above trivial baselines (predictive claim only).

Not claimed: real-user modeling, causal effects in observational chat logs, production transfer, or that behavioral proxies equal user welfare.

## 2. Structural world families

| Family | Difficulty axis | Proxy Y | Utility U |
|---|---|---|---|
| A fatigue/habit | stochasticity + partial observability | habit threshold | habit − fatigue/action costs |
| B hidden intent | exogenous latent shifts + hidden confounding | progress + shock threshold | matched progress |
| C delayed conversion | long/variable delay, competing causes | conversion | conversion value − send costs |
| D proxy divergence | Goodhart gap | return event | progress + trust − dependency − interruptions |

## 3. Method

One causal-Transformer architecture with three heads (terminal outcome, prefix value, action value); variants differ only in loss weights, so all comparisons are capacity-matched by construction. Oracle credit uses paired common-random-number counterfactuals with frozen continuation and adaptive Monte Carlo precision. Uncertainty is the between-member std of a 5-member bootstrap ensemble.

## 4. Results

### 4.1 Outcome accuracy vs credit recovery (in-distribution)

| Family | AUROC (outcome-only) | credit ρ outcome-only | credit ρ prefix/RUDDER | credit ρ credit-supervised |
|---|---:|---:|---:|---:|
| world_a | 0.945 | -0.071 | 0.001 | 0.939 |
| world_b | 0.771 | 0.047 | 0.283 | 0.419 |
| world_c | 1.000 | 0.122 | 0.119 | 0.362 |
| world_d | 0.502 | 0.008 | -0.178 | 0.823 |

Gate A (Worlds A/B, more regimes) shows the same pattern with a maximum credit-Spearman gap of 0.916 at outcome-AUROC differences below 0.044.

### 4.2 Uncertainty under distribution shift

| Family | shift | uncertainty–error ρ | error-detection AUROC | credit ρ degradation |
|---|---|---:|---:|---:|
| world_a | parameter_shift(noise scales) | 0.768 | 0.787 | 0.021 |
| world_b | logging_policy_shift(confounded->clean) | 0.832 | 0.806 | 0.361 |
| world_c | parameter_shift(delay structure) | 0.961 | 0.970 | 0.074 |
| world_d | parameter_shift(engagement/trust lifts) | 0.481 | 0.730 | 0.032 |

### 4.3 Real conversation logs (E1, LMSYS-Chat-1M)

20000 prepared conversations yield 82929 assistant-turn examples; the next-turn failure label has prevalence 0.124 on the test split.

| Model | AUROC | AUPRC | Brier | ECE | NLL |
|---|---:|---:|---:|---:|---:|
| base_rate | 0.500 | 0.119 | 0.109 | 0.003 | 0.375 |
| full_feature_ridge | 0.805 | 0.491 | 0.084 | 0.012 | 0.312 |
| sequence_transformer | 0.809 | 0.498 | 0.083 | 0.010 | 0.289 |
| trivial_length_ridge | 0.751 | 0.290 | 0.098 | 0.021 | 0.335 |

### 4.4 Policy sanity check (Gate A)

On `world_a_oracle`, a Q-greedy policy from the credit head reaches true utility 2.124 versus 1.307 for behavior cloning and 1.215 for the behavior policy, with no proxy-up/utility-down inversion.

## 5. Gate decisions

- Gate A: **pass**
- Gate B: **pass** (criteria: {"capacity_matched": true, "credit_recovery_across_families": true, "pass": true, "real_log_learnable": true, "uncertainty_under_shift": true})
- E1: **pass**

## 6. Limitations

- Real-log labels are rule-based behavioral proxies without human validation yet; the trivial length baseline alone reaches 0.751 AUROC, so much of the signal is positional.
- Credit supervision uses oracle labels available only in simulation; closing the gap from observable signals is future work.
- Cross-family transfer (leave-one-world-out training) is deferred; shift results cover parameter and logging-policy shifts within families.
- LMSYS conversations lack timestamps and user IDs (synthetic event times; possible cross-split user leakage).
- No LLM-native reranking yet (v0.3).

## 7. Reproducibility

```bash
make e0 && make gate-a && make gate-b        # structural experiments
make data-lmsys && make e1                   # real-log experiment (local data)
longfeedback report build                    # regenerate this report
```

Scientific metric hashes: e0 `759a055acf07`, gate_a `f919f6d1ccd5`, gate_b `951b3f9c8154`, e1 `814b23071376`.

