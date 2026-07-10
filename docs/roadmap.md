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

Remaining Gate B work before v0.3: WildChat adapter (primary conversational
source), human validation of rule labels, leave-one-family-out transfer, and
conservative (LCB) reward optimization with the overoptimization study.

## Gate C — infrastructure extraction

Extract event stores, outcome resolution, reward backfill, trainer-neutral batches,
and TRL/`verl` adapters only when at least two experiments use the same lifecycle
contract. Do not generalize a component used by one script.
