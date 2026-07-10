# Roadmap and gates

## E0 — deterministic vertical slice (done)

Schema → World A → exact/paired oracle → diagnostic baselines → report.

## Gate A — Worlds A/B and DOCM MVP (current release)

Stochastic World A with oracle/noisy/partial observability, World B with hidden
Markov intent and confounded logging, adaptive paired Monte Carlo oracle credit,
and three capacity-matched DOCM variants. `make gate-a` emits a machine-readable
`gate_a_decision` block with the three criteria:

1. at least one regime shows similar outcome AUROC across variants but a
   materially different oracle credit recovery;
2. the oracle intervention evaluator is stable under repeated seeds;
3. a Q-greedy policy from the credit head beats behavior cloning on true
   utility without raising the proxy while lowering utility.

## Gate B — robust research result

Add Worlds C/D, structural shifts, ensembles, conservative reward optimization,
and a carefully governed real-log predictive-outcome replication (the first
conversational dataset adapter lands here). Continue only if the result survives
multiple world families or can be honestly repositioned around uncertainty-aware
delayed reward modeling.

## Gate C — infrastructure extraction

Extract event stores, outcome resolution, reward backfill, trainer-neutral batches,
and TRL/`verl` adapters only when at least two experiments use the same lifecycle
contract. Do not generalize a component used by one script.
