# LongFeedback

LongFeedback studies how language agents can learn from behavioral outcomes that
arrive after many decisions. It keeps four quantities separate: an observed
behavioral proxy, true utility in controlled worlds, predictive contribution,
and interventional action credit.

The repository is currently at **v0.1 / Gate A**: the deterministic E0 vertical
slice plus the first scientific experiment. Gate A trains capacity-matched
sequence models on two stochastic structural worlds and demonstrates the
project's central claim: terminal-outcome prediction accuracy does not imply
per-step credit recovery.

## What E0 establishes

- Immutable, versioned event and trajectory contracts.
- A deterministic fatigue-and-habit structural world with separate proxy and utility.
- Frozen and policy-reactive counterfactual continuation semantics.
- Paired common-random-number oracle credit with exact deterministic checks.
- Outcome and oracle-supervised ridge diagnostics plus RUDDER-style redistribution.
- A one-command experiment with machine-readable artifacts and leakage tests.

## What Gate A establishes

- Stochastic World A with oracle/noisy/partial observability regimes.
- World B (hidden Markov intent) with clean and hidden-confounded logging
  policies, exogenous outcome shocks, and privileged-signal quarantine.
- Adaptive paired Monte Carlo oracle credit with a seed-stability check.
- Three capacity-matched DOCM variants (outcome-only, prefix/RUDDER,
  credit-supervised) sharing one causal-Transformer architecture.
- The outcome-accuracy-versus-credit-recovery gap across logged-data regimes,
  and a discrete Q-greedy policy check against behavior cloning on true utility.
- An explicit machine-readable `gate_a_decision` block with the three gate
  criteria from the roadmap.

Neither experiment claims to model real users, identify causal effects in
observational chat logs, or provide production delayed-RL infrastructure.

## Quick start

```bash
make bootstrap
make test
make e0       # deterministic pipeline sanity (CPU, seconds)
make gate-a   # Gate A experiment (CPU, about a minute)
```

`make bootstrap` installs an immutable local wheel (with the `research` extra,
including a CPU PyTorch) so the same package boundary is tested in development
and CI; `make bootstrap-core` installs the torch-free core environment. On
platforms where editable installs are preferred, ordinary `uv sync --group dev
--extra research` and `uv run ...` commands also work.

The experiment writes its resolved configuration, metrics, predictions, and plot
under `artifacts/e0/`. Override the destination without changing the tracked config:

```bash
uv run --no-sync longfeedback experiment run e0 --output-dir /tmp/longfeedback-e0
```

The CLI defaults match `configs/experiments/e0.yaml`; pass that file with `--config`
when you want the tracked configuration to be explicit in a command or script.

Quality checks:

```bash
make qa
```

## Repository map

```text
configs/                 Versioned experiment configuration
docs/                    Scientific contracts, assumptions, and decisions
src/longfeedback/schema  Canonical serialized records
src/longfeedback/worlds  Controlled structural environments
src/longfeedback/credit  Counterfactual and redistributed credit
src/longfeedback/models  DOCM sequence models (research extra, torch)
src/longfeedback/baselines Diagnostic outcome/credit models
src/longfeedback/evaluation Metrics and reports
src/longfeedback/experiments Reproducible vertical slices
tests/                   Unit, property, integration, and reproducibility tests
```

## Development sequence

1. E0 deterministic pipeline sanity. (done)
2. Stochastic Worlds A/B, RUDDER, and DOCM MVP (Gate A). (done)
3. Real-log predictive outcomes plus Worlds C/D and uncertainty (Gate B).
4. Policy improvement, reward overoptimization, and LLM reranking.
5. Delayed-reward infrastructure extraction only after repeated abstractions
   justify Gate C.

See [the scientific contract](docs/scientific_contract.md) and
[the roadmap](docs/roadmap.md) before extending a target or making a causal claim.

## Data and safety

No public interaction dataset is downloaded by E0. Future adapters must pin exact
dataset IDs, revisions, checksums, and licenses; sanitize locally before any external
labeling; and publish manifests or sanitized fixtures instead of raw conversation
text. Behavioral engagement is a proxy, not user welfare.

## License

Code is licensed under Apache-2.0. Dataset terms and derived-artifact licenses are
tracked separately in source manifests and may be more restrictive.
