# Architecture decisions

## ADR-001: Build vertically

Implement E0 end to end before adding unused package namespaces. Phase 2 storage and
trainer adapters wait until Gate B/C demonstrates repeated lifecycle abstractions.

## ADR-002: Boundary schemas

Use strict frozen Pydantic records at serialization boundaries and frozen dataclasses
for inner-loop world transitions. Store processed data as Arrow/Parquet later; use
JSON/YAML only for small configuration and manifests.

## ADR-003: Configuration and CLI

Use YAML validated by Pydantic and a Typer CLI. Hydra can be introduced only when
configuration composition materially reduces experiment duplication.

## ADR-004: Reproducibility source of truth

Local metric JSON, resolved configuration, and run manifests are authoritative.
External tracking services remain optional mirrors.

## ADR-005: Dependencies

Keep E0 NumPy-based and CPU-only. PyTorch, Arrow, and DuckDB are available through the
`research` extra so the core smoke path stays small. CUDA, TRL, and `verl` will be
pinned in later optional environments instead of entering core dependencies.

## ADR-006: Capacity matching by shared architecture

Gate A model variants differ only in loss weights over one network with three heads
(outcome, prefix value, action value). This makes every outcome-versus-credit
comparison capacity-matched by construction instead of by tuning, at the cost of
carrying unused heads in ablated variants. The metrics report asserts equal
parameter counts.

## ADR-007: Derived RNG streams for additive noise sources

New exogenous noise channels (for example World A observation noise) draw from a
separate stream derived from the episode seed (`Random(f"observation:{seed}")`)
rather than extending the primary stream. Enabling a new channel therefore never
perturbs dynamics or policy draws for existing seeds, which keeps earlier
experiments byte-reproducible.

## ADR-008: E6 is a single-step randomized bandit, not a delayed multi-day study

KuaiRand's `log_random_*` files carry a genuinely uniform-random exposure policy —
unlike WildChat/LMSYS, which license E1's predictive claims only, a known-propensity
random log licenses actual off-policy value estimates (importance sampling /
SNIPS) against `log_standard_*` (production-confounded) training data. That
propensity guarantee, not outcome delay, is E6's scientific contribution, so E6
scopes each impression as its own single-step trajectory (horizon 1) with an
immediate, deterministic engagement rule as the outcome — it does not attempt
cross-session return-visit linking. Extending to a delayed, session-linked
outcome is a possible follow-up once the single-step bridge result is in hand,
not a prerequisite for it.

## ADR-009: KuaiRand column names were unverified pending the real snapshot

`src/longfeedback/data/kuairand.py` requires only a minimal user/item/time
identifier column set and treats every other numeric column as an opaque,
passed-through engagement signal, rather than hard-coding the full column
list from documentation. **Update 2026-07-10:** KuaiRand-Pure was downloaded
directly from Zenodo (openly hosted, no gated form; see `docs/roadmap.md`)
and the real header matched `KUAIRAND_KNOWN_ENGAGEMENT_COLUMNS` almost
exactly, plus a few extra columns (`is_rand`, `tab`, etc.) the passthrough
design picked up automatically without any code change needed. The
passthrough design is kept as-is going forward (it is what let verification
happen without a code change), not because it is still unverified.

## ADR-010: WildChat reuses LMSYS's synthetic-time and canonical-conversion
scope, dropping real per-turn data the source provides

WildChat-1M ships real per-turn timestamps and toxicity flags that
LMSYS-Chat-1M does not. `src/longfeedback/data/wildchat.py` deliberately does
not thread the real timestamps into canonical `Trajectory` events -- it
reuses the exact same synthetic conversation-relative time convention as the
LMSYS adapter, because E1 only needs turn order and doing otherwise would
mean the two adapters produce structurally different data for the same
downstream experiment. `toxic=True` is folded into the same exclusion path as
`openai_moderation.flagged` (both are "drop this conversation" signals),
rather than becoming a second independent exclusion axis. Extending WildChat
to use its real timestamps is a possible future item (e.g. for a genuinely
delayed/session-linked real-log experiment) but is out of scope for making
E1 comparable across both sources today.

## ADR-011: leave-one-family-out transfer tests a threshold, not model weights,
and does not gate Gate B's `pass`

Gate B's four world families each have a different observation space and
action count by construction (four distinct structural challenges), so a
DOCM model trained on one family cannot literally be evaluated on another --
there is no shared input tensor shape. `leave_one_family_out_transfer` in
`src/longfeedback/experiments/gate_b.py` instead tests whether the *shape* of
the uncertainty-error relationship generalizes: standardize each family's
ensemble uncertainty and binarize its credit error against that family's own
median, fit a single classification threshold (Youden's J) by pooling three
families, and apply it unmodified to the fourth. This is a real, falsifiable
transfer claim (a threshold fit on unrelated families could easily fail to
beat chance) without requiring a shared feature space redesign across the
four worlds, which would undermine their purpose as four *distinct*
structural challenges. Because this criterion was added well after Gate B
was already recorded as `passed`, and a real run shows it holds for only 3 of
4 families (World C's threshold does not transfer, despite a perfect
self-calibrated score), it is reported as `gate_b_decision.transfer` but
deliberately excluded from the `pass` boolean -- a negative or partial result
here is a genuine finding to report, not grounds to silently flip an
already-established gate to failing.
