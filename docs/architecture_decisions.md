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
