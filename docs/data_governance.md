# Data governance checklist

Before any real-data adapter is enabled:

1. Pin the exact dataset organization/name, revision, checksum, and access method.
2. Record source, database, content, and derivative-artifact license obligations.
3. Split by conversation or permitted pseudonymous entity before label generation.
4. Remove location, network/header metadata, direct identifiers, and obvious PII.
5. Use a project-secret pseudonymous join key; never publish reversible identities.
6. Do not send raw rows to external labelers by default.
7. Cache label evidence with model/prompt/version provenance and provide deletion paths.
8. Publish code, manifests, aggregates, and synthetic/sanitized fixtures—not raw text.

## Enabled sources

### LMSYS-Chat-1M (local snapshot; adapter `longfeedback data prepare lmsys`)

- **Access:** gated HuggingFace dataset (`lmsys/lmsys-chat-1m`); each user accepts
  the LMSYS-Chat-1M Dataset License Agreement individually. The snapshot lives in
  the gitignored `data/` tree; the adapter pins the HF commit hash and per-shard
  sha256 checksums in `source_manifest.json`.
- **License obligations:** **no redistribution** of the dataset in whole or part
  (raw or processed text never leaves `data/`), delete-on-request compliance, and
  the dataset's own safety notice (unsafe conversations exist; the adapter's safe
  subset drops moderation-flagged conversations by default).
- **What may be published:** adapter code, source manifests, row identifiers
  (`shard:row` and source conversation IDs), and aggregate statistics only.
- **Sanitization:** the source ships pre-redacted PII placeholders; the adapter
  adds a local high-precision regex pass (emails, phone numbers, IP addresses),
  versioned as `pii_filter_version` in the manifest.
- **Known caveats (recorded in `stats.json`):** the source has no timestamps, so
  event times are synthetic and chronological splits are impossible (splits are
  stable conversation-id hashes); it has no user identifiers, so user-level
  leakage across splits cannot be ruled out.
- **Role:** secondary replication source. WildChat-1M (ODC-BY) is the primary
  conversational source (adapter below); this snapshot exists to confirm E1's
  finding is not an artifact of one source.

### WildChat-1M (local snapshot; adapter `longfeedback data prepare wildchat`)

- **Access:** ungated HuggingFace dataset (`allenai/WildChat-1M`), fetched directly
  (no sign-in or agreement required, unlike LMSYS). The snapshot lives in the
  gitignored `data/` tree; the adapter pins the HF commit hash (when fetched via
  `huggingface_hub`'s cache) or falls back to per-shard sha256 checksums in
  `source_manifest.json`.
- **License obligations:** ODC-BY 1.0 permits redistribution with attribution, but
  raw and processed text stay local anyway — one data-handling policy across every
  conversational source, not a license requirement specific to WildChat.
- **What may be published:** adapter code, source manifests, row identifiers
  (`shard:row` and source conversation hashes), and aggregate statistics only.
- **Sanitization:** the source ships a `redacted` flag and its own toxicity/OpenAI
  moderation labels; the adapter excludes flagged *and* toxic conversations by
  default and adds the same local high-precision regex PII pass used for LMSYS
  (`pii_filter_version`). The source also carries country, state, and hashed-IP
  fields per turn — the adapter never reads or forwards them into canonical
  payloads (only `role`/`content` cross into the `Trajectory`).
- **Known caveats (recorded in `stats.json`):** the source has real per-turn
  timestamps, but the adapter deliberately uses the same synthetic
  conversation-relative event times as the LMSYS adapter (E1 only needs turn
  order) rather than threading real timestamps through — a scope choice, not a
  data limitation; it has hashed IPs but no stable user identifier, so user-level
  leakage across splits cannot be ruled out.
- **Role:** primary conversational source. `make e1-wildchat` reruns E1 against
  it; results are compared against the LMSYS replication, never merged into one
  number.

Dataset work cannot block the structural-world experiments (E0/Gate A).

### HeartSteps V1 (public micro-randomized trial; E9)

- **Access and license:** public GitHub release, CC BY 4.0, pinned to commit
  `3016391de426116bdef41880d72bc8cd4b9b2477`; the manifest records checksums
  for `suggestions.csv`, `jbsteps.csv`, and `users.csv`.
- **Content:** deidentified decision records, wearable step counts, and survey
  covariates for 37 participants. Raw data remains in the gitignored `data/`
  tree even though redistribution is permitted with attribution.
- **Causal role:** at every available decision point, suggestion delivery was
  randomized with known probability 0.6. E9 uses `avail` for eligibility and
  `send` for assignment; the misleadingly named source field `is.randomized`
  must never be used as the inclusion indicator.
- **Scope:** E9 estimates group-level proximal and distal excursion effects.
  It does not expose an individual's missing counterfactual and is not a
  conversational-agent dataset.
