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
- **Role:** secondary replication source. WildChat-1M (ODC-BY) remains the primary
  conversational source and gets its adapter during Gate B.

Dataset work cannot block the structural-world experiments (E0/Gate A).
