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

Dataset work is optional and cannot block E0.
