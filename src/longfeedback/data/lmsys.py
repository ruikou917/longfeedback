"""LMSYS-Chat-1M adapter: gated local snapshot to canonical trajectories.

The dataset license prohibits redistribution; raw and processed artifacts must
stay under the gitignored ``data/`` tree, and only code, manifests, row IDs,
and aggregate statistics may be published. Requires the ``research`` extra
(pyarrow).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import HttpUrl

from longfeedback.config import LmsysDataConfig
from longfeedback.data.conversations import (
    PII_FILTER_VERSION,
    ConversationRecord,
    ConversationTurn,
    conversation_exclusion_reason,
    conversation_to_trajectory,
    split_by_conversation_hash,
)
from longfeedback.schema import SourceManifest, Trajectory

SOURCE_NAME = "LMSYS-Chat-1M"
_SOURCE_KEY = "lmsys-chat-1m"
_SOURCE_URL = "https://huggingface.co/datasets/lmsys/lmsys-chat-1m"
_SOURCE_LICENSE = "LMSYS-Chat-1M Dataset License Agreement (gated; no redistribution)"


@dataclass(frozen=True)
class LmsysPrepareResult:
    stats: dict[str, Any]
    manifest: SourceManifest
    artifacts: dict[str, Path]


def _shard_paths(input_dir: Path) -> list[Path]:
    shards = sorted((input_dir / "data").glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no LMSYS parquet shards under {input_dir / 'data'}; "
            "expected a HuggingFace snapshot of lmsys/lmsys-chat-1m"
        )
    return shards


def _row_flagged(moderation: list[dict[str, Any]] | None) -> bool:
    if not moderation:
        return False
    return any(bool(entry.get("flagged")) for entry in moderation)


def _record_from_row(row: dict[str, Any], *, shard: str, row_index: int) -> ConversationRecord:
    turns = tuple(
        ConversationTurn(role=str(message["role"]), content=str(message["content"]))
        for message in row["conversation"]
    )
    return ConversationRecord(
        conversation_id=str(row["conversation_id"]),
        turns=turns,
        assistant_model=str(row["model"]) if row.get("model") else None,
        language=str(row["language"]) if row.get("language") else None,
        flagged=_row_flagged(row.get("openai_moderation")),
        redacted=bool(row.get("redacted")),
        source=_SOURCE_KEY,
        source_row_id=f"{shard}:{row_index}",
    )


def iter_lmsys_records(input_dir: Path, *, batch_size: int = 1_024) -> Iterator[ConversationRecord]:
    """Stream conversations from the local snapshot in stable shard order."""

    columns = [
        "conversation_id",
        "model",
        "conversation",
        "language",
        "openai_moderation",
        "redacted",
    ]
    for shard_path in _shard_paths(input_dir):
        parquet_file = pq.ParquetFile(shard_path)
        row_index = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                yield _record_from_row(row, shard=shard_path.name, row_index=row_index)
                row_index += 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shard_checksums(input_dir: Path) -> tuple[dict[str, str], str | None]:
    """Return per-shard sha256s and the HF commit hash when available.

    The HuggingFace download cache stores ``<etag-commit>\n<sha256>\n<time>``
    metadata next to each shard; hashing 1.4 GB is the fallback.
    """

    checksums: dict[str, str] = {}
    commit: str | None = None
    for shard_path in _shard_paths(input_dir):
        cache_dir = input_dir / ".cache" / "huggingface" / "download" / "data"
        metadata_path = cache_dir / f"{shard_path.name}.metadata"
        sha256: str | None = None
        if metadata_path.is_file():
            lines = metadata_path.read_text(encoding="utf-8").splitlines()
            if len(lines) >= 2 and len(lines[1]) == 64:
                commit = commit or lines[0].strip() or None
                sha256 = lines[1].strip()
        if sha256 is None:
            sha256 = _sha256_file(shard_path)
        checksums[shard_path.name] = sha256
    return checksums, commit


def build_lmsys_manifest(input_dir: Path) -> SourceManifest:
    checksums, commit = _shard_checksums(input_dir)
    combined = hashlib.sha256(
        "\n".join(f"{name}:{value}" for name, value in sorted(checksums.items())).encode()
    ).hexdigest()
    return SourceManifest(
        source_name=SOURCE_NAME,
        source_version=commit or "unknown-local-snapshot",
        source_license=_SOURCE_LICENSE,
        source_url=HttpUrl(_SOURCE_URL),
        derivative_license="not-redistributable (derived text stays local)",
        redistribute_raw_text=False,
        required_attribution=True,
        pii_filter_version=PII_FILTER_VERSION,
        labeler_version="none",
        source_checksum=combined,
    )


def _events_table(rows: list[dict[str, Any]]) -> pa.Table:
    schema = pa.schema(
        [
            ("trajectory_id", pa.string()),
            ("event_id", pa.string()),
            ("step_index", pa.int32()),
            ("event_type", pa.string()),
            ("role", pa.string()),
            ("content", pa.string()),
            ("split", pa.string()),
            ("assistant_model", pa.string()),
            ("language", pa.string()),
            ("source_row_id", pa.string()),
            ("event_time", pa.string()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _trajectory_event_rows(trajectory: Trajectory, split: str) -> list[dict[str, Any]]:
    rows = []
    for event in trajectory.events:
        rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "event_id": event.event_id,
                "step_index": event.step_index,
                "event_type": event.event_type.value,
                "role": str(event.payload["role"]),
                "content": str(event.payload["content"]),
                "split": split,
                "assistant_model": trajectory.behavior_policy_id,
                "language": trajectory.metadata.get("language"),
                "source_row_id": event.source_row_id,
                "event_time": event.event_time.isoformat(),
            }
        )
    return rows


def prepare_lmsys(
    config: LmsysDataConfig,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> LmsysPrepareResult:
    """Filter, sanitize, convert, split, and persist the local snapshot."""

    resolved_input = (input_dir or config.input_dir).resolve()
    resolved_output = output_dir or config.output_dir
    resolved_output.mkdir(parents=True, exist_ok=True)

    manifest = build_lmsys_manifest(resolved_input)
    exclusions: dict[str, int] = {}
    split_conversations = dict.fromkeys(("train", "validation", "test"), 0)
    split_messages = dict.fromkeys(("train", "validation", "test"), 0)
    event_rows: list[dict[str, Any]] = []
    scanned = 0
    kept = 0
    total_redactions = 0

    for record in iter_lmsys_records(resolved_input):
        if config.max_conversations and kept >= config.max_conversations:
            break
        scanned += 1
        reason = conversation_exclusion_reason(
            record,
            language=config.language,
            min_assistant_turns=config.min_assistant_turns,
            max_message_chars=config.max_message_chars,
            exclude_flagged=config.exclude_flagged,
            include_redacted=config.include_redacted,
        )
        if reason is not None:
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        trajectory, redactions = conversation_to_trajectory(record, conversation_index=kept)
        split = split_by_conversation_hash(
            record.conversation_id,
            seed=config.seed,
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
        )
        rows = _trajectory_event_rows(trajectory, split)
        event_rows.extend(rows)
        split_conversations[split] += 1
        split_messages[split] += len(rows)
        total_redactions += redactions
        kept += 1

    if kept == 0:
        raise ValueError("no conversations satisfied the inclusion criteria")

    stats: dict[str, Any] = {
        "source": SOURCE_NAME,
        "source_version": manifest.source_version,
        "conversations_scanned": scanned,
        "conversations_kept": kept,
        "exclusions_by_reason": dict(sorted(exclusions.items())),
        "split_conversations": split_conversations,
        "split_messages": split_messages,
        "pii_filter_version": PII_FILTER_VERSION,
        "pii_redactions": total_redactions,
        "caveats": {
            "synthetic_time": "the source has no timestamps; event times are surrogates",
            "user_leakage_unknown": (
                "the source has no user identifiers; the same user may appear in multiple splits"
            ),
            "split_strategy": "conversation-id hash (chronological split impossible)",
        },
    }

    events_path = resolved_output / config.events_filename
    manifest_path = resolved_output / config.manifest_filename
    stats_path = resolved_output / config.stats_filename
    pq.write_table(_events_table(event_rows), events_path)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return LmsysPrepareResult(
        stats=stats,
        manifest=manifest,
        artifacts={"events": events_path, "manifest": manifest_path, "stats": stats_path},
    )
