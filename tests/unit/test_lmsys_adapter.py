"""Tests for the LMSYS reader and manifest (skipped without pyarrow)."""

from __future__ import annotations

from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")

import pyarrow.parquet as pq  # noqa: E402

from longfeedback.config import LmsysDataConfig  # noqa: E402
from longfeedback.data.lmsys import (  # noqa: E402
    build_lmsys_manifest,
    iter_lmsys_records,
    prepare_lmsys,
)

_MODERATION_CLEAN = [{"flagged": False}]
_MODERATION_FLAGGED = [{"flagged": False}, {"flagged": True}]


def _conversation(*contents: str) -> list[dict[str, str]]:
    return [
        {"role": "user" if index % 2 == 0 else "assistant", "content": content}
        for index, content in enumerate(contents)
    ]


def _write_snapshot(root: Path, *, with_cache_metadata: bool) -> Path:
    rows = [
        {
            "conversation_id": "keep-1",
            "model": "vicuna-13b",
            "conversation": _conversation("q1", "a1", "q2", "a2", "q3", "a3"),
            "language": "English",
            "openai_moderation": _MODERATION_CLEAN,
            "redacted": False,
        },
        {
            "conversation_id": "flagged-1",
            "model": "vicuna-13b",
            "conversation": _conversation("q1", "a1", "q2", "a2", "q3", "a3"),
            "language": "English",
            "openai_moderation": _MODERATION_FLAGGED,
            "redacted": False,
        },
        {
            "conversation_id": "short-1",
            "model": "vicuna-13b",
            "conversation": _conversation("q1", "a1"),
            "language": "English",
            "openai_moderation": _MODERATION_CLEAN,
            "redacted": False,
        },
        {
            "conversation_id": "keep-2",
            "model": "gpt-3.5-turbo",
            "conversation": _conversation("q1", "a1", "q2", "a2", "q3", "a3"),
            "language": "English",
            "openai_moderation": _MODERATION_CLEAN,
            "redacted": True,
        },
    ]
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    shard = data_dir / "train-00000-of-00001-test.parquet"
    pq.write_table(pa.Table.from_pylist(rows), shard)
    if with_cache_metadata:
        cache = root / ".cache" / "huggingface" / "download" / "data"
        cache.mkdir(parents=True)
        (cache / f"{shard.name}.metadata").write_text(
            "fakecommit123\n" + "ab" * 32 + "\n1234.5\n",
            encoding="utf-8",
        )
    return root


def test_iter_records_reads_schema_and_flags(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snap", with_cache_metadata=False)
    records = list(iter_lmsys_records(snapshot))
    assert [record.conversation_id for record in records] == [
        "keep-1",
        "flagged-1",
        "short-1",
        "keep-2",
    ]
    assert records[1].flagged and not records[0].flagged
    assert records[3].redacted
    assert records[0].source_row_id.endswith(":0")


def test_manifest_prefers_cache_metadata_and_falls_back_to_hashing(tmp_path: Path) -> None:
    with_cache = build_lmsys_manifest(_write_snapshot(tmp_path / "a", with_cache_metadata=True))
    assert with_cache.source_version == "fakecommit123"
    assert not with_cache.redistribute_raw_text

    without_cache = build_lmsys_manifest(_write_snapshot(tmp_path / "b", with_cache_metadata=False))
    assert without_cache.source_version == "unknown-local-snapshot"
    assert len(without_cache.source_checksum) == 64
    assert without_cache.source_checksum != with_cache.source_checksum


def test_prepare_writes_artifacts_and_reconciled_stats(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snap", with_cache_metadata=True)
    output = tmp_path / "processed"
    config = LmsysDataConfig(input_dir=snapshot, output_dir=output, min_assistant_turns=3)
    result = prepare_lmsys(config)

    assert result.stats["conversations_scanned"] == 4
    assert result.stats["conversations_kept"] == 2
    assert result.stats["exclusions_by_reason"] == {
        "moderation_flagged": 1,
        "too_few_assistant_turns": 1,
    }
    assert sum(result.stats["split_conversations"].values()) == 2
    assert sum(result.stats["split_messages"].values()) == 12

    table = pq.read_table(result.artifacts["events"])
    assert table.num_rows == 12
    kept_ids = set(table.column("trajectory_id").to_pylist())
    assert kept_ids == {"lmsys-chat-1m:keep-1", "lmsys-chat-1m:keep-2"}

    # Determinism: a second run produces byte-identical artifacts.
    second = prepare_lmsys(config, output_dir=tmp_path / "processed2")
    assert result.artifacts["events"].read_bytes() == second.artifacts["events"].read_bytes()
    assert result.artifacts["stats"].read_bytes() == second.artifacts["stats"].read_bytes()


def test_prepare_rejects_empty_results(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snap", with_cache_metadata=False)
    config = LmsysDataConfig(
        input_dir=snapshot,
        output_dir=tmp_path / "out",
        language="Klingon",
    )
    with pytest.raises(ValueError, match="no conversations"):
        prepare_lmsys(config)


def test_missing_snapshot_directory_is_reported(tmp_path: Path) -> None:
    config = LmsysDataConfig(input_dir=tmp_path / "absent", output_dir=tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="no LMSYS parquet shards"):
        prepare_lmsys(config)
