"""Tests for the KuaiRand CSV reader and manifest (skipped without pyarrow).

These synthetic fixtures double as the schema this adapter expects; ADR-009
records that the real column names are unverified until a local snapshot
exists, so this test file is the executable spec of that assumption.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow.parquet as pq

from longfeedback.config import KuaiRandDataConfig
from longfeedback.data.kuairand import (
    build_kuairand_manifest,
    impression_to_trajectory,
    iter_kuairand_records,
    prepare_kuairand,
)

_HEADER = [
    "user_id",
    "video_id",
    "time_ms",
    "date",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
]


def _row(
    *,
    user_id: str,
    video_id: str,
    time_ms: int,
    is_like: int = 0,
    is_follow: int = 0,
    long_view: int = 0,
) -> list[str]:
    return [
        user_id,
        video_id,
        str(time_ms),
        "20220422",
        "0",
        str(is_like),
        str(is_follow),
        "0",
        "0",
        "0",
        str(long_view),
        "3200",
        "9000",
    ]


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        writer.writerows(rows)


def _write_snapshot(root: Path) -> Path:
    data_dir = root / "data"
    _write_csv(
        data_dir / "log_random_4_22_to_5_08_pure.csv",
        [
            _row(user_id="u1", video_id="v1", time_ms=1_650_000_000_000, is_like=1),
            _row(user_id="u2", video_id="v2", time_ms=1_650_000_001_000),
        ],
    )
    _write_csv(
        data_dir / "log_standard_4_08_to_4_21_pure.csv",
        [_row(user_id="u1", video_id="v3", time_ms=1_649_000_000_000, long_view=1)],
    )
    _write_csv(
        data_dir / "log_standard_4_22_to_5_08_pure.csv",
        [_row(user_id="u3", video_id="v4", time_ms=1_650_100_000_000)],
    )
    return root


def _config(root: Path, output: Path) -> KuaiRandDataConfig:
    return KuaiRandDataConfig(input_dir=root, output_dir=output)


def test_iter_records_reads_identifiers_and_engagement(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snap")
    records = list(
        iter_kuairand_records(
            snapshot / "data" / "log_random_4_22_to_5_08_pure.csv", logging_policy="random"
        )
    )
    assert [record.user_id for record in records] == ["u1", "u2"]
    assert records[0].item_id == "v1"
    assert records[0].engagement["is_like"] == 1.0
    assert records[0].logging_policy == "random"
    assert records[0].row_id.endswith(":0")


def test_impression_to_trajectory_is_single_step_with_outcome(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snap")
    record = next(
        iter_kuairand_records(
            snapshot / "data" / "log_random_4_22_to_5_08_pure.csv", logging_policy="random"
        )
    )
    trajectory = impression_to_trajectory(record)
    assert len(trajectory.events) == 4
    assert all(event.step_index == 0 for event in trajectory.events)
    assert trajectory.metadata["logging_policy"] == "random"
    outcome_event = trajectory.events[-1]
    assert outcome_event.payload["engaged"] is True


def test_missing_identifier_column_raises_actionable_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["not_a_user_column", "video_id", "time_ms"])
        writer.writerow(["u1", "v1", "1650000000000"])
    with pytest.raises(ValueError, match="user_id"):
        list(iter_kuairand_records(path, logging_policy="random"))


def test_manifest_checksums_are_stable_and_local_only(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snap")
    config = _config(snapshot, tmp_path / "out")
    manifest = build_kuairand_manifest(snapshot, config)
    assert len(manifest.source_checksum) == 64
    assert not manifest.redistribute_raw_text
    assert build_kuairand_manifest(snapshot, config).source_checksum == manifest.source_checksum


def test_prepare_writes_artifacts_and_separates_logging_policies(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snap")
    config = _config(snapshot, tmp_path / "out")
    result = prepare_kuairand(config)

    assert result.stats["impressions_by_logging_policy"] == {"random": 2, "standard": 2}
    assert result.stats["engaged_by_logging_policy"]["random"] == 1
    assert set(result.stats["engagement_columns_consumed"]) >= {"is_like", "long_view"}

    table = pq.read_table(result.artifacts["events"])
    assert table.num_rows == 4 * 4
    policies = set(table.column("logging_policy").to_pylist())
    assert policies == {"random", "standard"}

    second = prepare_kuairand(config, output_dir=tmp_path / "out2")
    assert result.artifacts["events"].read_bytes() == second.artifacts["events"].read_bytes()


def test_prepare_reports_missing_file(tmp_path: Path) -> None:
    config = _config(tmp_path / "absent", tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="expected KuaiRand log"):
        prepare_kuairand(config)
