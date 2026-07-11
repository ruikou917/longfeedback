"""Integration test against a real local KuaiRand snapshot (auto-skips)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from longfeedback.config import KuaiRandDataConfig

_SNAPSHOT = Path(__file__).resolve().parents[2] / "data" / "kuairand-data"

pytestmark = pytest.mark.skipif(
    not (_SNAPSHOT / "data").is_dir(),
    reason="local KuaiRand-Pure snapshot not present",
)


def test_prepare_small_slice_of_real_snapshot(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from longfeedback.data.kuairand import prepare_kuairand

    config = KuaiRandDataConfig(
        input_dir=_SNAPSHOT,
        output_dir=tmp_path / "processed",
        max_rows_per_file=200,
    )
    result = prepare_kuairand(config)

    assert sum(result.stats["impressions_by_logging_policy"].values()) > 0
    assert result.stats["impressions_by_logging_policy"]["random"] > 0

    table = pq.read_table(result.artifacts["events"])
    splits = set(table.column("split").to_pylist())
    assert splits <= {"train", "validation", "test"}
    by_trajectory: dict[str, set[str]] = {}
    for trajectory_id, split in zip(
        table.column("trajectory_id").to_pylist(),
        table.column("split").to_pylist(),
        strict=True,
    ):
        by_trajectory.setdefault(trajectory_id, set()).add(split)
    assert all(len(assigned) == 1 for assigned in by_trajectory.values())
