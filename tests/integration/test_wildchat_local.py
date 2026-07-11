"""Integration test against the real local WildChat snapshot (auto-skips)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from longfeedback.config import WildChatDataConfig

_SNAPSHOT = Path(__file__).resolve().parents[2] / "data" / "wildchat-data"

pytestmark = pytest.mark.skipif(
    not (_SNAPSHOT / "data").is_dir(),
    reason="local WildChat-1M snapshot not present",
)


def test_prepare_small_slice_of_real_snapshot(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from longfeedback.data.wildchat import prepare_wildchat

    config = WildChatDataConfig(
        input_dir=_SNAPSHOT,
        output_dir=tmp_path / "processed",
        max_conversations=200,
    )
    result = prepare_wildchat(config)

    assert result.stats["conversations_kept"] == 200
    # source_version resolves to a real HF commit hash only when the snapshot
    # was fetched through huggingface_hub's cache (unlike LMSYS's local copy,
    # this snapshot may have been fetched by a direct download); either value
    # is a valid, honestly-reported outcome, so this only checks the field is
    # present and non-empty.
    assert result.manifest.source_version
    assert not result.manifest.redistribute_raw_text

    table = pq.read_table(result.artifacts["events"])
    assert table.num_rows == sum(result.stats["split_messages"].values())
    splits = set(table.column("split").to_pylist())
    assert splits <= {"train", "validation", "test"}
    # Conversations are split-atomic: one split per trajectory id.
    by_trajectory: dict[str, set[str]] = {}
    for trajectory_id, split in zip(
        table.column("trajectory_id").to_pylist(),
        table.column("split").to_pylist(),
        strict=True,
    ):
        by_trajectory.setdefault(trajectory_id, set()).add(split)
    assert all(len(assigned) == 1 for assigned in by_trajectory.values())
