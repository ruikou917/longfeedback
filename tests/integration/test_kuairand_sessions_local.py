from __future__ import annotations

import csv
from pathlib import Path

import pyarrow.parquet as pq

from longfeedback.config import KuaiRandSessionsDataConfig
from longfeedback.data.kuairand_sessions import prepare_kuairand_sessions

HEADER = [
    "user_id",
    "video_id",
    "time_ms",
    "duration_ms",
    "play_time_ms",
    "long_view",
    "is_rand",
]


def _write_log(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)


def test_prepare_kuairand_sessions_merges_policies_and_marks_survival(tmp_path: Path) -> None:
    source = tmp_path / "source" / "data"
    source.mkdir(parents=True)
    _write_log(source / "random.csv", [[1, 10, 1_000, 10_000, 2_000, 0, 1]])
    _write_log(
        source / "standard.csv",
        [[1, 11, 2_000, 20_000, 3_000, 1, 0], [1, 12, 4_000_000, 30_000, 1_000, 0, 0]],
    )
    with (source / "videos.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "video_type"])
        writer.writerows([[10, "NORMAL"], [11, "NORMAL"], [12, "AD"]])

    result = prepare_kuairand_sessions(
        KuaiRandSessionsDataConfig(
            input_dir=tmp_path / "source",
            output_dir=tmp_path / "processed",
            log_random_files=("random.csv",),
            log_standard_files=("standard.csv",),
            video_features_filename="videos.csv",
            survival_horizons=(1,),
        )
    )

    table = pq.read_table(result.artifacts["sessions"]).to_pydict()
    assert result.stats["steps"] == 3
    assert result.stats["sessions"] == 2
    assert result.stats["mixed_policy_sessions"] == 1
    random_index = table["is_rand"].index(1)
    assert table["survival_1"][random_index] == 1
