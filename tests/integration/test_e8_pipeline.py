"""End-to-end E8 pipeline test on tiny synthetic KuaiRand-shaped CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

from longfeedback.config import E8Config, KuaiRandSessionsDataConfig
from longfeedback.data.kuairand_sessions import prepare_kuairand_sessions
from longfeedback.experiments.e8 import run_e8

_LOG_HEADER = [
    "user_id",
    "video_id",
    "time_ms",
    "duration_ms",
    "play_time_ms",
    "long_view",
    "is_rand",
]
_MINUTE = 60_000


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _make_snapshot(root: Path, *, corrupt_is_rand: bool = False) -> Path:
    """Two users; sessions with interleaved randomized and standard steps."""

    data_dir = root / "snapshot" / "data"
    data_dir.mkdir(parents=True)
    standard_rows: list[list[object]] = []
    random_rows: list[list[object]] = []
    video = 0
    for user in range(1, 9):
        base = user * 1_000_000_000
        # One long session: 8 steps, with randomized insertions at steps 2, 5.
        for step in range(8):
            video += 1
            is_rand = 1 if step in (2, 5) else 0
            row = [
                user,
                video,
                base + step * _MINUTE,
                float(30_000 + 10_000 * (video % 7)),
                float(5_000 + 1_000 * (step % 3)),
                step % 2,
                is_rand,
            ]
            (random_rows if is_rand else standard_rows).append(row)
        # A second, short session after a 3-hour gap: 2 steps, one randomized.
        for step in range(2):
            video += 1
            is_rand = 1 if step == 0 else 0
            row = [
                user,
                video,
                base + 180 * _MINUTE + step * _MINUTE,
                float(45_000 + 5_000 * (video % 5)),
                4_000.0,
                0,
                is_rand,
            ]
            (random_rows if is_rand else standard_rows).append(row)
    if corrupt_is_rand:
        standard_rows[0][-1] = 1  # standard file claiming a randomized row
    _write_csv(data_dir / "log_random.csv", _LOG_HEADER, random_rows)
    _write_csv(data_dir / "log_standard.csv", _LOG_HEADER, standard_rows)
    _write_csv(
        data_dir / "video_features.csv",
        ["video_id", "video_type"],
        [[index, "NORMAL" if index % 3 else "AD"] for index in range(1, video + 1)],
    )
    return data_dir.parent


def _sessions_config(snapshot: Path, output: Path) -> KuaiRandSessionsDataConfig:
    return KuaiRandSessionsDataConfig.model_validate(
        {
            "input_dir": str(snapshot),
            "output_dir": str(output),
            "log_random_files": ["log_random.csv"],
            "log_standard_files": ["log_standard.csv"],
            "video_features_filename": "video_features.csv",
            "session_gap_minutes": 30,
            "survival_horizons": [1, 3],
        }
    )


def test_e8_pipeline_prepares_sessions_and_runs_the_power_gate(tmp_path: Path) -> None:
    snapshot = _make_snapshot(tmp_path)
    processed = tmp_path / "processed"
    prepared = prepare_kuairand_sessions(_sessions_config(snapshot, processed))

    stats = prepared.stats
    assert stats["users"] == 8
    assert stats["sessions"] == 16  # two sessions per user
    assert stats["randomized_steps"] == 8 * 3
    assert stats["mixed_policy_sessions"] == 16
    rates = stats["randomized_survival_base_rates"]
    # Long-session randomized steps (2, 5) survive >=1; the short-session
    # step at position 0 of 2 also survives >=1 -> rate 1.0. Horizon 3 is
    # survived only by the step at position 2 of 8.
    assert rates["1"] == 1.0
    assert rates["3"] == pytest.approx(1.0 / 3.0)

    config = E8Config.model_validate(
        {
            "processed_dir": str(processed),
            "output_dir": str(tmp_path / "artifacts"),
            "survival_horizons": [1, 3],
            "primary_horizon": 3,
            "duration_quantile_bins": 3,
            "bootstrap": {"resamples": 200, "confidence": 0.9, "seed": 0},
        }
    )
    result = run_e8(config)
    metrics = result.metrics
    decision = metrics["e8_decision"]
    assert set(decision) >= {
        "power_gate_pass",
        "ci_excludes_zero",
        "hypothesis_h8_delayed_step_effect",
        "phase_2_authorized",
    }
    assert decision["hypothesis_h8_delayed_step_effect"] in {
        "supported",
        "refuted_at_this_granularity",
    }
    assert decision["phase_2_authorized"] == decision["power_gate_pass"]
    assert metrics["primary_horizon"] == 3
    assert metrics["data"]["randomized_steps_analyzed"] == 24
    for block in metrics["horizon_effects"].values():
        assert block["ci_low"] <= block["slope"] <= block["ci_high"]
    assert {row["video_type"] for row in metrics["video_type_table"]} <= {"NORMAL", "AD", "unknown"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in result.artifacts.values())


def test_prepare_rejects_is_rand_source_file_mismatch(tmp_path: Path) -> None:
    snapshot = _make_snapshot(tmp_path, corrupt_is_rand=True)
    with pytest.raises(ValueError, match="disagree"):
        prepare_kuairand_sessions(_sessions_config(snapshot, tmp_path / "processed"))
