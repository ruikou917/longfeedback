"""Tests for the E6 randomized-bridge bias comparison (skipped without pyarrow)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from longfeedback.config import E6Config, KuaiRandDataConfig
from longfeedback.data.kuairand import prepare_kuairand
from longfeedback.experiments.e6 import (
    build_examples,
    load_feature_table,
    load_impressions,
    per_video_engagement_rate,
    run_e6,
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


def _row(*, user_id: str, video_id: str, time_ms: int, engaged: bool) -> list[str]:
    return [
        user_id,
        video_id,
        str(time_ms),
        "20220422",
        "0",
        "1" if engaged else "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "3200",
        "9000",
    ]


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        writer.writerows(rows)


def _write_biased_snapshot(root: Path) -> Path:
    """video 'popular' is over-targeted in standard but only average when random."""

    data_dir = root / "data"
    random_rows = []
    standard_rows = []
    for index in range(20):
        # Random exposure: both videos have the same true 50% engagement rate.
        random_rows.append(
            _row(
                user_id=f"ru{index}",
                video_id="popular",
                time_ms=1_650_000_000_000 + index,
                engaged=index % 2 == 0,
            )
        )
        random_rows.append(
            _row(
                user_id=f"ru{index}",
                video_id="niche",
                time_ms=1_650_000_100_000 + index,
                engaged=index % 2 == 0,
            )
        )
        # Standard/confounded exposure: the production policy over-targets
        # "popular" toward users who like it (always engaged) and shows
        # "niche" mostly to indifferent users (rarely engaged).
        standard_rows.append(
            _row(
                user_id=f"su{index}",
                video_id="popular",
                time_ms=1_649_000_000_000 + index,
                engaged=True,
            )
        )
        standard_rows.append(
            _row(
                user_id=f"su{index}",
                video_id="niche",
                time_ms=1_649_100_000_000 + index,
                engaged=False,
            )
        )
    _write_csv(data_dir / "log_random_4_22_to_5_08_pure.csv", random_rows)
    _write_csv(data_dir / "log_standard_4_08_to_4_21_pure.csv", standard_rows)
    _write_csv(data_dir / "log_standard_4_22_to_5_08_pure.csv", [])

    import csv as csv_module

    with (data_dir / "user_features_pure.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.writer(handle)
        writer.writerow(["user_id", "user_active_degree", "follow_user_num"])
        for index in range(20):
            writer.writerow([f"ru{index}", "full_active", str(index)])
            writer.writerow([f"su{index}", "high_active", str(index * 2)])
    with (data_dir / "video_features_basic_pure.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv_module.writer(handle)
        writer.writerow(["video_id", "video_type", "video_duration"])
        writer.writerow(["popular", "NORMAL", "12000"])
        writer.writerow(["niche", "NORMAL", "9000"])
    return root


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    snapshot = _write_biased_snapshot(tmp_path / "snap")
    processed = tmp_path / "processed"
    kuairand_config = KuaiRandDataConfig(input_dir=snapshot, output_dir=processed)
    prepare_kuairand(kuairand_config)
    return processed, snapshot


def test_per_video_engagement_rate_separates_logging_policies(tmp_path: Path) -> None:
    processed, _snapshot = _prepare(tmp_path)
    impressions = load_impressions(processed / "events.parquet")

    confounded = per_video_engagement_rate(impressions, logging_policy="standard", min_exposures=5)
    randomized = per_video_engagement_rate(impressions, logging_policy="random", min_exposures=5)

    assert confounded["popular"][0] == pytest.approx(1.0)
    assert confounded["niche"][0] == pytest.approx(0.0)
    assert randomized["popular"][0] == pytest.approx(0.5)
    assert randomized["niche"][0] == pytest.approx(0.5)


def test_load_feature_table_hashes_categoricals_deterministically(tmp_path: Path) -> None:
    _processed, snapshot = _prepare(tmp_path)
    user_features = load_feature_table(
        snapshot / "data" / "user_features_pure.csv", id_column="user_id"
    )
    video_features = load_feature_table(
        snapshot / "data" / "video_features_basic_pure.csv", id_column="video_id"
    )
    assert user_features["ru0"].shape == (2,)
    assert video_features["popular"].shape == (2,)
    # Same categorical value -> same hash-derived feature every time.
    assert user_features["ru0"][0] == user_features["ru5"][0]  # both "full_active"
    assert video_features["popular"][0] == video_features["niche"][0]  # both "NORMAL"
    # Numeric columns pass through untouched.
    assert video_features["popular"][1] == pytest.approx(12000.0)


def test_build_examples_joins_and_drops_missing_features(tmp_path: Path) -> None:
    processed, snapshot = _prepare(tmp_path)
    impressions = load_impressions(processed / "events.parquet")
    user_features = load_feature_table(
        snapshot / "data" / "user_features_pure.csv", id_column="user_id"
    )
    video_features = load_feature_table(
        snapshot / "data" / "video_features_basic_pure.csv", id_column="video_id"
    )

    standard_examples = build_examples(
        impressions,
        logging_policy="standard",
        user_features=user_features,
        video_features=video_features,
    )
    assert len(standard_examples) == 40  # 2 videos x 20 standard-log users
    assert all(example.features.shape == (4,) for example in standard_examples)

    # Dropping the user feature table entirely means no example can be built.
    assert not build_examples(
        impressions, logging_policy="standard", user_features={}, video_features=video_features
    )


def test_run_e6_detects_confounding_bias(tmp_path: Path) -> None:
    processed, snapshot = _prepare(tmp_path)
    config = E6Config(
        processed_dir=processed,
        raw_dir=snapshot,
        output_dir=tmp_path / "out",
        min_exposures_per_video=5,
    )
    result = run_e6(config)

    decision = result.metrics["e6_decision"]
    assert decision["videos_compared"] == 2
    assert decision["confounding_bias_detected"] is True
    # "popular" is inflated (+0.5) and "niche" deflated (-0.5) by the same
    # magnitude, so the signed mean_bias cancels to ~0 even though every
    # video is badly miscalibrated -- mean_absolute_bias is what should catch
    # that, which is exactly why confounding_bias_detected uses it.
    assert decision["mean_bias"] == pytest.approx(0.0, abs=1e-9)
    assert decision["mean_absolute_bias"] == pytest.approx(0.5, abs=1e-9)
    assert decision["hypothesis_h6_confounded_log_bias"] in (
        "biased_but_rank_useful",
        "biased_and_uninformative",
    )

    feature_adjustment = result.metrics["e6_feature_adjustment"]
    assert feature_adjustment["train_examples"] == 40
    assert feature_adjustment["eval_examples"] == 40
    assert feature_adjustment["hypothesis_h6b_feature_adjustment_helps"] in (
        "supported",
        "partially_supported_beats_trivial_not_naive_video_rate",
        "refuted_in_this_environment",
    )
    assert result.artifacts["feature_adjustment_predictions"].is_file()

    # Determinism: a second run over the same inputs is byte-identical.
    second = run_e6(config, output_dir=tmp_path / "out2")
    assert (
        result.metrics["scientific_metrics_sha256"] == second.metrics["scientific_metrics_sha256"]
    )


def test_run_e6_requires_overlapping_videos(tmp_path: Path) -> None:
    processed, snapshot = _prepare(tmp_path)
    config = E6Config(
        processed_dir=processed,
        raw_dir=snapshot,
        output_dir=tmp_path / "out",
        min_exposures_per_video=1_000,
    )
    with pytest.raises(ValueError, match="no video appears"):
        run_e6(config)


def test_run_e6_requires_feature_coverage(tmp_path: Path) -> None:
    processed, _snapshot = _prepare(tmp_path)
    empty_features = tmp_path / "empty_features"
    (empty_features / "data").mkdir(parents=True)
    import csv as csv_module

    for name, id_column in (
        ("user_features_pure.csv", "user_id"),
        ("video_features_basic_pure.csv", "video_id"),
    ):
        with (empty_features / "data" / name).open("w", newline="", encoding="utf-8") as handle:
            csv_module.writer(handle).writerow([id_column, "some_feature"])
    config = E6Config(
        processed_dir=processed,
        raw_dir=empty_features,
        output_dir=tmp_path / "out",
        min_exposures_per_video=5,
    )
    with pytest.raises(ValueError, match="no impression had both user and video"):
        run_e6(config)
