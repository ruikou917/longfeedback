"""End-to-end E6 test against the real prepared KuaiRand events (auto-skips)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from longfeedback.config import E6Config

_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed" / "kuairand"

pytestmark = pytest.mark.skipif(
    not (_PROCESSED / "events.parquet").is_file(),
    reason="run `make data-kuairand` first to prepare local KuaiRand events",
)


def test_e6_pipeline_detects_confounding_bias_on_real_data(tmp_path: Path) -> None:
    from longfeedback.experiments.e6 import run_e6

    config = E6Config(processed_dir=_PROCESSED, output_dir=tmp_path / "e6")
    result = run_e6(config)

    decision = result.metrics["e6_decision"]
    assert decision["videos_compared"] > 0
    # The production recommender targets far better than uniform random
    # exposure, so the confounded rate is expected to overestimate the true
    # rate on this real snapshot -- this is the E6 finding, not an assumption
    # the test forces; a differently-shaped real snapshot could flip it, and
    # the assertion is intentionally loose (any material bias) for that
    # reason.
    assert decision["confounding_bias_detected"] is True

    feature_adjustment = result.metrics["e6_feature_adjustment"]
    assert feature_adjustment["train_examples"] > 0
    assert feature_adjustment["eval_examples"] > 0
    # Loose on purpose: whether feature-conditioning actually beats the naive
    # per-video rate on real data is the open empirical question this block
    # exists to answer, not something the test should assume either way.
    assert feature_adjustment["hypothesis_h6b_feature_adjustment_helps"] in (
        "supported",
        "partially_supported_beats_trivial_not_naive_video_rate",
        "refuted_in_this_environment",
    )
