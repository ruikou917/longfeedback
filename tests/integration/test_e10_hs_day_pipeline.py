from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from longfeedback.experiments.e10_hs_day import E10HSDayConfig, run_e10_hs_day


def test_e10_hs_day_reconstructs_days_and_passes_strong_signal(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    rng = np.random.default_rng(12)
    rows: list[dict[str, object]] = []
    for user in range(20):
        for day in range(12):
            actions = rng.random(5) < 0.6
            terminal_score = 4.0 + 0.20 * float(np.sum(actions))
            proximal_steps = float(np.expm1(terminal_score))
            for position, action in enumerate(actions, start=1):
                rows.append(
                    {
                        "user_id": f"u{user}",
                        "study_day": day + 1,
                        "decision_number": 5 * day + position,
                        "available": True,
                        "randomized": True,
                        "action": bool(action),
                        "home_work": (position + day) % 2,
                        "proximal_steps30": proximal_steps,
                    }
                )
    # An incomplete day must be excluded before any effect calculation.
    for position in range(1, 5):
        rows.append(
            {
                "user_id": "incomplete",
                "study_day": 1,
                "decision_number": position,
                "available": True,
                "randomized": True,
                "action": position % 2 == 0,
                "home_work": position % 2,
                "proximal_steps30": 100.0,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), processed / "decisions.parquet")

    result = run_e10_hs_day(
        E10HSDayConfig(
            processed_dir=processed,
            output_dir=tmp_path / "artifacts",
            bootstrap_resamples=100,
            bootstrap_seed=4,
        )
    )

    assert result.metrics["data"]["complete_participant_days"] == 240
    assert result.metrics["data"]["eligible_randomized_decisions"] == 1_200
    assert result.metrics["global_effect"]["effect"] > 0.10
    assert result.metrics["e10_hs_day_decision"]["signal_gate_pass"] is True
    assert all(path.is_file() for path in result.artifacts.values())
