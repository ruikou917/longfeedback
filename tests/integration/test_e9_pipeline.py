from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from longfeedback.config import E9Config
from longfeedback.experiments.e9 import run_e9


def test_e9_pipeline_writes_randomized_effect_artifacts(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    rows: list[dict[str, object]] = []
    for user in range(12):
        distal = 8_000.0 + 50.0 * user
        for decision in range(40):
            action = decision % 2 == 0
            rows.append(
                {
                    "user_id": f"u{user}",
                    "available": True,
                    "randomized": True,
                    "action": action,
                    "action_probability": 0.5,
                    "proximal_steps30": 120.0 + 20.0 * action + decision,
                    "prior_steps30": 100.0 + decision,
                    "home_work": decision % 3 == 0,
                    "study_day": decision // 5 + 1,
                    "analysis_period": True,
                    "distal_week_daily_steps": distal,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), processed / "decisions.parquet")
    result = run_e9(
        E9Config(
            processed_dir=processed,
            output_dir=tmp_path / "artifacts",
            crossfit_folds=3,
            bootstrap={"resamples": 100, "confidence": 0.9, "seed": 3},
        )
    )

    assert result.metrics["data"]["participants"] == 12
    assert result.metrics["proximal_positive_control"]["log_step_effect"] > 0.0
    assert "distal_average_excursion_effect" in result.metrics
    assert all(path.is_file() for path in result.artifacts.values())
