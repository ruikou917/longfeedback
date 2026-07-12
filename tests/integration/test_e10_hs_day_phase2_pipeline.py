from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from longfeedback.experiments.e10_hs_day_phase2 import (
    E10HSDayPhase2Config,
    run_phase2,
)


def test_phase2_pipeline_crossfits_and_writes_artifacts(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    rows: list[dict[str, object]] = []
    for user in range(40):
        for day in range(4):
            actions = np.asarray([(user + day + position) % 2 for position in range(5)])
            terminal = 3.0 + 0.15 * float(np.sum(actions)) + 0.01 * user
            component = float(np.expm1(terminal))
            for position, action in enumerate(actions, start=1):
                rows.append(
                    {
                        "user_id": f"u{user}",
                        "study_day": float(day + 1),
                        "decision_number": 5 * day + position,
                        "available": True,
                        "randomized": True,
                        "action": bool(action),
                        "action_probability": 0.6,
                        "proximal_steps30": component,
                        "prior_steps30": 50.0 + position,
                        "home_work": (day + position) % 2,
                    }
                )
    pq.write_table(pa.Table.from_pylist(rows), processed / "decisions.parquet")
    config = E10HSDayPhase2Config.model_validate(
        {
            "name": "e10_hs_day_phase2",
            "processed_dir": processed,
            "decisions_filename": "decisions.parquet",
            "output_dir": tmp_path / "artifacts",
            "evaluation_folds": 2,
            "nuisance_folds": 2,
            "ridge_alpha": 1.0,
            "episode_slots": 5,
            "bootstrap_resamples": 100,
            "bootstrap_confidence": 0.9,
            "bootstrap_seed": 3,
            "model": {"d_model": 8, "n_layers": 1, "n_heads": 2, "dropout": 0.0},
            "training": {
                "epochs": 1,
                "batch_size": 32,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
            },
            "seeds": [0],
            "min_positive_seeds": 1,
            "outcome_rmse_tolerance": 10.0,
            "metrics_filename": "metrics.json",
            "predictions_filename": "predictions.csv",
            "manifest_filename": "run_manifest.json",
        }
    )

    result = run_phase2(config)

    assert result.metrics["data"]["complete_participant_days"] == 160
    assert result.metrics["data"]["participants"] == 40
    assert result.metrics["model"]["capacity_matched"] is True
    assert all(path.is_file() for path in result.artifacts.values())
