"""Prepare the public HeartSteps V1 micro-randomized trial for E9."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
from pydantic import HttpUrl

from longfeedback.config import HeartStepsDataConfig
from longfeedback.schema import SourceManifest

SOURCE_NAME = "HeartSteps V1"
SOURCE_URL = "https://github.com/klasnja/HeartStepsV1"
SOURCE_LICENSE = "CC BY 4.0"


@dataclass(frozen=True)
class HeartStepsPrepareResult:
    stats: dict[str, Any]
    manifest: SourceManifest
    artifacts: dict[str, Path]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_heartsteps(
    config: HeartStepsDataConfig,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> HeartStepsPrepareResult:
    """Reproduce the public DCEE preprocessing in an auditable parquet table."""

    root = (input_dir or config.input_dir).resolve()
    data_dir = root / "data_files"
    resolved_output = (output_dir or config.output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    paths = {
        "suggestions": data_dir / config.suggestions_filename,
        "steps": data_dir / config.steps_filename,
        "users": data_dir / config.users_filename,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"expected HeartSteps source file at {path}")
    revision = _git_revision(root)
    if revision is not None and revision != config.source_revision:
        raise ValueError(
            f"HeartSteps revision {revision} does not match pinned {config.source_revision}"
        )

    checksums = {name: _sha256_file(path) for name, path in paths.items()}
    combined = hashlib.sha256(
        "\n".join(f"{name}:{value}" for name, value in sorted(checksums.items())).encode()
    ).hexdigest()
    manifest = SourceManifest(
        source_name=SOURCE_NAME,
        source_version=config.source_revision,
        source_license=SOURCE_LICENSE,
        source_url=HttpUrl(SOURCE_URL),
        derivative_license="CC BY 4.0; attribution required",
        redistribute_raw_text=False,
        required_attribution=True,
        pii_filter_version="public-deidentified-v1",
        labeler_version="heartsteps-distal-week-v1",
        source_checksum=combined,
    )

    con = duckdb.connect()
    decisions_path = resolved_output / config.decisions_filename
    try:
        con.execute(
            f"CREATE TEMP VIEW suggestions AS SELECT * FROM "
            f"read_csv_auto('{_sql_path(paths['suggestions'])}', header=true)"
        )
        con.execute(
            f"CREATE TEMP VIEW users AS SELECT * FROM "
            f"read_csv_auto('{_sql_path(paths['users'])}', header=true)"
        )
        con.execute(
            f"CREATE TEMP VIEW steps AS SELECT * FROM "
            f"read_csv_auto('{_sql_path(paths['steps'])}', header=true)"
        )
        con.execute(
            """
            CREATE TEMP TABLE filtered AS
            SELECT s.*,
                   row_number() OVER (
                     PARTITION BY s."user.index"
                     ORDER BY try_cast(s."sugg.decision.utime" AS TIMESTAMP), s."decision.index"
                   ) - 1 AS decision_number
            FROM suggestions s
            JOIN users u USING ("user.index")
            WHERE try_cast(s."sugg.decision.utime" AS TIMESTAMP) IS NOT NULL
              AND (
                try_cast(u."travel.start" AS DATE) IS NULL
                OR CAST(try_cast(s."sugg.decision.utime" AS TIMESTAMP) AS DATE)
                     < try_cast(u."travel.start" AS DATE)
                OR CAST(try_cast(s."sugg.decision.utime" AS TIMESTAMP) AS DATE)
                     > try_cast(u."travel.end" AS DATE)
              )
            """
        )
        con.execute(
            """
            CREATE TEMP TABLE interval_steps AS
            SELECT "user.index" AS user_id, "decision.index" AS decision_index,
                   sum(coalesce(steps, 0)) AS steps_until_next_decision
            FROM steps
            WHERE "decision.index" IS NOT NULL
            GROUP BY "user.index", "decision.index"
            """
        )
        con.execute(
            """
            CREATE TEMP TABLE base AS
            SELECT CAST(f."user.index" AS VARCHAR) AS user_id,
                   CAST(f."decision.index" AS BIGINT) AS decision_index,
                   CAST(f.decision_number AS BIGINT) AS decision_number,
                   try_cast(f."sugg.decision.utime" AS TIMESTAMP) AS decision_time,
                   CAST(f."sugg.decision.slot" AS INTEGER) AS decision_slot,
                   CAST(f.avail AS BOOLEAN) AS available,
                   CAST(f.avail AS BOOLEAN) AS randomized,
                   CAST(f."is.randomized" AS BOOLEAN) AS source_is_randomized,
                   CAST(f.send AS BOOLEAN) AS action,
                   ?::DOUBLE AS action_probability,
                   coalesce(CAST(f.jbsteps30 AS DOUBLE), 0.0) AS proximal_steps30,
                   coalesce(CAST(f.jbsteps30pre AS DOUBLE), 0.0) AS prior_steps30,
                   CAST(f."dec.location.category" IN ('home', 'work') AS INTEGER) AS home_work,
                   coalesce(i.steps_until_next_decision, 0.0) AS steps_until_next_decision
            FROM filtered f
            LEFT JOIN interval_steps i
              ON f."user.index" = i.user_id AND f."decision.index" = i.decision_index
            WHERE f.decision_number < ?
            """,
            [config.suggestion_probability, config.max_decisions],
        )
        con.execute(
            """
            CREATE TEMP TABLE distal AS
            SELECT user_id,
                   sum(steps_until_next_decision) / 7.0 AS distal_week_daily_steps
            FROM (
              SELECT *, row_number() OVER (
                PARTITION BY user_id ORDER BY decision_number DESC
              ) AS reverse_position
              FROM base
            )
            WHERE reverse_position <= ?
            GROUP BY user_id
            """,
            [config.distal_week_decisions],
        )
        con.execute(
            f"""
            COPY (
              SELECT b.*,
                     floor(b.decision_number / 5) + 1 AS study_day,
                     count(*) OVER (PARTITION BY b.user_id) AS participant_decisions,
                     b.decision_number < count(*) OVER (PARTITION BY b.user_id) -
                       {config.distal_week_decisions} AS analysis_period,
                     d.distal_week_daily_steps
              FROM base b JOIN distal d USING (user_id)
              ORDER BY b.user_id, b.decision_number
            ) TO '{_sql_path(decisions_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        decisions_sql = _sql_path(decisions_path)
        con.execute(f"CREATE TEMP VIEW prepared AS SELECT * FROM read_parquet('{decisions_sql}')")
        participants, decisions, randomized, available_randomized = con.execute(
            """SELECT count(DISTINCT user_id), count(*), sum(CAST(randomized AS INTEGER)),
                      sum(CAST(randomized AND available AS INTEGER)) FROM prepared"""
        ).fetchall()[0]
        action_rate = con.execute(
            "SELECT avg(CAST(action AS DOUBLE)) FROM prepared WHERE randomized AND available"
        ).fetchall()[0][0]
        analysis_rows = con.execute(
            "SELECT count(*) FROM prepared WHERE randomized AND available AND analysis_period"
        ).fetchall()[0][0]
    finally:
        con.close()

    stats: dict[str, Any] = {
        "source": SOURCE_NAME,
        "source_revision": config.source_revision,
        "participants": int(participants),
        "decisions": int(decisions),
        "randomized_decisions": int(randomized),
        "available_randomized_decisions": int(available_randomized),
        "analysis_rows": int(analysis_rows),
        "observed_action_rate": float(action_rate),
        "known_action_probability": config.suggestion_probability,
        "distal_outcome": "average daily Jawbone steps over final 35 decision points",
        "source_checksums": checksums,
    }
    manifest_path = resolved_output / config.manifest_filename
    stats_path = resolved_output / config.stats_filename
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return HeartStepsPrepareResult(
        stats=stats,
        manifest=manifest,
        artifacts={"decisions": decisions_path, "manifest": manifest_path, "stats": stats_path},
    )
