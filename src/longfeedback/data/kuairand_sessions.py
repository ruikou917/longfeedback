"""Sessionize KuaiRand impressions for E8's real sequential power gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
from pydantic import HttpUrl

from longfeedback.config import KuaiRandSessionsDataConfig
from longfeedback.data.conversations import split_by_conversation_hash
from longfeedback.schema import SourceManifest

SOURCE_NAME = "KuaiRand-Pure sessions"
SOURCE_VERSION = "pure-session-v1"
SOURCE_LICENSE = "CC BY-SA 4.0"
SOURCE_URL = "https://zenodo.org/records/10439422"


@dataclass(frozen=True)
class KuaiRandSessionsPrepareResult:
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


def prepare_kuairand_sessions(
    config: KuaiRandSessionsDataConfig,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> KuaiRandSessionsPrepareResult:
    """Merge logs, sessionize by user/time, and persist one row per real step."""

    source_dir = (input_dir or config.input_dir).resolve() / "data"
    resolved_output = (output_dir or config.output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    files = [*config.log_random_files, *config.log_standard_files]
    paths = [source_dir / name for name in files]
    feature_path = source_dir / config.video_features_filename
    for path in [*paths, feature_path]:
        if not path.is_file():
            raise FileNotFoundError(f"expected KuaiRand source file at {path}")

    checksums = {path.name: _sha256_file(path) for path in [*paths, feature_path]}
    combined = hashlib.sha256(
        "\n".join(f"{name}:{value}" for name, value in sorted(checksums.items())).encode()
    ).hexdigest()
    manifest = SourceManifest(
        source_name=SOURCE_NAME,
        source_version=SOURCE_VERSION,
        source_license=SOURCE_LICENSE,
        source_url=HttpUrl(SOURCE_URL),
        derivative_license="CC BY-SA 4.0; processed rows remain local",
        redistribute_raw_text=False,
        required_attribution=True,
        pii_filter_version="none",
        labeler_version="session-survival-v1",
        source_checksum=combined,
    )

    con = duckdb.connect()
    try:
        selects = []
        for name in config.log_random_files:
            log_sql = _sql_path(source_dir / name)
            selects.append(
                f"SELECT *, 'random' AS source_policy FROM read_csv_auto('{log_sql}', header=true)"
            )
        for name in config.log_standard_files:
            log_sql = _sql_path(source_dir / name)
            selects.append(
                "SELECT *, 'standard' AS source_policy "
                f"FROM read_csv_auto('{log_sql}', header=true)"
            )
        con.execute("CREATE TEMP VIEW raw AS " + " UNION ALL BY NAME ".join(selects))
        con.execute(
            """
            CREATE TEMP TABLE ordered AS
            SELECT *, CASE WHEN lag_time IS NULL OR time_ms - lag_time > ?
                           THEN 1 ELSE 0 END AS new_session
            FROM (
              SELECT *, lag(time_ms) OVER (
                PARTITION BY user_id ORDER BY time_ms, video_id
              ) AS lag_time
              FROM raw WHERE time_ms IS NOT NULL
            )
            """,
            [int(config.session_gap_minutes * 60_000)],
        )
        con.execute(
            """
            CREATE TEMP TABLE positioned AS
            SELECT *, sum(new_session) OVER (
                     PARTITION BY user_id ORDER BY time_ms, video_id ROWS UNBOUNDED PRECEDING
                   ) AS session_number
            FROM ordered
            """
        )
        user_rows = con.execute("SELECT DISTINCT user_id FROM positioned").fetchall()
        user_ids = [str(row[0]) for row in user_rows]
        splits = [
            split_by_conversation_hash(
                user_id,
                seed=config.seed,
                train_fraction=config.train_fraction,
                validation_fraction=config.validation_fraction,
            )
            for user_id in user_ids
        ]
        con.register("user_splits", pa.table({"user_key": user_ids, "split": splits}))
        horizon_columns = ",\n".join(
            f"CAST(position + {horizon} < session_length AS INTEGER) AS survival_{horizon}"
            for horizon in config.survival_horizons
        )
        sessions_path = resolved_output / config.sessions_filename
        con.execute(
            f"""
            COPY (
              WITH steps AS (
                SELECT p.*,
                       row_number() OVER w - 1 AS position,
                       count(*) OVER w AS session_length
                FROM positioned p
                WINDOW w AS (PARTITION BY user_id, session_number ORDER BY time_ms, video_id
                             ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
              )
              SELECT concat(
                       CAST(s.user_id AS VARCHAR), ':', CAST(s.session_number AS VARCHAR)
                     ) AS session_id,
                     CAST(s.user_id AS VARCHAR) AS user_id,
                     CAST(s.video_id AS VARCHAR) AS video_id,
                     s.time_ms, s.position, s.session_length,
                     CAST(s.is_rand AS INTEGER) AS is_rand,
                     s.source_policy, us.split,
                     CAST(s.duration_ms AS DOUBLE) AS duration_ms,
                     CAST(s.play_time_ms AS DOUBLE) AS play_time_ms,
                     CAST(s.long_view AS INTEGER) AS long_view,
                     vf.video_type,
                     {horizon_columns}
              FROM steps s
              JOIN user_splits us ON CAST(s.user_id AS VARCHAR) = us.user_key
              LEFT JOIN read_csv_auto('{_sql_path(feature_path)}', header=true) vf USING (video_id)
              ORDER BY s.user_id, s.time_ms, s.video_id
            ) TO '{_sql_path(sessions_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        quoted_sessions_path = _sql_path(sessions_path)
        con.execute(
            f"CREATE TEMP VIEW sessions AS SELECT * FROM read_parquet('{quoted_sessions_path}')"
        )

        def fetch_row(query: str) -> tuple[Any, ...]:
            row = con.execute(query).fetchone()
            if row is None:  # pragma: no cover - aggregate queries always return one row
                raise RuntimeError(f"query returned no row: {query}")
            return row

        total_steps, users, sessions, randomized = fetch_row(
            "SELECT count(*), count(DISTINCT user_id), count(DISTINCT session_id), "
            "sum(is_rand) FROM sessions"
        )
        mixed_sessions = fetch_row(
            """SELECT count(*) FROM (
                 SELECT session_id FROM sessions GROUP BY session_id
                 HAVING min(is_rand) = 0 AND max(is_rand) = 1
               )"""
        )[0]
        mismatches = fetch_row(
            """SELECT count(*) FROM sessions
               WHERE (source_policy = 'random') != (is_rand = 1)"""
        )[0]
        base_rates = {
            str(horizon): float(
                fetch_row(f"SELECT avg(survival_{horizon}) FROM sessions WHERE is_rand = 1")[0]
            )
            for horizon in config.survival_horizons
        }
        split_users = dict(
            con.execute(
                "SELECT split, count(DISTINCT user_id) FROM sessions GROUP BY split"
            ).fetchall()
        )
    finally:
        con.close()

    if mismatches:
        sessions_path.unlink(missing_ok=True)
        raise ValueError(f"{mismatches} rows disagree between source policy and is_rand")
    stats: dict[str, Any] = {
        "source": SOURCE_NAME,
        "source_revision": SOURCE_VERSION,
        "steps": int(total_steps),
        "users": int(users),
        "sessions": int(sessions),
        "randomized_steps": int(randomized),
        "mixed_policy_sessions": int(mixed_sessions),
        "session_gap_minutes": config.session_gap_minutes,
        "randomized_survival_base_rates": base_rates,
        "split_users": {str(key): int(value) for key, value in split_users.items()},
        "source_checksums": checksums,
        "caveats": {
            "estimand": "group-level effects at randomized steps, never individual counterfactuals",
            "outcome": "within-session survival only; not return visits or welfare",
            "sessionization": "merged source logs; gap strictly greater than configured threshold",
        },
    }
    manifest_path = resolved_output / config.manifest_filename
    stats_path = resolved_output / config.stats_filename
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return KuaiRandSessionsPrepareResult(
        stats=stats,
        manifest=manifest,
        artifacts={"sessions": sessions_path, "manifest": manifest_path, "stats": stats_path},
    )
