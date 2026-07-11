"""KuaiRand-Pure adapter: local randomized-and-standard logs to canonical trajectories.

E6's scientific value is the ``log_random_*`` files' genuinely uniform exposure
policy (the "randomized bridge"): a known-exact logging propensity licenses
off-policy value estimates against real user data, unlike WildChat/LMSYS,
which license E1's predictive claims only. ``log_standard_*`` rows remain
production-confounded and must never be treated as randomized. See
``docs/scientific_contract.md`` ("E6 acceptance contract") and
ADR-008/ADR-009 in ``docs/architecture_decisions.md`` for the single-step
scoping and the column-verification caveat this reader carries until a real
snapshot lands (none exists locally yet).

Each impression becomes its own single-step trajectory (ADR-008); this module
only requires user/item/time identifier columns and passes every other
numeric column through as an opaque engagement signal (ADR-009), so it
degrades gracefully if the real header differs from public documentation.
Requires the ``research`` extra (pyarrow) to write the canonical parquet
events; the CSV reader itself only needs the standard library.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import HttpUrl

from longfeedback.config import KuaiRandDataConfig
from longfeedback.data.conversations import split_by_conversation_hash
from longfeedback.schema import Event, EventType, ObservationRegime, SourceManifest, Trajectory

SOURCE_NAME = "KuaiRand-Pure"
_SOURCE_KEY = "kuairand-pure"
_SOURCE_URL = "https://kuairand.com/"
_SOURCE_LICENSE = "KuaiRand research-use license (non-commercial; see kuairand.com)"

LABELER_VERSION = "kuairand-rules-v1"

LoggingPolicy = Literal["random", "standard"]

_USER_ID_CANDIDATES = ("user_id",)
_ITEM_ID_CANDIDATES = ("video_id", "item_id")
_TIME_CANDIDATES = ("time_ms", "timestamp", "time")
_DATE_CANDIDATES = ("date",)

# Best-effort names from public KuaiRand documentation; the reader does not
# require these, and treats every other numeric column as an opaque
# passthrough engagement signal (ADR-009). Verify against a real header.
KUAIRAND_KNOWN_ENGAGEMENT_COLUMNS: tuple[str, ...] = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
)

# Any of these signals being nonzero counts as a positive engagement outcome;
# a deliberately simple, versioned, deterministic rule over immediate signals
# only (ADR-008) -- not a session-linked outcome.
_POSITIVE_ENGAGEMENT_COLUMNS: tuple[str, ...] = (
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "long_view",
)


@dataclass(frozen=True, slots=True)
class ImpressionRecord:
    """One source impression row before canonical conversion."""

    row_id: str
    user_id: str
    item_id: str
    event_time: datetime
    logging_policy: LoggingPolicy
    engagement: dict[str, float]


@dataclass(frozen=True)
class KuaiRandPrepareResult:
    stats: dict[str, Any]
    manifest: SourceManifest
    artifacts: dict[str, Path]


def _require_column(header: list[str], candidates: tuple[str, ...]) -> str:
    lowered = {name.lower(): name for name in header}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        f"none of {candidates} found in CSV header {header}; the KuaiRand "
        "column names may differ from documentation -- see ADR-009"
    )


def _find_column(header: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {name.lower(): name for name in header}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _parse_event_time(
    row: dict[str, str], *, time_column: str | None, date_column: str | None
) -> datetime:
    if time_column is not None and row.get(time_column):
        raw = row[time_column]
        try:
            millis = int(float(raw))
        except ValueError as error:
            raise ValueError(f"non-numeric time value {raw!r} in column {time_column!r}") from error
        return datetime.fromtimestamp(millis / 1000.0, tz=UTC)
    if date_column is not None and row.get(date_column):
        return datetime.strptime(row[date_column], "%Y%m%d").replace(tzinfo=UTC)
    raise ValueError("row has neither a usable time nor a date column")


def _row_engagement(
    row: dict[str, str], *, header: list[str], identifier_columns: set[str]
) -> dict[str, float]:
    engagement: dict[str, float] = {}
    for column in header:
        if column in identifier_columns:
            continue
        raw = row.get(column, "")
        if raw == "":
            continue
        try:
            engagement[column] = float(raw)
        except ValueError:
            continue
    return engagement


def iter_kuairand_records(
    path: Path,
    *,
    logging_policy: LoggingPolicy,
    max_rows: int = 0,
) -> Iterator[ImpressionRecord]:
    """Stream impressions from one KuaiRand log CSV in row order."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if not header:
            raise ValueError(f"{path} has no CSV header")
        header_list = list(header)
        user_column = _require_column(header_list, _USER_ID_CANDIDATES)
        item_column = _require_column(header_list, _ITEM_ID_CANDIDATES)
        time_column = _find_column(header_list, _TIME_CANDIDATES)
        date_column = _find_column(header_list, _DATE_CANDIDATES)
        if time_column is None and date_column is None:
            raise ValueError(
                f"{path} has neither a time nor a date column among {header_list}; "
                "cannot order impressions -- see ADR-009"
            )
        identifier_columns = {
            column for column in (user_column, item_column, time_column, date_column) if column
        }
        for row_index, row in enumerate(reader):
            if max_rows and row_index >= max_rows:
                break
            event_time = _parse_event_time(row, time_column=time_column, date_column=date_column)
            engagement = _row_engagement(
                row, header=header_list, identifier_columns=identifier_columns
            )
            yield ImpressionRecord(
                row_id=f"{path.name}:{row_index}",
                user_id=str(row[user_column]),
                item_id=str(row[item_column]),
                event_time=event_time,
                logging_policy=logging_policy,
                engagement=engagement,
            )


def _is_engaged(engagement: dict[str, float]) -> bool:
    return any(engagement.get(column, 0.0) != 0.0 for column in _POSITIVE_ENGAGEMENT_COLUMNS)


def impression_to_trajectory(record: ImpressionRecord) -> Trajectory:
    """Convert one impression into a single-step canonical trajectory (ADR-008)."""

    trajectory_id = f"{_SOURCE_KEY}:{record.logging_policy}:{record.row_id}"
    base_time = record.event_time
    engaged = _is_engaged(record.engagement)
    events = (
        Event(
            trajectory_id=trajectory_id,
            event_id=f"{trajectory_id}:0",
            event_time=base_time,
            step_index=0,
            event_type=EventType.OBSERVATION,
            payload={"user_id": record.user_id},
            source=_SOURCE_KEY,
            source_row_id=record.row_id,
            policy_id=None,
        ),
        Event(
            trajectory_id=trajectory_id,
            event_id=f"{trajectory_id}:1",
            event_time=base_time,
            step_index=0,
            event_type=EventType.ACTION,
            payload={"video_id": record.item_id},
            source=_SOURCE_KEY,
            source_row_id=record.row_id,
            policy_id="uniform_random" if record.logging_policy == "random" else "production",
        ),
        Event(
            trajectory_id=trajectory_id,
            event_id=f"{trajectory_id}:2",
            event_time=base_time,
            step_index=0,
            event_type=EventType.USER_RESPONSE,
            payload=dict(record.engagement),
            source=_SOURCE_KEY,
            source_row_id=record.row_id,
            policy_id=None,
        ),
        Event(
            trajectory_id=trajectory_id,
            event_id=f"{trajectory_id}:3",
            event_time=base_time,
            step_index=0,
            event_type=EventType.OUTCOME,
            payload={"engaged": engaged, "rule_version": LABELER_VERSION},
            source=_SOURCE_KEY,
            source_row_id=record.row_id,
            policy_id=None,
        ),
    )
    return Trajectory(
        trajectory_id=trajectory_id,
        events=events,
        start_time=base_time,
        end_time=base_time,
        behavior_policy_id=events[1].policy_id,
        observation_regime=ObservationRegime.CLEAN,
        metadata={
            "source": _SOURCE_KEY,
            "logging_policy": record.logging_policy,
            "synthetic_time": False,
        },
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_kuairand_manifest(input_dir: Path, config: KuaiRandDataConfig) -> SourceManifest:
    files = [*config.log_random_files, *config.log_standard_files]
    checksums = {name: _sha256_file(input_dir / "data" / name) for name in files}
    combined = hashlib.sha256(
        "\n".join(f"{name}:{value}" for name, value in sorted(checksums.items())).encode()
    ).hexdigest()
    return SourceManifest(
        source_name=SOURCE_NAME,
        source_version="pure",
        source_license=_SOURCE_LICENSE,
        source_url=HttpUrl(_SOURCE_URL),
        derivative_license="not-redistributable (derived rows stay local)",
        redistribute_raw_text=False,
        required_attribution=True,
        pii_filter_version="none",
        labeler_version=LABELER_VERSION,
        source_checksum=combined,
    )


def _events_table(rows: list[dict[str, Any]]) -> pa.Table:
    schema = pa.schema(
        [
            ("trajectory_id", pa.string()),
            ("event_id", pa.string()),
            ("step_index", pa.int32()),
            ("event_type", pa.string()),
            ("logging_policy", pa.string()),
            ("split", pa.string()),
            ("source_row_id", pa.string()),
            ("event_time", pa.string()),
            ("payload_json", pa.string()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _trajectory_event_rows(
    trajectory: Trajectory, *, logging_policy: LoggingPolicy, split: str
) -> list[dict[str, Any]]:
    rows = []
    for event in trajectory.events:
        rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "event_id": event.event_id,
                "step_index": event.step_index,
                "event_type": event.event_type.value,
                "logging_policy": logging_policy,
                "split": split,
                "source_row_id": event.source_row_id,
                "event_time": event.event_time.isoformat(),
                "payload_json": json.dumps(event.payload, sort_keys=True),
            }
        )
    return rows


def prepare_kuairand(
    config: KuaiRandDataConfig,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> KuaiRandPrepareResult:
    """Read the local logs, convert to canonical trajectories, split, and persist."""

    resolved_input = (input_dir or config.input_dir).resolve()
    resolved_output = output_dir or config.output_dir
    resolved_output.mkdir(parents=True, exist_ok=True)

    files: list[tuple[str, LoggingPolicy]] = [
        (name, "random") for name in config.log_random_files
    ] + [(name, "standard") for name in config.log_standard_files]
    for filename, _ in files:
        path = resolved_input / "data" / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"expected KuaiRand log at {path}; see docs/roadmap.md for the "
                "required local snapshot layout"
            )

    manifest = build_kuairand_manifest(resolved_input, config)
    event_rows: list[dict[str, Any]] = []
    split_trajectories = dict.fromkeys(("train", "validation", "test"), 0)
    split_events = dict.fromkeys(("train", "validation", "test"), 0)
    rows_by_policy: dict[str, int] = {"random": 0, "standard": 0}
    engaged_by_policy: dict[str, int] = {"random": 0, "standard": 0}
    columns_consumed: set[str] = set()

    for filename, logging_policy in files:
        path = resolved_input / "data" / filename
        for record in iter_kuairand_records(
            path, logging_policy=logging_policy, max_rows=config.max_rows_per_file
        ):
            trajectory = impression_to_trajectory(record)
            split = split_by_conversation_hash(
                record.user_id,
                seed=config.seed,
                train_fraction=config.train_fraction,
                validation_fraction=config.validation_fraction,
            )
            rows = _trajectory_event_rows(trajectory, logging_policy=logging_policy, split=split)
            event_rows.extend(rows)
            split_trajectories[split] += 1
            split_events[split] += len(rows)
            rows_by_policy[logging_policy] += 1
            engaged_by_policy[logging_policy] += int(_is_engaged(record.engagement))
            columns_consumed.update(record.engagement.keys())

    if not event_rows:
        raise ValueError("no impressions were read from the configured KuaiRand log files")

    stats: dict[str, Any] = {
        "source": SOURCE_NAME,
        "source_version": manifest.source_version,
        "impressions_by_logging_policy": rows_by_policy,
        "engaged_by_logging_policy": engaged_by_policy,
        "split_trajectories": split_trajectories,
        "split_events": split_events,
        "engagement_columns_consumed": sorted(columns_consumed),
        "labeler_version": LABELER_VERSION,
        "caveats": {
            "randomized_bridge": (
                "only logging_policy='random' rows carry a known-uniform exposure "
                "propensity; 'standard' rows are production-confounded, exactly "
                "like WildChat/LMSYS, and must not be treated as randomized"
            ),
            "single_step_only": (
                "each row is its own single-step trajectory; no cross-session or "
                "delayed outcome is modeled (ADR-008)"
            ),
            "schema_unverified": (
                "engagement_columns_consumed reflects whatever the real CSV header "
                "contained; cross-check it against KUAIRAND_KNOWN_ENGAGEMENT_COLUMNS "
                "since no local snapshot existed when this adapter was written "
                "(ADR-009)"
            ),
            "split_strategy": "user-id hash (keeps one user's impressions in one split)",
        },
    }

    events_path = resolved_output / config.events_filename
    manifest_path = resolved_output / config.manifest_filename
    stats_path = resolved_output / config.stats_filename
    pq.write_table(_events_table(event_rows), events_path)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return KuaiRandPrepareResult(
        stats=stats,
        manifest=manifest,
        artifacts={"events": events_path, "manifest": manifest_path, "stats": stats_path},
    )
