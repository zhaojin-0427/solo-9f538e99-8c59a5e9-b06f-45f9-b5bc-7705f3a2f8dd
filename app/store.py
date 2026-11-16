"""SQLite metadata store and atomic version publishing.

Version publication follows this sequence, all while holding a SQLite
``BEGIN IMMEDIATE`` write transaction:

1. compare the caller's expected current version with the stored one;
2. allocate the next version number and render a complete GeoTIFF in a
   temporary file;
3. re-open and verify the rendered file, fsync it and atomically rename it to
   its immutable final path;
4. insert the version row and move the ``current_version`` pointer in the same
   transaction.

Consequently the current pointer only moves after a complete image exists on
disk.  A crash before commit leaves the previous version current and at worst
an unreferenced file, which startup reconciliation removes.  Versions are
never overwritten or deleted, so every published version stays downloadable.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import rasters
from .config import Settings
from .rasters import SourceBlock, iso_utc

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    version        INTEGER PRIMARY KEY,
    day            TEXT    NOT NULL,
    status         TEXT    NOT NULL CHECK (status IN ('ready')),
    published_at   TEXT    NOT NULL,
    width          INTEGER NOT NULL,
    height         INTEGER NOT NULL,
    crs_wkt        TEXT    NOT NULL,
    crs_label      TEXT    NOT NULL,
    transform      TEXT    NOT NULL,
    bounds         TEXT    NOT NULL,
    res_x          REAL    NOT NULL,
    res_y          REAL    NOT NULL,
    dtype          TEXT    NOT NULL,
    nodata         INTEGER NOT NULL,
    sha256         TEXT    NOT NULL,
    size_bytes     INTEGER NOT NULL,
    block_meta     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS versions_day_idx ON versions(day);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    report_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    longitude          REAL    NOT NULL,
    latitude           REAL    NOT NULL,
    start_day          TEXT    NOT NULL,
    end_day            TEXT    NOT NULL,
    threshold_mm       REAL    NOT NULL,
    params_key         TEXT    NOT NULL,
    requested_days     TEXT    NOT NULL,
    selected_versions  TEXT    NOT NULL,
    snapshot_version   INTEGER NOT NULL,
    snapshot_at        TEXT    NOT NULL,
    evidence           TEXT    NOT NULL,
    events             TEXT    NOT NULL,
    created_at         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS analyses_params_idx ON analyses(params_key, report_id);

CREATE TABLE IF NOT EXISTS analysis_current (
    params_key        TEXT    PRIMARY KEY,
    report_id         INTEGER NOT NULL,
    snapshot_version  INTEGER NOT NULL,
    snapshot_at       TEXT    NOT NULL
);
"""

CURRENT_STATE_KEY = "current_version"
ORPHAN_GRACE_SECONDS = 3600.0


class VersionConflict(Exception):
    """The caller's expected version is not the published current version."""

    def __init__(self, expected_version: int, current_version: int, current: dict | None):
        super().__init__(
            f"expected version {expected_version} but current version is {current_version}"
        )
        self.expected_version = expected_version
        self.current_version = current_version
        self.current = current


class PublishError(Exception):
    """The new version could not be rendered or registered."""


class PublishBusy(Exception):
    """Another publication held the write lock for longer than the timeout."""


@dataclass(frozen=True)
class DownloadRecord:
    version: int
    day: str
    path: Path
    filename: str
    sha256: str
    size_bytes: int


def connect(settings: Settings) -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(settings.db_path),
        timeout=settings.sqlite_timeout_seconds,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {int(settings.sqlite_timeout_seconds * 1000)}")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_store(settings: Settings) -> None:
    """Create directories/schema, recover the pointer and clean leftovers."""

    settings.ensure_directories()
    conn = connect(settings)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO state (key, value) VALUES (?, ?)",
            (CURRENT_STATE_KEY, "0"),
        )
        stored = _read_pointer(conn)
        max_version = _max_version(conn)
        if stored not in (0,) and _fetch_row(conn, stored) is None:
            log.warning("current version %s is missing; recovering to %s", stored, max_version)
            _write_pointer(conn, max_version)
        elif stored > max_version:
            log.warning("current version %s exceeds stored history; recovering to %s", stored, max_version)
            _write_pointer(conn, max_version)
    finally:
        conn.close()

    _reconcile_files(settings)


def _reconcile_files(settings: Settings) -> None:
    now = time.time()
    for path in settings.tmp_dir.glob("*"):
        try:
            if path.is_file() and now - path.stat().st_mtime > ORPHAN_GRACE_SECONDS:
                path.unlink()
        except OSError:
            log.warning("could not remove stale temporary file %s", path, exc_info=True)

    conn = connect(settings)
    try:
        registered = {int(row["version"]) for row in conn.execute("SELECT version FROM versions")}
    finally:
        conn.close()

    for path in settings.versions_dir.glob("v*.tif"):
        try:
            version = int(path.stem[1:])
        except ValueError:
            continue
        if version not in registered and now - path.stat().st_mtime > ORPHAN_GRACE_SECONDS:
            try:
                path.unlink()
                log.warning("removed orphaned mosaic file %s", path)
            except OSError:
                log.warning("could not remove orphaned mosaic %s", path, exc_info=True)


def _read_pointer(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM state WHERE key = ?", (CURRENT_STATE_KEY,)
    ).fetchone()
    if row is None:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 0


def _write_pointer(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT INTO state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (CURRENT_STATE_KEY, str(version)),
    )


def _max_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM versions").fetchone()
    return int(row["v"])


def _fetch_row(conn: sqlite3.Connection, version: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM versions WHERE version = ?", (version,)).fetchone()


def _row_to_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "version": int(row["version"]),
        "day": row["day"],
        "status": row["status"],
        "published_at": row["published_at"],
        "image": {
            "width": int(row["width"]),
            "height": int(row["height"]),
            "dtype": row["dtype"],
            "nodata": int(row["nodata"]),
            "crs": row["crs_label"],
            "crs_wkt": row["crs_wkt"],
            "transform": json.loads(row["transform"]),
            "bounds": json.loads(row["bounds"]),
            "res_x": float(row["res_x"]),
            "res_y": float(row["res_y"]),
            "sha256": row["sha256"],
            "size_bytes": int(row["size_bytes"]),
        },
        "blocks": json.loads(row["block_meta"]),
        "download_url": f"/api/v1/mosaics/{int(row['version'])}/download",
    }


def _current_payload(conn: sqlite3.Connection) -> tuple[int, dict[str, Any] | None]:
    version = _read_pointer(conn)
    if version <= 0:
        return 0, None
    row = _fetch_row(conn, version)
    if row is None:
        return 0, None
    return version, _row_to_payload(row)


def get_current_version(settings: Settings) -> int:
    conn = connect(settings)
    try:
        return _read_pointer(conn)
    finally:
        conn.close()


def get_current(settings: Settings) -> tuple[int, dict[str, Any] | None]:
    conn = connect(settings)
    try:
        return _current_payload(conn)
    finally:
        conn.close()


def get_version(settings: Settings, version: int) -> dict[str, Any] | None:
    conn = connect(settings)
    try:
        row = _fetch_row(conn, version)
        return _row_to_payload(row) if row is not None else None
    finally:
        conn.close()


def list_versions(
    settings: Settings, *, day: str | None = None, limit: int = 20, offset: int = 0
) -> tuple[list[dict[str, Any]], int]:
    conn = connect(settings)
    try:
        if day is None:
            total = conn.execute("SELECT COUNT(*) AS n FROM versions").fetchone()["n"]
            rows = conn.execute(
                "SELECT * FROM versions ORDER BY version DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        else:
            total = conn.execute(
                "SELECT COUNT(*) AS n FROM versions WHERE day = ?", (day,)
            ).fetchone()["n"]
            rows = conn.execute(
                "SELECT * FROM versions WHERE day = ? ORDER BY version DESC LIMIT ? OFFSET ?",
                (day, limit, offset),
            ).fetchall()
        return [_row_to_payload(row) for row in rows], int(total)
    finally:
        conn.close()


def get_download(settings: Settings, version: int) -> DownloadRecord | None:
    conn = connect(settings)
    try:
        row = _fetch_row(conn, version)
        if row is None:
            return None
        return DownloadRecord(
            version=int(row["version"]),
            day=row["day"],
            path=settings.versions_dir / f"v{int(row['version'])}.tif",
            filename=f"rainmosaic-{row['day']}-v{int(row['version'])}.tif",
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
        )
    finally:
        conn.close()


def get_current_download(settings: Settings) -> DownloadRecord | None:
    conn = connect(settings)
    try:
        version, _ = _current_payload(conn)
        if version <= 0:
            return None
        row = _fetch_row(conn, version)
        if row is None:
            return None
        return DownloadRecord(
            version=int(row["version"]),
            day=row["day"],
            path=settings.versions_dir / f"v{int(row['version'])}.tif",
            filename=f"rainmosaic-{row['day']}-v{int(row['version'])}.tif",
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
        )
    finally:
        conn.close()


def publish(
    settings: Settings,
    *,
    day: str,
    expected_version: int,
    left: SourceBlock,
    right: SourceBlock,
) -> dict[str, Any]:
    """Render and publish a new version atomically."""

    temp_path = settings.tmp_dir / f"publish-{uuid.uuid4().hex}.tif"
    final_path: Path | None = None
    new_version: int | None = None
    conn = connect(settings)
    try:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise PublishBusy() from exc
            raise
        current_version, current_payload = _current_payload(conn)
        if current_version != expected_version:
            raise VersionConflict(expected_version, current_version, current_payload)

        new_version = max(current_version, _max_version(conn)) + 1

        # Heavy work happens while the write lock is held.  Readers in WAL
        # mode keep seeing and downloading the old current version.
        info = rasters.write_mosaic(left, right, temp_path)
        rasters.verify_mosaic(temp_path, info)
        _fsync_file(temp_path)
        sha256 = rasters.sha256_file(temp_path)
        size_bytes = temp_path.stat().st_size

        final_path = settings.versions_dir / f"v{new_version}.tif"
        os.replace(temp_path, final_path)
        _fsync_dir(settings.versions_dir)

        published_at = iso_utc(datetime.now(timezone.utc))
        block_meta = [left.metadata(), right.metadata()]
        conn.execute(
            """
            INSERT INTO versions (
                version, day, status, published_at, width, height, crs_wkt,
                crs_label, transform, bounds, res_x, res_y, dtype, nodata,
                sha256, size_bytes, block_meta
            ) VALUES (?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_version,
                day,
                published_at,
                info.width,
                info.height,
                info.crs.to_wkt(),
                info.crs.to_string() or info.crs.to_wkt(),
                json.dumps(list(info.transform)[:6]),
                json.dumps(list(info.bounds)),
                info.res_x,
                info.res_y,
                rasters.DTYPE,
                rasters.NODATA,
                sha256,
                size_bytes,
                json.dumps(block_meta),
            ),
        )
        _write_pointer(conn, new_version)
        row = _fetch_row(conn, new_version)
        assert row is not None
        conn.execute("COMMIT")
        log.info(
            "published version %s for %s (%sx%s, %s bytes)",
            new_version,
            day,
            info.width,
            info.height,
            size_bytes,
        )
        payload = _row_to_payload(row)
        return payload
    except VersionConflict:
        _rollback(conn)
        raise
    except PublishBusy:
        _rollback(conn)
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        _rollback(conn)
        temp_path.unlink(missing_ok=True)
        if final_path is not None and new_version is not None:
            # Only remove the renamed file if the version really is not in the
            # database (guards against an ambiguous commit outcome).
            try:
                row = _fetch_row(conn, new_version)
            except sqlite3.Error:
                row = None
            if row is None:
                final_path.unlink(missing_ok=True)
        log.exception("publishing mosaic failed")
        raise PublishError("mosaic generation or registration failed") from exc
    finally:
        conn.close()


def _rollback(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            log.warning("rollback failed", exc_info=True)


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Station interpretation reports
# ---------------------------------------------------------------------------

# A report always uses one fixed set of version numbers selected in a single
# read snapshot.  ``snapshot_version`` is the largest version that existed
# when the selection was made; because version numbers only grow, comparing
# these values totally orders competing reports for the same parameters.


class ReportSelectionError(Exception):
    """The version set for a report could not be selected."""


def analysis_params_key(
    *,
    longitude: float,
    latitude: float,
    start_day: str,
    end_day: str,
    threshold_mm: float,
) -> str:
    """Canonical key for "the same parameters" of an interpretation report."""

    def normalise(value: float) -> float:
        # Collapse -0.0 to 0.0 and make 1 and 1.0 compare identically.
        return float(value) + 0.0

    return json.dumps(
        [
            normalise(longitude),
            normalise(latitude),
            start_day,
            end_day,
            normalise(threshold_mm),
        ],
        separators=(",", ":"),
    )


def _latest_versions(
    conn: sqlite3.Connection,
    days: list[str],
    cache: dict[str, int | None] | None = None,
) -> dict[str, int | None]:
    latest: dict[str, int | None] = {}
    for day in days:
        if cache is not None and day in cache:
            latest[day] = cache[day]
            continue
        row = conn.execute(
            "SELECT MAX(version) AS version FROM versions WHERE day = ?", (day,)
        ).fetchone()
        value = int(row["version"]) if row["version"] is not None else None
        latest[day] = value
        if cache is not None:
            cache[day] = value
    return latest


def select_versions(
    settings: Settings, days: list[str]
) -> tuple[int, dict[str, int | None], str]:
    """Pick the newest version per day from one consistent database snapshot.

    The deferred read transaction holds one WAL snapshot for all queries, so a
    publication committing halfway through can never contribute a version to
    only part of the selection.  ``snapshot_version`` is the highest version
    number in that snapshot.
    """

    conn = connect(settings)
    try:
        try:
            conn.execute("BEGIN")
        except sqlite3.OperationalError as exc:
            raise ReportSelectionError("could not open a selection snapshot") from exc
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM versions"
            ).fetchone()
            snapshot_version = int(row["version"])
            selected = _latest_versions(conn, days)
            snapshot_at = iso_utc(datetime.now(timezone.utc))
            conn.execute("COMMIT")
        except Exception:
            _rollback(conn)
            raise
        return snapshot_version, selected, snapshot_at
    except sqlite3.Error as exc:
        raise ReportSelectionError("could not select published versions") from exc
    finally:
        conn.close()


def _report_row_to_payload(
    row: sqlite3.Row,
    *,
    current_report_id: int | None,
    latest: dict[str, int | None],
) -> dict[str, Any]:
    days: list[str] = json.loads(row["requested_days"])
    selected: dict[str, int | None] = json.loads(row["selected_versions"])
    events = json.loads(row["events"])
    stale_days = [day for day in days if selected.get(day) != latest.get(day)]
    return {
        "report_id": int(row["report_id"]),
        "parameters": {
            "longitude": float(row["longitude"]),
            "latitude": float(row["latitude"]),
            "start_day": row["start_day"],
            "end_day": row["end_day"],
            "threshold_mm": float(row["threshold_mm"]),
        },
        "basis": {
            "snapshot_version": int(row["snapshot_version"]),
            "selected_at": row["snapshot_at"],
            "days": [
                {"day": day, "version": selected.get(day)} for day in days
            ],
            "versions": {day: selected.get(day) for day in days},
        },
        "events": events,
        "event_count": len(events),
        "days": json.loads(row["evidence"]),
        "created_at": row["created_at"],
        "is_current": current_report_id is not None and int(row["report_id"]) == current_report_id,
        "stale": bool(stale_days),
        "stale_days": stale_days,
    }


def _report_summary(
    row: sqlite3.Row,
    *,
    current_report_id: int | None,
    latest: dict[str, int | None],
) -> dict[str, Any]:
    payload = _report_row_to_payload(
        row, current_report_id=current_report_id, latest=latest
    )
    return {
        "report_id": payload["report_id"],
        "parameters": payload["parameters"],
        "event_count": payload["event_count"],
        "event_days": [
            [event["start_day"], event["end_day"]] for event in payload["events"]
        ],
        "snapshot_version": payload["basis"]["snapshot_version"],
        "selected_at": payload["basis"]["selected_at"],
        "created_at": payload["created_at"],
        "is_current": payload["is_current"],
        "stale": payload["stale"],
        "stale_days": payload["stale_days"],
    }


def save_report(
    settings: Settings,
    *,
    longitude: float,
    latitude: float,
    start_day: str,
    end_day: str,
    threshold_mm: float,
    days: list[str],
    snapshot_version: int,
    snapshot_at: str,
    selected: dict[str, int | None],
    evidence: list[dict],
    events: list[dict],
) -> dict[str, Any]:
    """Persist one complete report and update the per-parameter conclusion.

    The report is only written after its evidence has been computed, so a
    failure leaves neither a partial report nor a moved conclusion pointer.
    A report computed from an older snapshot than the conclusion currently on
    file is stored and remains queryable, but never becomes the conclusion:
    a late-finishing computation cannot overwrite a newer one.
    """

    params_key = analysis_params_key(
        longitude=longitude,
        latitude=latitude,
        start_day=start_day,
        end_day=end_day,
        threshold_mm=threshold_mm,
    )
    created_at = iso_utc(datetime.now(timezone.utc))
    conn = connect(settings)
    try:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise PublishBusy() from exc
            raise
        try:
            existing = conn.execute(
                "SELECT report_id, snapshot_version FROM analysis_current WHERE params_key = ?",
                (params_key,),
            ).fetchone()
            is_current = existing is None or snapshot_version > int(existing["snapshot_version"])

            cursor = conn.execute(
                """
                INSERT INTO analyses (
                    longitude, latitude, start_day, end_day, threshold_mm,
                    params_key, requested_days, selected_versions,
                    snapshot_version, snapshot_at, evidence, events, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    longitude,
                    latitude,
                    start_day,
                    end_day,
                    threshold_mm,
                    params_key,
                    json.dumps(days),
                    json.dumps(selected),
                    snapshot_version,
                    snapshot_at,
                    json.dumps(evidence),
                    json.dumps(events),
                    created_at,
                ),
            )
            report_id = int(cursor.lastrowid)
            if is_current:
                conn.execute(
                    """
                    INSERT INTO analysis_current (
                        params_key, report_id, snapshot_version, snapshot_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(params_key) DO UPDATE SET
                        report_id = excluded.report_id,
                        snapshot_version = excluded.snapshot_version,
                        snapshot_at = excluded.snapshot_at
                    """,
                    (
                        params_key,
                        report_id,
                        snapshot_version,
                        snapshot_at,
                    ),
                )
            latest = _latest_versions(conn, days)
            current_row = conn.execute(
                "SELECT report_id FROM analysis_current WHERE params_key = ?",
                (params_key,),
            ).fetchone()
            current_report_id = (
                int(current_row["report_id"]) if current_row is not None else None
            )
            row = conn.execute(
                "SELECT * FROM analyses WHERE report_id = ?", (report_id,)
            ).fetchone()
            assert row is not None
            conn.execute("COMMIT")
        except Exception:
            _rollback(conn)
            raise
        log.info(
            "saved interpretation report %s for %s..%s (snapshot %s, current=%s)",
            report_id,
            start_day,
            end_day,
            snapshot_version,
            is_current,
        )
        return _report_row_to_payload(
            row, current_report_id=current_report_id, latest=latest
        )
    finally:
        conn.close()


@contextlib.contextmanager
def _read_snapshot(conn: sqlite3.Connection):
    """Hold one consistent WAL read snapshot for a group of queries."""

    conn.execute("BEGIN")
    try:
        yield
    except Exception:
        _rollback(conn)
        raise
    else:
        conn.execute("COMMIT")


def get_report(settings: Settings, report_id: int) -> dict[str, Any] | None:
    conn = connect(settings)
    try:
        with _read_snapshot(conn):
            row = conn.execute(
                "SELECT * FROM analyses WHERE report_id = ?", (report_id,)
            ).fetchone()
            if row is None:
                return None
            days: list[str] = json.loads(row["requested_days"])
            latest = _latest_versions(conn, days)
            current = conn.execute(
                "SELECT report_id FROM analysis_current WHERE params_key = ?",
                (row["params_key"],),
            ).fetchone()
            current_report_id = int(current["report_id"]) if current is not None else None
            return _report_row_to_payload(
                row, current_report_id=current_report_id, latest=latest
            )
    finally:
        conn.close()


def get_current_report(
    settings: Settings,
    *,
    longitude: float,
    latitude: float,
    start_day: str,
    end_day: str,
    threshold_mm: float,
) -> dict[str, Any] | None:
    params_key = analysis_params_key(
        longitude=longitude,
        latitude=latitude,
        start_day=start_day,
        end_day=end_day,
        threshold_mm=threshold_mm,
    )
    conn = connect(settings)
    try:
        with _read_snapshot(conn):
            pointer = conn.execute(
                "SELECT report_id FROM analysis_current WHERE params_key = ?", (params_key,)
            ).fetchone()
            if pointer is None:
                return None
            row = conn.execute(
                "SELECT * FROM analyses WHERE report_id = ?", (int(pointer["report_id"]),)
            ).fetchone()
            if row is None:  # defensive: a pointer without its report is not a conclusion
                return None
            days: list[str] = json.loads(row["requested_days"])
            latest = _latest_versions(conn, days)
            return _report_row_to_payload(
                row, current_report_id=int(pointer["report_id"]), latest=latest
            )
    finally:
        conn.close()


def list_reports(
    settings: Settings,
    *,
    params_key: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    conn = connect(settings)
    try:
        with _read_snapshot(conn):
            if params_key is None:
                total = conn.execute("SELECT COUNT(*) AS n FROM analyses").fetchone()["n"]
                rows = conn.execute(
                    "SELECT * FROM analyses ORDER BY report_id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                total = conn.execute(
                    "SELECT COUNT(*) AS n FROM analyses WHERE params_key = ?", (params_key,)
                ).fetchone()["n"]
                rows = conn.execute(
                    "SELECT * FROM analyses WHERE params_key = ? "
                    "ORDER BY report_id DESC LIMIT ? OFFSET ?",
                    (params_key, limit, offset),
                ).fetchall()

            pointers = {
                row["params_key"]: int(row["report_id"])
                for row in conn.execute("SELECT params_key, report_id FROM analysis_current")
            }
            items: list[dict[str, Any]] = []
            latest_cache: dict[str, int | None] = {}
            for row in rows:
                days: list[str] = json.loads(row["requested_days"])
                latest = _latest_versions(conn, days, latest_cache)
                current_report_id = pointers.get(row["params_key"])
                items.append(
                    _report_summary(
                        row, current_report_id=current_report_id, latest=latest
                    )
                )
            return items, int(total)
    finally:
        conn.close()

