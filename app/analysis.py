"""Station interpretation over published daily rainfall mosaics.

A report is computed from one fixed *snapshot* of published versions: the
caller first selects, for every day in the requested range, the newest
published version of that day in a single database snapshot.  This module then
only ever opens those version files, so publications that happen while a
report is being computed can never change the evidence a report is based on.

For every day the station's WGS 84 longitude/latitude is transformed into the
selected version's own CRS, the containing pixel is located with the raster's
affine transform and that one pixel is read.  A day is only usable when a
version exists, the station falls inside the raster and the pixel is not
NoData; any other outcome breaks a continuous run.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Mapping

import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from .config import Settings
from .rasters import NODATA

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"

STATUS_OK = "ok"
STATUS_MISSING_VERSION = "missing_version"
STATUS_OUT_OF_BOUNDS = "out_of_bounds"
STATUS_NODATA = "nodata"

MAX_RANGE_DAYS = 14


class AnalysisError(Exception):
    """A report could not be computed from the selected versions."""


def date_range(start_day: str, end_day: str) -> list[str]:
    """Every UTC day from ``start_day`` to ``end_day``, inclusive."""

    first = date.fromisoformat(start_day)
    last = date.fromisoformat(end_day)
    return [(first + timedelta(days=offset)).isoformat() for offset in range((last - first).days + 1)]


def read_day(
    settings: Settings,
    *,
    day: str,
    version: int | None,
    longitude: float,
    latitude: float,
) -> dict:
    """Locate the station in one selected version and read its pixel."""

    if version is None:
        return _day_evidence(day, STATUS_MISSING_VERSION, version=None)

    path = settings.versions_dir / f"v{version}.tif"
    try:
        with rasterio.open(path) as dataset:
            crs_label = (
                dataset.crs.to_string()
                if dataset.crs is not None and dataset.crs.to_string()
                else (dataset.crs.to_wkt() if dataset.crs is not None else None)
            )
            if dataset.crs is None:
                raise AnalysisError(f"version {version} for {day} has no CRS")

            try:
                transformer = Transformer.from_crs(WGS84, dataset.crs, always_xy=True)
                x, y = transformer.transform(longitude, latitude)
            except Exception:
                # The station cannot be expressed in this version's CRS; for
                # this report that is equivalent to the station lying outside
                # the raster.
                log.warning(
                    "could not transform station into CRS of version %s (%s)",
                    version,
                    day,
                    exc_info=True,
                )
                return _day_evidence(day, STATUS_OUT_OF_BOUNDS, version=version, crs=crs_label)

            if not (math.isfinite(x) and math.isfinite(y)):
                return _day_evidence(day, STATUS_OUT_OF_BOUNDS, version=version, crs=crs_label)

            try:
                col_f, row_f = (~dataset.transform) * (x, y)
            except Exception as exc:  # non-invertible transform should never be published
                raise AnalysisError(
                    f"version {version} for {day} has an unusable affine transform"
                ) from exc
            if not (math.isfinite(col_f) and math.isfinite(row_f)):
                return _day_evidence(day, STATUS_OUT_OF_BOUNDS, version=version, crs=crs_label)

            # Half-open pixel extents: the top/left edges belong to the pixel,
            # the bottom/right edges belong to the next one (or outside).
            col = math.floor(col_f)
            row = math.floor(row_f)
            if col < 0 or row < 0 or col >= dataset.width or row >= dataset.height:
                return _day_evidence(day, STATUS_OUT_OF_BOUNDS, version=version, crs=crs_label)

            pixel = dataset.read(1, window=Window(col, row, 1, 1))[0, 0]
            nodata = dataset.nodata if dataset.nodata is not None else NODATA
            if int(pixel) == int(nodata):
                return _day_evidence(
                    day,
                    STATUS_NODATA,
                    version=version,
                    row=int(row),
                    col=int(col),
                    crs=crs_label,
                )
            return _day_evidence(
                day,
                STATUS_OK,
                version=version,
                value_mm=int(pixel),
                row=int(row),
                col=int(col),
                crs=crs_label,
            )
    except AnalysisError:
        raise
    except Exception as exc:
        raise AnalysisError(f"version {version} selected for {day} could not be read") from exc


def _day_evidence(
    day: str,
    status: str,
    *,
    version: int | None,
    value_mm: int | None = None,
    row: int | None = None,
    col: int | None = None,
    crs: str | None = None,
) -> dict:
    return {
        "day": day,
        "status": status,
        "version": version,
        "value_mm": value_mm,
        "row": row,
        "col": col,
        "crs": crs,
    }


def compute(
    settings: Settings,
    *,
    longitude: float,
    latitude: float,
    days: list[str],
    selected_versions: Mapping[str, int],
    threshold_mm: float,
) -> tuple[list[dict], list[dict]]:
    """Read every day and derive events from consecutive qualifying days."""

    evidence = [
        read_day(
            settings,
            day=day,
            version=selected_versions.get(day),
            longitude=longitude,
            latitude=latitude,
        )
        for day in days
    ]
    events: list[dict] = []
    run: list[dict] = []

    def close_run() -> None:
        if len(run) < 2:
            return
        events.append(
            {
                "start_day": run[0]["day"],
                "end_day": run[-1]["day"],
                "length_days": len(run),
                "total_mm": sum(item["value_mm"] for item in run),
                "daily": [
                    {
                        "day": item["day"],
                        "value_mm": item["value_mm"],
                        "version": item["version"],
                    }
                    for item in run
                ],
            }
        )

    for item in evidence:
        qualifies = (
            item["status"] == STATUS_OK
            and item["value_mm"] is not None
            and item["value_mm"] >= threshold_mm
        )
        if qualifies:
            run.append(item)
        else:
            close_run()
            run = []
    close_run()
    return evidence, events
