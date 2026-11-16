"""Station interpretation: daily version selection, pixel lookup, event runs."""

from __future__ import annotations

import numpy as np

from app import analysis, store
from app.rasters import NODATA
from tests.conftest import make_geotiff, submit_payload

# Station at (0.005, 0.015) falls in the first pixel of a 1x1 block whose
# origin is (0.0, 0.02) with 0.01 degree pixels.
STATION_LON = 0.005
STATION_LAT = 0.015


def _publish(
    client,
    *,
    day: str,
    value: int | None,
    expected_version: int,
    crs: str = "EPSG:4326",
    x0: float = 0.0,
    y0: float = 0.02,
    res: float = 0.01,
    a_time: str | None = None,
    b_time: str | None = None,
):
    """Publish one day whose first block carries ``value`` at the station."""

    pixel = NODATA if value is None else value
    a = np.array([[pixel]], dtype=np.int16)
    b = np.array([[NODATA]], dtype=np.int16)
    stamp = a_time or f"{day}T01:00:00Z"
    later = b_time or f"{day}T02:00:00Z"
    data, files = submit_payload(
        block_a=make_geotiff(a, crs=crs, x0=x0, y0=y0, res=res),
        block_b=make_geotiff(b, crs=crs, x0=x0 + res, y0=y0, res=res),
        expected_version=expected_version,
        day=day,
        a_time=stamp,
        b_time=later,
    )
    response = client.post("/api/v1/mosaics", data=data, files=files)
    assert response.status_code == 201, response.text
    return response.json()


def _analyse(client, *, start="2026-01-01", end="2026-01-04", threshold=50.0, **params):
    body = {
        "longitude": STATION_LON,
        "latitude": STATION_LAT,
        "start_day": start,
        "end_day": end,
        "threshold_mm": threshold,
    }
    body.update(params)
    return client.post("/api/v1/analyses", json=body)


def test_two_consecutive_days_form_event_with_versions(client):
    versions = {}
    for index, value in enumerate((60, 80, 10, 5)):
        day = f"2026-01-0{index + 1}"
        versions[day] = _publish(
            client, day=day, value=value, expected_version=index
        )["version"]

    response = _analyse(client)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["report_id"] == 1
    assert body["is_current"] is True
    assert body["stale"] is False
    assert body["event_count"] == 1
    event = body["events"][0]
    assert event["start_day"] == "2026-01-01"
    assert event["end_day"] == "2026-01-02"
    assert event["length_days"] == 2
    assert event["total_mm"] == 140
    assert [item["day"] for item in event["daily"]] == ["2026-01-01", "2026-01-02"]
    assert [item["value_mm"] for item in event["daily"]] == [60, 80]
    assert [item["version"] for item in event["daily"]] == [
        versions["2026-01-01"],
        versions["2026-01-02"],
    ]
    assert body["basis"]["days"] == [
        {"day": day, "version": versions[day]} for day in sorted(versions)
    ]
    assert [item["status"] for item in body["days"]] == ["ok", "ok", "ok", "ok"]

    stored = client.get(f"/api/v1/analyses/{body['report_id']}")
    assert stored.status_code == 200
    assert stored.json()["report"]["events"] == body["events"]
    current = client.get(
        "/api/v1/analyses/current",
        params={
            "longitude": STATION_LON,
            "latitude": STATION_LAT,
            "start_day": "2026-01-01",
            "end_day": "2026-01-04",
            "threshold_mm": 50,
        },
    )
    assert current.status_code == 200
    assert current.json()["report"]["report_id"] == body["report_id"]


def test_single_qualifying_day_is_not_an_event(client):
    _publish(client, day="2026-01-01", value=80, expected_version=0)
    _publish(client, day="2026-01-02", value=10, expected_version=1)
    response = _analyse(client, end="2026-01-02")
    assert response.status_code == 201
    assert response.json()["events"] == []


def test_missing_day_breaks_a_run(client):
    _publish(client, day="2026-01-01", value=80, expected_version=0)
    _publish(client, day="2026-01-03", value=80, expected_version=1)
    response = _analyse(client, start="2026-01-01", end="2026-01-03")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["events"] == []
    by_day = {item["day"]: item for item in body["days"]}
    assert by_day["2026-01-02"]["status"] == "missing_version"
    assert by_day["2026-01-02"]["version"] is None


def test_nodata_pixel_breaks_a_run(client):
    _publish(client, day="2026-01-01", value=80, expected_version=0)
    _publish(client, day="2026-01-02", value=None, expected_version=1)
    _publish(client, day="2026-01-03", value=80, expected_version=2)
    response = _analyse(client, end="2026-01-03")
    assert response.status_code == 201
    body = response.json()
    assert body["events"] == []
    assert [item["status"] for item in body["days"]] == ["ok", "nodata", "ok"]
    assert body["days"][1]["value_mm"] is None


def test_station_outside_raster_breaks_a_run(client):
    _publish(client, day="2026-01-01", value=80, expected_version=0)
    # Second day's raster is far away from the station.
    _publish(
        client, day="2026-01-02", value=80, expected_version=1, x0=40.0, y0=40.0
    )
    _publish(client, day="2026-01-03", value=80, expected_version=2)
    response = _analyse(client, end="2026-01-03")
    assert response.status_code == 201
    body = response.json()
    assert body["events"] == []
    assert [item["status"] for item in body["days"]] == ["ok", "out_of_bounds", "ok"]


def test_each_day_uses_its_newest_version(client):
    _publish(client, day="2026-01-01", value=10, expected_version=0)
    _publish(client, day="2026-01-02", value=80, expected_version=1)
    newest = _publish(client, day="2026-01-01", value=90, expected_version=2)
    response = _analyse(client, end="2026-01-02")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["basis"]["versions"]["2026-01-01"] == newest["version"]
    assert body["days"][0]["value_mm"] == 90
    assert body["events"][0]["total_mm"] == 170


def test_station_is_located_in_each_version_crs(client):
    # Build a Web Mercator raster that contains the station.
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(
        STATION_LON, STATION_LAT
    )
    res = 100.0
    x0 = res * np.floor(x / res)
    y0 = res * np.floor(y / res) + res
    _publish(
        client,
        day="2026-01-01",
        value=70,
        expected_version=0,
        crs="EPSG:3857",
        x0=x0,
        y0=y0,
        res=res,
    )
    _publish(
        client,
        day="2026-01-02",
        value=70,
        expected_version=1,
        crs="EPSG:3857",
        x0=x0,
        y0=y0,
        res=res,
    )
    response = _analyse(client, end="2026-01-02")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["events"][0]["total_mm"] == 140
    assert body["days"][0]["crs"] == "EPSG:3857"
    assert body["days"][0]["row"] == 0 and body["days"][0]["col"] == 0


def test_threshold_is_inclusive(client):
    _publish(client, day="2026-01-01", value=50, expected_version=0)
    _publish(client, day="2026-01-02", value=50, expected_version=1)
    response = _analyse(client, end="2026-01-02", threshold=50)
    assert response.status_code == 201
    assert response.json()["events"][0]["total_mm"] == 100
    below = _analyse(client, end="2026-01-02", threshold=50.5)
    assert below.status_code == 201
    assert below.json()["events"] == []


def test_report_parameters_are_validated(client):
    too_long = _analyse(client, start="2026-01-01", end="2026-01-15")
    assert too_long.status_code == 422
    assert too_long.json()["detail"]["code"] == "day_range_too_long"

    reversed_range = _analyse(client, start="2026-01-05", end="2026-01-01")
    assert reversed_range.status_code == 422
    assert reversed_range.json()["detail"]["code"] == "invalid_day_range"

    bad_day = _analyse(client, start="01-01-2026", end="2026-01-02")
    assert bad_day.status_code == 422
    assert bad_day.json()["detail"]["code"] == "invalid_day"

    out_of_range = _analyse(client, longitude=200.0)
    assert out_of_range.status_code == 422
    assert out_of_range.json()["detail"]["code"] == "invalid_request"

    negative_threshold = _analyse(client, threshold=-1)
    assert negative_threshold.status_code == 422

    extra_field = client.post(
        "/api/v1/analyses",
        json={
            "longitude": STATION_LON,
            "latitude": STATION_LAT,
            "start_day": "2026-01-01",
            "end_day": "2026-01-02",
            "threshold_mm": 10,
            "unexpected": True,
        },
    )
    assert extra_field.status_code == 422


def test_fourteen_day_range_is_allowed(client):
    response = _analyse(client, start="2026-01-01", end="2026-01-14")
    assert response.status_code == 201
    assert len(response.json()["days"]) == 14


def test_backfilled_old_day_marks_old_report_stale(client):
    _publish(client, day="2026-01-01", value=80, expected_version=0)
    _publish(client, day="2026-01-02", value=80, expected_version=1)
    original = _analyse(client, end="2026-01-02").json()
    assert original["stale"] is False

    # Backfill a later-day version for the old date.
    _publish(client, day="2026-01-01", value=5, expected_version=2)

    old = client.get(f"/api/v1/analyses/{original['report_id']}")
    assert old.status_code == 200
    payload = old.json()["report"]
    assert payload["stale"] is True
    assert payload["stale_days"] == ["2026-01-01"]
    # The old report still carries exactly the evidence it was computed from.
    assert payload["events"][0]["total_mm"] == 160
    assert payload["is_current"] is True

    refreshed = _analyse(client, end="2026-01-02")
    assert refreshed.status_code == 201
    refreshed_body = refreshed.json()
    assert refreshed_body["stale"] is False
    assert refreshed_body["events"] == []
    current = client.get(
        "/api/v1/analyses/current",
        params={
            "longitude": STATION_LON,
            "latitude": STATION_LAT,
            "start_day": "2026-01-01",
            "end_day": "2026-01-02",
            "threshold_mm": 50,
        },
    ).json()["report"]
    assert current["report_id"] == refreshed_body["report_id"]
    # The superseded report remains queryable and marked stale.
    assert client.get(f"/api/v1/analyses/{original['report_id']}").json()["report"][
        "is_current"
    ] is False


def test_late_older_basis_report_does_not_replace_newer_conclusion(client):
    _publish(client, day="2026-01-01", value=60, expected_version=0)
    _publish(client, day="2026-01-02", value=70, expected_version=1)

    settings = client.app.state.settings
    days = ["2026-01-01", "2026-01-02"]
    old_snapshot, old_selected, old_at = store.select_versions(settings, days)
    _publish(client, day="2026-01-01", value=10, expected_version=2)
    new_snapshot, new_selected, new_at = store.select_versions(settings, days)
    assert new_snapshot > old_snapshot

    new_evidence, new_events = analysis.compute(
        settings,
        longitude=STATION_LON,
        latitude=STATION_LAT,
        days=days,
        selected_versions={
            day: version for day, version in new_selected.items() if version is not None
        },
        threshold_mm=50,
    )
    newer = store.save_report(
        settings,
        longitude=STATION_LON,
        latitude=STATION_LAT,
        start_day=days[0],
        end_day=days[-1],
        threshold_mm=50,
        days=days,
        snapshot_version=new_snapshot,
        snapshot_at=new_at,
        selected=new_selected,
        evidence=new_evidence,
        events=new_events,
    )
    assert newer["is_current"] is True
    assert newer["events"] == []
    # A computation that selected the older snapshot finishes afterwards.
    old_evidence, old_events = analysis.compute(
        settings,
        longitude=STATION_LON,
        latitude=STATION_LAT,
        days=days,
        selected_versions={
            day: version for day, version in old_selected.items() if version is not None
        },
        threshold_mm=50,
    )
    late = store.save_report(
        settings,
        longitude=STATION_LON,
        latitude=STATION_LAT,
        start_day=days[0],
        end_day=days[-1],
        threshold_mm=50,
        days=days,
        snapshot_version=old_snapshot,
        snapshot_at=old_at,
        selected=old_selected,
        evidence=old_evidence,
        events=old_events,
    )
    # It is stored and queryable, but must not replace the newer conclusion.
    assert late["is_current"] is False
    assert late["events"][0]["total_mm"] == 130
    stored_late = store.get_report(settings, late["report_id"])
    assert stored_late is not None and stored_late["event_count"] == 1
    current = store.get_current_report(
        settings,
        longitude=STATION_LON,
        latitude=STATION_LAT,
        start_day=days[0],
        end_day=days[-1],
        threshold_mm=50,
    )
    assert current is not None and current["report_id"] == newer["report_id"]


def test_failed_computation_leaves_no_report(client, monkeypatch):
    _publish(client, day="2026-01-01", value=80, expected_version=0)

    def broken(*args, **kwargs):
        raise analysis.AnalysisError("simulated unreadable version")

    monkeypatch.setattr(analysis, "read_day", broken)
    response = _analyse(client, end="2026-01-01")
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "analysis_failed"
    assert client.get("/api/v1/analyses").json()["total"] == 0


def test_failed_storage_leaves_no_report_or_moved_conclusion(client, monkeypatch):
    _publish(client, day="2026-01-01", value=60, expected_version=0)
    _publish(client, day="2026-01-02", value=70, expected_version=1)
    first = _analyse(client, end="2026-01-02").json()
    assert first["is_current"] is True

    _publish(client, day="2026-01-01", value=90, expected_version=2)
    real_save = store.save_report

    def half_written(*args, **kwargs):
        # Simulate a failure after the evidence was computed; nothing may be
        # committed by the caller.
        raise store.PublishError("simulated storage failure")

    monkeypatch.setattr(store, "save_report", half_written)
    failed = _analyse(client, end="2026-01-02")
    assert failed.status_code == 500
    assert failed.json()["detail"]["code"] == "analysis_failed"
    monkeypatch.setattr(store, "save_report", real_save)

    # No extra report exists and the earlier conclusion is untouched.
    assert client.get("/api/v1/analyses").json()["total"] == 1
    current = client.get(
        "/api/v1/analyses/current",
        params={
            "longitude": STATION_LON,
            "latitude": STATION_LAT,
            "start_day": "2026-01-01",
            "end_day": "2026-01-02",
            "threshold_mm": 50,
        },
    ).json()["report"]
    assert current["report_id"] == first["report_id"]
    assert current["events"] == first["events"]


def test_report_selection_survives_publication_mid_computation(client, monkeypatch):
    for index in range(3):
        _publish(client, day=f"2026-01-0{index + 1}", value=60, expected_version=index)

    settings = client.app.state.settings
    real_read = analysis.read_day
    published = []

    def read_with_publish(*args, **kwargs):
        if not published:
            # A publication lands while this report is being computed.
            result = _publish(client, day="2026-01-02", value=99, expected_version=3)
            published.append(result["version"])
        return real_read(*args, **kwargs)

    monkeypatch.setattr(analysis, "read_day", read_with_publish)
    response = _analyse(client, end="2026-01-03")
    assert response.status_code == 201, response.text
    body = response.json()
    assert published == [4]
    # The report uses the pre-publication version of the day, not version 4.
    assert body["basis"]["versions"]["2026-01-02"] == 2
    assert body["days"][1]["value_mm"] == 60
    assert body["events"][0]["total_mm"] == 180


def test_listing_filters_by_parameters_and_paginates(client):
    _publish(client, day="2026-01-01", value=60, expected_version=0)
    _publish(client, day="2026-01-02", value=60, expected_version=1)
    first = _analyse(client, end="2026-01-02").json()
    second = _analyse(client, end="2026-01-02").json()
    other = _analyse(client, end="2026-01-02", threshold=10).json()

    listing = client.get("/api/v1/analyses").json()
    assert listing["total"] == 3
    assert [item["report_id"] for item in listing["items"]] == [
        other["report_id"],
        second["report_id"],
        first["report_id"],
    ]

    filtered = client.get(
        "/api/v1/analyses",
        params={
            "longitude": STATION_LON,
            "latitude": STATION_LAT,
            "start_day": "2026-01-01",
            "end_day": "2026-01-02",
            "threshold_mm": 50,
        },
    ).json()
    assert filtered["total"] == 2
    # Identical parameters and the same snapshot: the earlier stored report
    # stays the conclusion, even though listing shows newest first.
    assert filtered["items"][0]["is_current"] is False
    assert filtered["items"][1]["is_current"] is True

    page = client.get("/api/v1/analyses", params={"limit": 1, "offset": 1}).json()
    assert page["total"] == 3
    assert len(page["items"]) == 1

    incomplete = client.get("/api/v1/analyses", params={"longitude": STATION_LON})
    assert incomplete.status_code == 422
    assert incomplete.json()["detail"]["code"] == "incomplete_parameters"

    assert client.get("/api/v1/analyses/999").status_code == 404
    missing_current = client.get(
        "/api/v1/analyses/current",
        params={
            "longitude": 10,
            "latitude": 10,
            "start_day": "2026-01-01",
            "end_day": "2026-01-02",
            "threshold_mm": 50,
        },
    )
    assert missing_current.status_code == 404


def test_reports_survive_restart(restartable_client):
    first = restartable_client()
    _publish(first, day="2026-01-01", value=60, expected_version=0)
    _publish(first, day="2026-01-02", value=70, expected_version=1)
    created = _analyse(first, end="2026-01-02").json()
    first.__exit__(None, None, None)

    restarted = restartable_client()
    fetched = restarted.get(f"/api/v1/analyses/{created['report_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["report"]["events"] == created["events"]
    assert restarted.get("/api/v1/analyses").json()["total"] == 1


def test_publishing_and_downloads_still_work_with_reports(client):
    published = _publish(client, day="2026-01-01", value=60, expected_version=0)
    _analyse(client, end="2026-01-01")
    assert client.get("/api/v1/mosaics/current").status_code == 200
    download = client.get(f"/api/v1/mosaics/{published['version']}/download")
    assert download.status_code == 200
    assert client.get("/docs").status_code == 200
    openapi = client.get("/openapi.json").json()
    assert "/api/v1/analyses" in openapi["paths"]
    assert "/api/v1/analyses/{report_id}" in openapi["paths"]
