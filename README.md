# Rainfall Mosaic Service

Combines two same-day, adjacent single-band rainfall GeoTIFF blocks into one
research-area mosaic that models can read, with versioned publication over a
FastAPI API, Rasterio for raster work and SQLite for metadata.

## Run with Docker Compose

```bash
docker compose up --build -d
# Swagger UI:   http://localhost:8000/docs
# Health:       http://localhost:8000/healthz
docker compose logs -f api
```

Published images and the SQLite database live in the named volume
`rainmosaic-data`, so versions and the current pointer survive
`docker compose restart`, `down` and container upgrades (`down -v` deletes
them). Change the host port with `RAINMOSAIC_PUBLISHED_PORT=9000 docker compose up -d`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/v1/mosaics` | Submit two blocks; publishes a new version |
| GET | `/api/v1/mosaics` | List versions, newest first (`day`, `limit`, `offset`) |
| GET | `/api/v1/mosaics/current` | Current version metadata |
| GET | `/api/v1/mosaics/{version}` | Metadata for a fixed version |
| GET | `/api/v1/mosaics/{version}/download` | GeoTIFF for a fixed version |
| GET | `/api/v1/download/current` | GeoTIFF for the current version |
| POST | `/api/v1/analyses` | Interpret consecutive heavy-rain days at a station |
| GET | `/api/v1/analyses` | List stored reports, newest first |
| GET | `/api/v1/analyses/current` | Current conclusion for one parameter set |
| GET | `/api/v1/analyses/{report_id}` | One stored report by id |
| GET | `/healthz` | Liveness, database and current-version check |

`/docs` supports submitting (multipart form), browsing and downloading.

### Submitting

`multipart/form-data` fields:

| Field | Meaning |
| --- | --- |
| `block_a`, `block_b` | The two GeoTIFF files |
| `block_a_no`, `block_b_no` | Distinct block numbers |
| `block_a_captured_at`, `block_b_captured_at` | UTC acquisition instants, ISO-8601 with offset |
| `expected_version` | Version the submitter last saw; `0` if none is published |
| `day` | Optional `YYYY-MM-DD`; taken from the earlier capture instant if omitted |

Raster contract, enforced per block:

* GeoTIFF, exactly one band, `int16`, NoData declared as `-32768`;
* at most 64×64 pixels;
* north-up grid (no rotation/shear);
* both blocks: same CRS (compared semantically, so an equivalent WKT/PROJ
  spelling is fine), same pixel size, origins offset by whole pixels.

```bash
curl -X POST http://localhost:8000/api/v1/mosaics \
  -F day=2026-01-01 \
  -F expected_version=0 \
  -F block_a_no=101 -F block_a_captured_at=2026-01-01T01:00:00Z \
  -F block_b_no=102 -F block_b_captured_at=2026-01-01T02:00:00Z \
  -F block_a=@block_a.tif -F block_b=@block_b.tif
```

Response `201` carries `version`, `day`, `image` (size, CRS, transform, bounds,
resolution, dtype, NoData, SHA-256, byte size), both block descriptors and a
`download_url`.

### Mosaic rule

The output covers the union bounding rectangle of the two blocks. Each
overlapping pixel takes the value of the block acquired later (UTC); where that
block is NoData the other block's value is used; equal instants are broken by
the larger block number. Pixels covered by neither block stay `-32768`. Output
is a single-band `int16` GeoTIFF with the block CRS/resolution and NoData
declared as `-32768`.

### Versioning and concurrency

`expected_version` makes publication a compare-and-swap. Submissions that
started from the same expected version result in exactly one new version; the
others get `409 version_conflict` with the winning version embedded. Raster
validation errors (`422`), missing/oversized uploads (`413`) and failed
generation (`500 publish_failed`) never create a version and never disturb the
current one.

A new version is rendered to a temporary file, verified by re-opening it,
fsynced and atomically renamed, and only then inserted into SQLite and pointed
to as current, all inside one `BEGIN IMMEDIATE` transaction. The current pointer
therefore never references a partial image; a crash simply leaves the previous
version current. Readers use WAL snapshots and keep downloading the old version
while a publication is in flight.

Every version keeps its own immutable file, so `GET /api/v1/mosaics/{version}/download`
always returns exactly the bytes registered under that version (with a stable
SHA-256 `ETag`), including after restarts.

### Errors

Errors use `{"detail": {...}}` with `code` and `message`; raster errors also
carry `slot` (`a`/`b`) and, where applicable, `field` (`crs`, `resolution`,
`grid_alignment`, `dimensions`, `block_no`).

## Station interpretation reports

`POST /api/v1/analyses` answers "was there continuous heavy rain at this
station?" over a bounded window of published daily mosaics.

Request body (JSON):

| Field | Meaning |
| --- | --- |
| `longitude`, `latitude` | Station position in WGS 84 (`EPSG:4326`), decimal degrees |
| `start_day`, `end_day` | Inclusive UTC day range, `YYYY-MM-DD`; at most 14 days |
| `threshold_mm` | Daily rainfall threshold in millimetres (inclusive) |

```bash
curl -X POST http://localhost:8000/api/v1/analyses \
  -H 'content-type: application/json' \
  -d '{"longitude":120.505,"latitude":30.215,
       "start_day":"2026-01-01","end_day":"2026-01-07","threshold_mm":50}'
```

### How a report is computed

1. **One snapshot selects every version.** In a single SQLite WAL read
   transaction the service records the highest published version number and,
   for every requested day, the newest published version of that day. That
   version set is fixed for the whole report; publications that land while the
   report is being computed cannot add a version to some days only.
2. **Per-day pixel lookup.** For each day the station's WGS 84 coordinates are
   transformed with pyproj into the selected version's own CRS (so daily
   mosaics may legitimately use different CRSs), the containing pixel is found
   with the raster's affine transform, and that single pixel is read.
3. **Breaks and events.** A day is *usable* only when a version exists, the
   station falls inside the raster, and the pixel is not NoData. A missing
   version, an out-of-bounds station and a NoData pixel each break a continuous
   run. A run of at least two consecutive usable days at or above
   `threshold_mm` forms an event.

The `201` response contains `events` (each with `start_day`, `end_day`,
`length_days`, `total_mm` and `daily` entries carrying the day, value and
version), the full per-day `days` evidence with the lookup row/column and CRS,
and `basis` with the selection timestamp and the version chosen for every day.

### Reports, staleness and late results

Every report is stored with the exact version set it used and the pixels it
read, so it stays reproducible and queryable after newer images are published.
`stale` is `true` when the newest published version of any requested day is no
longer the version the report used — for example after a previously missing or
older date is backfilled; `stale_days` lists those days. The old report keeps
its original evidence and remains available at `GET /api/v1/analyses/{id}`.

For each distinct parameter set (station, day range, threshold) exactly one
report is the *current conclusion*, exposed by `GET /api/v1/analyses/current`.
A report computed from a snapshot older than the one recorded for the
conclusion is still stored and queryable but is marked `"is_current": false`
and never replaces the newer conclusion, so a slow computation cannot
overwrite a faster, newer one.

Reports are written only after every selected pixel has been read: a request
that fails midway stores nothing and leaves the current conclusion untouched.

```bash
# All reports for one parameter set (all five filters required together)
curl -G http://localhost:8000/api/v1/analyses \
  --data-urlencode longitude=120.505 --data-urlencode latitude=30.215 \
  --data-urlencode start_day=2026-01-01 --data-urlencode end_day=2026-01-07 \
  --data-urlencode threshold_mm=50
```

Publishing and downloading mosaics continues to work while reports are being
computed and stored; the new tables live in the same SQLite database and the
same `rainmosaic-data` volume.

## Design notes

* SQLite runs in WAL mode with `synchronous=FULL`; state lives in
  `$RAINMOSAIC_DATA_DIR/rainmosaic.sqlite3`, images in `versions/`, scratch files
  in `tmp/`.
* Startup validates the schema, repairs a dangling current pointer and removes
  old orphan files left by interrupted publications.
* Generation and file I/O run in a worker thread so other requests, including
  downloads, keep flowing.
* Only one API worker is used by default: SQLite plus a local disk volume make
  horizontal scaling a deployment concern rather than a code feature.

## Local development and tests

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
RAINMOSAIC_DATA_DIR=./data python main.py          # http://127.0.0.1:8000/docs
python -m pytest -q
```

The test suite covers the mosaic rules (overlap priority, NoData fallback, tie
break, gaps, alignment/CRS/size validation), publishing, version pinning,
conflicts, concurrent submissions, failed generation, downloads during
publication, restart persistence, listing/pagination and error handling. For
interpretation it covers daily newest-version selection, per-CRS pixel lookup,
inclusive thresholds, run breaks from missing days/NoData/out-of-bounds, the
14-day bound, snapshot isolation from concurrent publications, staleness after
a backfill, late older-basis reports not replacing newer conclusions, failure
without a stored report, and restart persistence.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAINMOSAIC_DATA_DIR` | `/data` in the image, `./data` locally | Writable persistent directory |
| `RAINMOSAIC_MAX_UPLOAD_BYTES` | `2097152` | Per-file upload cap |
| `RAINMOSAIC_PORT` | `8000` | Listen port |
| `RAINMOSAIC_LOG_LEVEL` | `INFO` | Log level |
