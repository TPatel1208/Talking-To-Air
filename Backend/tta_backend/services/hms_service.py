"""Bounded NOAA HMS vector ingestion, independent of Earthdata MCP.

Archives stay in memory; members are read explicitly, never extracted to disk.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
from datetime import date, datetime, timezone
from pathlib import PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

import httpx
import shapefile
from pyproj import CRS, Transformer
from shapely import make_valid
from shapely.geometry import mapping, shape
from shapely.ops import transform

# HMS analysts often draw smoke outlines counter-clockwise, which pyshp reads
# as holes; it re-encodes them as outlines correctly but logs a warning per
# polygon, flooding the logs on every request.
shapefile.VERBOSE = False

BASE_URL = "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile"
MAX_ZIP_BYTES = 16 * 1024 * 1024
MAX_EXPANDED_BYTES = 64 * 1024 * 1024
MAX_FEATURES = 20000
MAX_GEOJSON_BYTES = 8 * 1024 * 1024
CAVEAT = (
    "HMS shows satellite-observed smoke, not surface PM2.5 concentration. "
    "Clouds and observation conditions can hide smoke. Daily polygons represent "
    "different observation intervals, not continuous all-day exposure."
)


class HMSError(ValueError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def product_date(value: str) -> date:
    try:
        day = date.fromisoformat(value)
        if day.isoformat() != value or day > datetime.now(timezone.utc).date():
            raise ValueError
        return day
    except (ValueError, TypeError):
        raise HMSError("invalid_date", "Use a non-future date in YYYY-MM-DD format.") from None


def download_url(value: str) -> str:
    day = product_date(value)
    return f"{BASE_URL}/{day:%Y}/{day:%m}/hms_smoke{day:%Y%m%d}.zip"


async def download_archive(value: str, *, transport: httpx.AsyncBaseTransport | None = None) -> bytes:
    """Retry transient failures only; missing archives are never called no-smoke."""
    url = download_url(value)
    async with httpx.AsyncClient(timeout=45, transport=transport, follow_redirects=False) as client:
        for attempt in range(3):
            try:
                async with client.stream("GET", url) as response:
                    if response.status_code == 404:
                        raise HMSError("unavailable", "NOAA has no archive at this date's URL; smoke presence is unknown.")
                    response.raise_for_status()
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        chunks.extend(chunk)
                        if len(chunks) > MAX_ZIP_BYTES:
                            raise HMSError("too_large", "HMS archive exceeds the download limit.")
                    return bytes(chunks)
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code in (429, 500, 502, 503, 504)
                if attempt == 2 or not retryable:
                    raise HMSError("download_failed", "NOAA HMS download failed; try again later.") from exc
                await asyncio.sleep(0.5 * 2**attempt)
    raise HMSError("download_failed", "NOAA HMS download failed.")


def _time(value: str) -> str:
    value = value.strip()
    try:
        parsed = datetime.strptime(value, "%Y%j %H%M")
        if parsed.strftime("%Y%j %H%M") != value:
            raise ValueError
        return parsed.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        raise HMSError("invalid_data", f"Invalid HMS observation time: {value!r}.") from None


def parse_archive(content: bytes, value: str, region: Any = None) -> dict[str, Any]:
    """Return EPSG:4326 GeoJSON, retaining each observation's interval and category."""
    product_date(value)
    if len(content) > MAX_ZIP_BYTES:
        raise HMSError("too_large", "HMS archive exceeds the download limit.")
    try:
        return _parse_archive(content, value, region)
    except HMSError:
        raise
    except Exception as exc:
        raise HMSError("invalid_data", "HMS archive or Shapefile could not be read.") from exc


def _parse_archive(content: bytes, value: str, region: Any) -> dict[str, Any]:
    try:
        archive = ZipFile(io.BytesIO(content))
    except BadZipFile:
        raise HMSError("invalid_data", "NOAA response is not a ZIP archive.") from None
    with archive:
        entries = archive.infolist()
        if len(entries) > 32 or sum(item.file_size for item in entries) > MAX_EXPANDED_BYTES:
            raise HMSError("too_large", "HMS ZIP exceeds the extraction limits.")
        names = {}
        for entry in entries:
            path = PurePosixPath(entry.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or ":" in entry.filename:
                raise HMSError("invalid_data", "Unsafe archive member path.")
            key = str(path).lower()
            if key in names:
                raise HMSError("invalid_data", "Duplicate archive member.")
            names[key] = entry
        candidates = [name for name in names if name.endswith(".shp")]
        if len(candidates) != 1:
            raise HMSError("invalid_data", "Expected exactly one smoke Shapefile.")
        stem = candidates[0][:-4]
        if any(stem + ext not in names for ext in (".shx", ".dbf", ".prj")):
            raise HMSError("invalid_data", "Shapefile requires matching SHP, SHX, DBF and PRJ members.")
        def member(ext: str) -> bytes:
            return archive.read(names[stem + ext])

        crs = CRS.from_wkt(member(".prj").decode("utf-8-sig"))
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        encoding = member(".cpg").decode("ascii").strip() if stem + ".cpg" in names else "utf-8"
        encoding = {"65001": "utf-8", "1252": "cp1252"}.get(encoding, encoding)
        features = []
        repaired = skipped = 0
        counts = dict.fromkeys(("light", "medium", "heavy", "unknown"), 0)
        with shapefile.Reader(shp=io.BytesIO(member(".shp")), shx=io.BytesIO(member(".shx")),
                              dbf=io.BytesIO(member(".dbf")), encoding=encoding) as reader:
            if reader.shapeType != shapefile.POLYGON:
                raise HMSError("invalid_data", "Expected polygon smoke data.")
            if reader.numRecords > MAX_FEATURES:
                raise HMSError("too_large", "HMS feature count exceeds the limit.")
            fields = {field[0].lower() for field in reader.fields[1:]}
            if not {"satellite", "start", "end", "density"} <= fields:
                raise HMSError("invalid_data", "Missing HMS attribute fields.")
            for index, record in enumerate(reader.iterShapeRecords()):
                attrs = {key.lower(): str(val).strip() for key, val in record.record.as_dict().items()}
                if record.shape.shapeType == shapefile.NULL:
                    skipped += 1
                    continue
                geom = transform(transformer.transform, shape(record.shape.__geo_interface__))
                if not geom.is_valid:
                    geom = make_valid(geom)
                    repaired += 1
                if geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
                    skipped += 1
                    continue
                if not all(math.isfinite(v) for v in geom.bounds):
                    raise HMSError("invalid_data", "Non-finite polygon coordinates.")
                if region is not None:
                    if not geom.intersects(region):
                        continue
                    geom = geom.intersection(region)
                    if geom.is_empty or geom.area == 0:
                        continue
                    if geom.geom_type not in ("Polygon", "MultiPolygon"):
                        skipped += 1
                        continue
                start, end = _time(attrs["start"]), _time(attrs["end"])
                if end < start:
                    raise HMSError("invalid_data", "HMS observation ends before it starts.")
                raw = attrs["density"]
                density = raw.lower()  # NOAA writes Light / Medium / Heavy
                if density not in counts:
                    density = "unknown"
                counts[density] += 1
                features.append({"type": "Feature", "id": index, "geometry": mapping(geom), "properties": {
                    "satellite": attrs["satellite"], "start": start, "end": end,
                    "density": density, "density_raw": raw,
                }})
        collection = {"type": "FeatureCollection", "features": features}
        if len(json.dumps(collection, allow_nan=False).encode()) > MAX_GEOJSON_BYTES:
            raise HMSError("too_large", "Smoke geometry is too large; select a smaller region.")
        return {"geojson": collection, "counts": counts, "feature_count": len(features),
                "status": "ready" if features else ("invalid_geometry" if skipped else "no_observed_smoke"),
                "provenance": {"source": "NOAA HMS", "source_url": download_url(value),
                    "product_date": value, "sha256": hashlib.sha256(content).hexdigest(),
                    "source_crs": crs.to_string(), "output_crs": "EPSG:4326",
                    "parser_version": 1, "repaired_records": repaired, "skipped_records": skipped,
                    "caveat": CAVEAT}}
