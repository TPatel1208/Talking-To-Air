import os
import earthaccess
import logging
from datetime import datetime, timezone
import time
from harmony import BBox, Client, Collection, Request, Environment
from typing import Tuple, List, Optional
from pathlib import Path
import xarray as xr
import hashlib
import zarr
import concurrent.futures
import netCDF4 as nc
import re
import pandas as pd
import numpy as np

DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

_EPOCH_DAYS = datetime(1972, 1, 1, tzinfo=timezone.utc)   # OMI
_EPOCH_SECS = datetime(1980, 1, 6, tzinfo=timezone.utc)   # TEMPO
_EPOCH_1990 = datetime(1990, 1, 1, tzinfo=timezone.utc)   # MODIS AOD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PostGIS metadata index — imported lazily so the module still loads when
# psycopg / the DB is unavailable (e.g. unit-test runs without a live DB).
# ---------------------------------------------------------------------------
try:
    from preprocessing import cache_index as _ci
    _CI_AVAILABLE = True
except ImportError:
    try:
        import cache_index as _ci   # fallback when run as __main__
        _CI_AVAILABLE = True
    except ImportError:
        _ci = None
        _CI_AVAILABLE = False


def _get_db_conn():
    """
    Return an open psycopg (v3) connection to the metadata DB, or None if the
    cache-index module is unavailable or the DB is unreachable.
    Calls ensure_schema() on every connection so the table always exists.
    """
    if not _CI_AVAILABLE:
        return None
    try:
        conn = _ci.get_connection()
        _ci.ensure_schema(conn)
        return conn
    except Exception as exc:
        logger.warning("data_loader: DB metadata index unavailable — %s", exc)
        return None


class DataLoader:
    def __init__(self):
        try:
            self.auth = earthaccess.login(strategy="environment")
            if not self.auth:
                raise RuntimeError("Authentication failed. Please check your credentials.")
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            raise

        self.harmony_client = Client(
            env=Environment.PROD,
            auth=(os.getenv("EDL_USERNAME"), os.getenv("EDL_PASSWORD")),
        )

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def download_dataset_harmony(
        self,
        collection_id: str,
        temporal: Tuple[str, str],
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        variables: Optional[List[str]] = None,
        max_results: int = 10,
        output_format: str = "application/x-netcdf4",
        cache_path: str = "cache.zarr",
    ) -> xr.Dataset:
        """
        Return a dataset for the requested collection / time range / bbox.

        Lookup order
        ------------
        1. PostGIS metadata index — exact group_key match (B-tree); fastest path,
           avoids opening the Zarr store at all.
        2. Zarr store key check   — safety fallback when the DB is unreachable.
        3. NASA Harmony fetch     — only on a true cache miss; result is written to
                                    the Zarr store and a metadata row is inserted.

        Args:
            collection_id : NASA CMR concept-id.
            temporal      : (start_iso, end_iso) in 'YYYY-MM-DDTHH:MM:SSZ' format.
            bounding_box  : (min_lon, min_lat, max_lon, max_lat), or None for global.
            variables     : Optional list of variable paths for Harmony subsetting.
            max_results   : Maximum number of granules to request from Harmony.
            output_format : MIME type passed to Harmony (default: netCDF4).
            cache_path    : Path to the Zarr store on disk.
        """
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        try:
            start_dt = datetime.strptime(temporal[0], fmt)
            end_dt   = datetime.strptime(temporal[1], fmt)
        except Exception as e:
            logger.error(f"Invalid date format: {e}")
            raise ValueError(
                "Temporal parameters must be in 'YYYY-MM-DDTHH:MM:SSZ' (ISO 8601) format"
            )

        group_key = self.make_group_key(
            collection_id, temporal[0], temporal[1],
            bounding_box if bounding_box else (),
        )
        logger.info(f"Cache group key: {group_key}")

        # ------------------------------------------------------------------
        # 1. PostGIS metadata index lookup
        # ------------------------------------------------------------------
        conn = _get_db_conn()
        if conn is not None:
            try:
                meta_row = _ci.lookup(conn, collection_id, group_key, temporal, bounding_box)
                if meta_row is not None:
                    stored_path      = meta_row["cache_path"]
                    cached_group_key = meta_row["group_key"]
                    is_superset      = meta_row.get("superset_hit", False)
                    logger.info(
                        "Metadata index %s — loading from Zarr at %s  group=%s",
                        "superset hit" if is_superset else "exact hit",
                        stored_path, cached_group_key,
                    )
                    try:
                        combined = xr.open_zarr(
                            stored_path, group=cached_group_key, consolidated=False
                        )

                        # --------------------------------------------------
                        # Spatial superset: trim the cached dataset down to
                        # the originally requested bbox.
                        # Only safe when lat/lon are 1-D and monotonic.
                        # --------------------------------------------------
                        if is_superset and bounding_box:
                            combined = self._subset_bbox(combined, bounding_box)

                        conn.close()
                        return combined
                    except Exception as zarr_exc:
                        logger.warning(
                            "Metadata index pointed to stale Zarr entry (%s) — "
                            "falling through to re-fetch. Error: %s",
                            stored_path, zarr_exc,
                        )
            except Exception as exc:
                logger.warning("Metadata index lookup error — %s", exc)
            # conn stays open; we'll use it for the insert after a Harmony fetch
        else:
            # ------------------------------------------------------------------
            # 2. Zarr-only fallback (DB unavailable)
            # ------------------------------------------------------------------
            if self.is_cached(cache_path, group_key):
                logger.info(
                    "Zarr cache hit (no DB) — loading from %s  group=%s",
                    cache_path, group_key,
                )
                return xr.open_zarr(cache_path, group=group_key, consolidated=False)

        # ------------------------------------------------------------------
        # 3. NASA Harmony fetch (true cache miss)
        # ------------------------------------------------------------------
        logger.info("Cache miss — fetching from NASA Harmony")
        collection = Collection(id=collection_id)

        request_params = {
            "collection": collection,
            "temporal":   {"start": start_dt, "stop": end_dt},
            "max_results": max_results,
            "format":      output_format,
        }
        if variables:
            request_params["variables"] = variables
        if bounding_box:
            request_params["spatial"] = BBox(*bounding_box)

        request = Request(**request_params)
        if not request.is_valid():
            logger.error("Invalid Harmony request parameters")
            if conn:
                conn.close()
            raise ValueError("Harmony request parameters are invalid")

        logger.info("Submitting Harmony request")
        job_id = self.harmony_client.submit(request)
        fetch_start = time.time()
        self.harmony_client.wait_for_processing(job_id, show_progress=True)
        granule_times = self._get_granule_times(collection_id, temporal, bounding_box)

        datasets = []
        futures = self.harmony_client.download_all(job_id, directory=DOWNLOAD_DIR, overwrite=True)
        for future in concurrent.futures.as_completed(futures):
            filename = future.result()
            logger.info(f"Downloaded: {filename}")
            datasets.append(self._open_dataset(filename, granule_times=granule_times))

        if not datasets:
            logger.error("No datasets were downloaded")
            if conn:
                conn.close()
            raise RuntimeError("Failed to download any datasets from Harmony")

        if len(datasets) == 1:
            combined = datasets[0]
        else:
            valid = [ds for ds in datasets if "time" in ds.dims or "time" in ds.coords]
            dropped = len(datasets) - len(valid)
            if dropped:
                logger.warning(f"{dropped} granule(s) had no time coordinate and were dropped")
            if not valid:
                if conn:
                    conn.close()
                raise RuntimeError("No granules with a time coordinate — cannot concatenate")
            combined = xr.concat(valid, dim="time")

        for coord in combined.coords:
            combined[coord].attrs.pop("units", None)
            combined[coord].attrs.pop("calendar", None)

        # Write Zarr (unchanged behaviour)
        combined.to_zarr(cache_path, group=group_key, mode="w", consolidated=True)
        logger.info(f"Cached to: {cache_path}  group={group_key}")
        logger.info(f"Harmony fetch + load: {time.time() - fetch_start:.2f}s")

        # ------------------------------------------------------------------
        # 4. Insert metadata row into PostGIS index
        # ------------------------------------------------------------------
        if conn is not None:
            try:
                _ci.insert(
                    conn,
                    collection_id=collection_id,
                    group_key=group_key,
                    cache_path=cache_path,
                    temporal=temporal,
                    bounding_box=bounding_box,
                )
            except Exception as exc:
                # Non-fatal: Zarr data is already safely on disk
                logger.warning("Metadata index insert failed (non-fatal) — %s", exc)
            finally:
                conn.close()

        return combined

    # -------------------------------------------------------------------------
    # Cache helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def make_group_key(
        collection_id: str,
        start_str: str,
        end_str: str,
        bbox: tuple,
    ) -> str:
        """Stable MD5-based key that addresses a group inside the Zarr store."""
        bbox_str = "_".join(map(str, bbox))
        raw = f"{collection_id}_{start_str}_{end_str}_{bbox_str}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def is_cached(cache_path: str, group_key: str) -> bool:
        """Return True if group_key already exists in the Zarr store."""
        try:
            store = zarr.open(cache_path, mode="r")
            return group_key in store
        except Exception:
            return False
    @staticmethod
    def _subset_bbox(
        ds: xr.Dataset,
        bounding_box: Tuple[float, float, float, float],
    ) -> xr.Dataset:
        """
        Trim *ds* to *bounding_box* using label-based selection.

        Only operates on dimension coordinates named 'latitude'/'longitude'
        (or 'lat'/'lon') that are 1-D and strictly monotonic.  If those
        conditions aren't met the dataset is returned unmodified so callers
        always get something usable.

        Parameters
        ----------
        ds           : xarray Dataset loaded from a superset cache entry
        bounding_box : (min_lon, min_lat, max_lon, max_lat)
        """
        min_lon, min_lat, max_lon, max_lat = bounding_box

        # Normalise coordinate names to ('latitude', 'longitude')
        rename_map = {}
        for canonical, aliases in (
            ("latitude",  ["lat"]),
            ("longitude", ["lon"]),
        ):
            if canonical not in ds.dims:
                for alias in aliases:
                    if alias in ds.dims:
                        rename_map[alias] = canonical
                        break
        if rename_map:
            ds = ds.rename(rename_map)

        sel_kwargs: dict = {}
        for dim, lo, hi in (
            ("latitude",  min_lat, max_lat),
            ("longitude", min_lon, max_lon),
        ):
            if dim not in ds.dims:
                logger.debug("_subset_bbox: dim '%s' not found — skipping slice", dim)
                continue
            coord = ds[dim]
            if coord.ndim != 1:
                logger.warning(
                    "_subset_bbox: '%s' is %d-D — cannot slice, returning full dataset",
                    dim, coord.ndim,
                )
                return ds
            vals = coord.values
            diffs = np.diff(vals)
            if not (np.all(diffs > 0) or np.all(diffs < 0)):
                logger.warning(
                    "_subset_bbox: '%s' is not monotonic — cannot slice, returning full dataset",
                    dim,
                )
                return ds
            # slice handles both ascending and descending axes correctly
            sel_kwargs[dim] = slice(lo, hi) if vals[0] <= vals[-1] else slice(hi, lo)

        if not sel_kwargs:
            return ds

        try:
            return ds.sel(**sel_kwargs)
        except Exception as exc:
            logger.warning("_subset_bbox: sel() failed (%s) — returning full dataset", exc)
            return ds

    # -------------------------------------------------------------------------
    # NetCDF4 / HDF-EOS5 parsing helpers
    # -------------------------------------------------------------------------

    def _open_dataset(self, filename: str, granule_times: Optional[dict] = None) -> xr.Dataset:
        """
        Open a NASA NetCDF4 / HDF-EOS5 file and normalise the time coordinate.

        Supported file layouts
        ----------------------
        TEMPO       — coords at root (seconds since 1980-01-06), data in 'product' group
        OMI NO2     — flat file, time = days since 1972-01-01
        OMI HCHO    — grouped (key_science_data / qa_statistics), no time dim
        TROPOMI     — flat, no time dim → synthesised from CMR metadata or filename
        MODIS AOD   — HDF-EOS5 grid format
        """
        if granule_times is None:
            granule_times = {}

        try:
            root = xr.open_dataset(filename, engine="netcdf4", decode_times=False)
        except Exception as e:
            logger.error(f"Failed to open {filename}: {e}")
            raise

        try:
            with nc.Dataset(filename) as f:
                groups = list(f.groups.keys())
        except Exception:
            groups = []

        logger.debug(f"File groups: {groups}")

        # --- TEMPO: coords at root, data in 'product' group ---
        if "product" in groups:
            try:
                product = xr.open_dataset(
                    filename, group="product", engine="netcdf4", decode_times=False
                )
                coords = {}
                for coord in ["latitude", "longitude"]:
                    if coord in root:
                        coords[coord] = root[coord]
                if "time" in root:
                    coords["time"] = self._decode_time(root["time"], _EPOCH_SECS, unit="s")
                return product.assign_coords(**coords)
            except Exception as e:
                logger.warning(f"Failed to open 'product' group, falling back: {e}")

        # --- MODIS AOD: HDF-EOS5 grid ---
        if "HDFEOS" in groups and "product" not in groups:
            try:
                import h5py
                data_vars = {}

                with h5py.File(filename, "r") as f:
                    grids      = f["HDFEOS"]["GRIDS"]
                    grid_name  = list(grids.keys())[0]
                    data_fields = grids[grid_name]["Data Fields"]
                    grid_group  = grids[grid_name]

                    grid_span = np.asarray(
                        grid_group.attrs.get("GridSpan", b"(-180,180,-90,90)")
                    ).flat[0]
                    n_lon = int(np.asarray(
                        grid_group.attrs.get("NumberOfLongitudesInGrid", 1440)
                    ).flat[0])
                    n_lat = int(np.asarray(
                        grid_group.attrs.get("NumberOfLatitudesInGrid", 720)
                    ).flat[0])

                    span_str = grid_span.decode() if isinstance(grid_span, bytes) else str(grid_span)
                    lon_min, lon_max, lat_min, lat_max = [
                        float(x) for x in span_str.strip("()").split(",")
                    ]
                    lons = np.linspace(lon_min, lon_max, n_lon, endpoint=False) + (lon_max - lon_min) / (2 * n_lon)
                    lats = np.linspace(lat_min, lat_max, n_lat, endpoint=False) + (lat_max - lat_min) / (2 * n_lat)

                    for var_name in data_fields.keys():
                        data = data_fields[var_name][()]
                        fill = data_fields[var_name].attrs.get("_FillValue", None)
                        arr  = data.astype(np.float32)
                        if fill is not None:
                            try:
                                fill_val = float(np.asarray(fill).flat[0])
                                arr = np.where(
                                    np.isclose(arr, fill_val, rtol=0, atol=abs(fill_val) * 1e-3),
                                    np.nan, arr,
                                )
                            except Exception:
                                pass
                        safe_attrs = {}
                        for k, v in data_fields[var_name].attrs.items():
                            try:
                                scalar = np.asarray(v).flat[0]
                                if isinstance(scalar, bytes):
                                    safe_attrs[k] = scalar.decode("utf-8", errors="replace")
                                elif hasattr(scalar, "item"):
                                    safe_attrs[k] = scalar.item()
                                else:
                                    safe_attrs[k] = str(scalar)
                            except Exception:
                                pass
                        data_vars[var_name] = xr.DataArray(
                            arr, dims=["latitude", "longitude"], attrs=safe_attrs
                        )

                if not data_vars:
                    raise RuntimeError("No variables found in HDF-EOS5 file")

                ds = xr.Dataset(data_vars).assign_coords(
                    latitude=("latitude", lats),
                    longitude=("longitude", lons),
                )
                stem       = Path(filename).stem
                synth_time = granule_times.get(stem) or self._extract_time_from_filename(filename)
                if synth_time is None:
                    logger.warning(f"Could not determine time for {filename}; using NaT")
                    synth_time = pd.NaT
                synth_time_np = (
                    np.datetime64(synth_time.to_datetime64(), "ns")
                    if not pd.isna(synth_time)
                    else np.datetime64("NaT", "ns")
                )
                logger.info(f"Opened HDF-EOS5: {grid_name}, vars={list(data_vars.keys())}")
                return ds.expand_dims(dim={"time": [synth_time_np]})

            except Exception as e:
                logger.warning(f"Failed to open HDF-EOS5 file: {e}")

        # --- OMI HCHO: named groups, no 'product' group ---
        KNOWN_DATA_GROUPS = {"key_science_data", "qa_statistics", "support_data", "geolocation"}
        if any(g in KNOWN_DATA_GROUPS for g in groups) and "product" not in groups:
            try:
                merged_vars = {}
                for g in groups:
                    if g in KNOWN_DATA_GROUPS:
                        try:
                            grp_ds = xr.open_dataset(
                                filename, group=g, engine="netcdf4", decode_times=False
                            )
                            for var in grp_ds.data_vars:
                                merged_vars[var] = grp_ds[var]
                        except Exception as e:
                            logger.warning(f"Could not open group '{g}': {e}")

                if not merged_vars:
                    raise RuntimeError("No variables found in any known group")

                ds = xr.Dataset(merged_vars)
                coords = {
                    coord: root[coord]
                    for coord in ["latitude", "longitude"]
                    if coord in root
                }
                if coords:
                    ds = ds.assign_coords(**coords)

                stem       = Path(filename).stem
                synth_time = granule_times.get(stem) or self._extract_time_from_filename(filename)
                if synth_time is None:
                    logger.warning(f"Could not determine time for {filename}; using NaT")
                    synth_time = pd.NaT
                synth_time_np = (
                    np.datetime64(synth_time.to_datetime64(), "ns")
                    if not pd.isna(synth_time)
                    else np.datetime64("NaT", "ns")
                )
                return ds.expand_dims(dim={"time": [synth_time_np]})

            except Exception as e:
                logger.warning(f"Failed to open grouped dataset, falling back: {e}")

        # --- OMI NO2 / MODIS: flat file with numeric time axis ---
        time_key = "Time" if "Time" in root else "time" if "time" in root else None
        if time_key:
            units = root[time_key].attrs.get("units", "")
            if "1972" in units:
                decoded_time = self._decode_time(root[time_key], _EPOCH_DAYS, unit="D")
                root = root.rename({time_key: "time"}) if time_key != "time" else root
                root = root.squeeze("Time", drop=True) if "Time" in root.dims else root
                root = root.assign_coords(time=("time", decoded_time.values))
            elif "1990" in units:
                decoded_time = self._decode_time(root[time_key], _EPOCH_1990, unit="D")
                root = root.rename({time_key: "time"}) if time_key != "time" else root
                root = root.squeeze("Time", drop=True) if "Time" in root.dims else root
                root = root.assign_coords(time=("time", decoded_time.values))
            else:
                try:
                    root = xr.decode_cf(root)
                except Exception:
                    pass
            root = root.drop_vars(
                [v for v in ["LatitudeBounds", "LongitudeBounds", "TimeBounds", "BoundsIndex", "crs"]
                 if v in root],
                errors="ignore",
            )
            return root

        # --- TROPOMI: no time dim → synthesise from CMR metadata or filename ---
        stem       = Path(filename).stem
        synth_time = granule_times.get(stem) or self._extract_time_from_filename(filename)
        if synth_time is None:
            logger.warning(f"Could not determine time for {filename}; using NaT")
            synth_time = pd.NaT
        synth_time_np = (
            np.datetime64(synth_time.to_datetime64(), "ns")
            if not pd.isna(synth_time)
            else np.datetime64("NaT", "ns")
        )
        return root.expand_dims(dim={"time": [synth_time_np]})

    def _get_granule_times(
        self,
        collection_id: str,
        temporal: Tuple[str, str],
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
    ) -> dict:
        """
        Query earthaccess CMR and return a dict mapping
        producer granule filename stem → pd.Timestamp of granule start time.
        """
        try:
            params = {"concept_id": collection_id, "temporal": temporal}
            if bounding_box:
                params["bounding_box"] = bounding_box
            results = earthaccess.search_data(**params)
            lookup  = {}
            for granule in results:
                meta         = granule.get("umm", {})
                identifiers  = meta.get("DataGranule", {}).get("Identifiers", [])
                granule_id   = next(
                    (i["Identifier"] for i in identifiers
                     if i.get("IdentifierType") == "ProducerGranuleId"),
                    None,
                )
                time_str = (
                    meta.get("TemporalExtent", {})
                        .get("RangeDateTime", {})
                        .get("BeginningDateTime")
                )
                if granule_id and time_str:
                    stem = Path(granule_id).stem
                    lookup[stem] = pd.Timestamp(time_str, tz="UTC")
                    logger.debug(f"Granule time: {stem} → {lookup[stem]}")
            logger.info(f"Built time lookup for {len(lookup)} granule(s)")
            return lookup
        except Exception as e:
            logger.warning(
                f"Failed to build granule time lookup, falling back to filename parsing: {e}"
            )
            return {}

    @staticmethod
    def _decode_time(time_var: xr.DataArray, epoch: datetime, unit: str) -> xr.DataArray:
        values     = time_var.values.astype("float64")
        deltas     = pd.to_timedelta(values, unit=unit)
        timestamps = pd.Timestamp(epoch) + deltas
        result     = timestamps if hasattr(timestamps, "__len__") else [timestamps]
        result = [
            np.datetime64(t.to_datetime64(), "ns") if not pd.isna(t)
            else np.datetime64("NaT", "ns")
            for t in (result if hasattr(result, "__iter__") else [result])
        ]
        return xr.DataArray(result, dims=time_var.dims, attrs=time_var.attrs)

    @staticmethod
    def _extract_time_from_filename(filename: str) -> Optional[pd.Timestamp]:
        """Fallback: parse a timestamp from the filename when CMR metadata is unavailable."""
        stem = Path(filename).stem
        for pattern, fmt in [
            (r"(\d{8}T\d{6})",        "%Y%m%dT%H%M%S"),  # TEMPO:   20260210T172301
            (r"(\d{8}_\d{6})",        "%Y%m%d_%H%M%S"),  # generic: 20260210_172301
            (r"(?<!\d)(\d{8})(?!\d)", "%Y%m%d"),          # TROPOMI: 20240810
        ]:
            m = re.search(pattern, stem)
            if m:
                try:
                    return pd.Timestamp(datetime.strptime(m.group(1), fmt), tz="UTC")
                except ValueError:
                    continue
        return None


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    logging.basicConfig(level=logging.DEBUG)

    data_loader = DataLoader()
    ds = data_loader.download_dataset_harmony(
        collection_id="C1266136037-GES_DISC",  # OMI Ozone
        temporal=("2024-07-08T00:00:00Z", "2024-08-08T23:59:59Z"),
        bounding_box=(-106.6458, 25.8371, -93.5078, 36.5005),
        output_format="application/x-netcdf4",
        max_results=1,
        cache_path="cache_test.zarr",
    )

    print("\n=== Dataset structure ===")
    print("Data vars:", list(ds.data_vars))
    print("Coords:   ", list(ds.coords))
    print("Dims:     ", dict(ds.sizes))
    for var in ds.data_vars:
        print(f"  {var}: {ds[var].dims} {ds[var].shape}")


if __name__ == "__main__":
    main()