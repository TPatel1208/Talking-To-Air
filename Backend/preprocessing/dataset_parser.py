"""
preprocessing/dataset_parser.py
===============================
Pure parsing logic — open and normalise NASA granule files.

Supports: TEMPO, OMI, TROPOMI, MODIS, and other NetCDF4/HDF5 layouts.

This module has no network calls, no I/O beyond reading the input file,
and no caching logic. Just parse and return an xr.Dataset.

Example
-------
    from preprocessing.dataset_parser import DatasetParser

    parser = DatasetParser()
    ds = parser.parse_granule("OMI_NO2.nc", granule_times={"OMI_NO2": pd.Timestamp(...)})
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Standard NASA time epochs
_EPOCH_DAYS = datetime(1972, 1, 1)      # OMI NO2
_EPOCH_SECS = datetime(1980, 1, 6)      # TEMPO (GPS epoch)
_EPOCH_1990 = datetime(1990, 1, 1)      # some MODIS products


class DatasetParser:
    """Parse granule files and normalise time coordinates."""

    def parse_granule(
        self,
        filename: str,
        granule_times: Optional[Dict[str, pd.Timestamp]] = None,
    ) -> xr.Dataset:
        """
        Open a NASA NetCDF4 / HDF-EOS5 file and normalise the time coordinate.

        Supports
        --------
        - TEMPO: coords at root, data in 'product' group, time in seconds since 1980-01-06
        - OMI NO2: flat, time in days since 1972-01-01
        - OMI HCHO: multiple groups (key_science_data, qa_statistics)
        - TROPOMI: flat, no time dim → synthesised from CMR or filename
        - MODIS AOD: HDF-EOS5 grid format

        Parameters
        ----------
        filename : str
            Path to the granule file.
        granule_times : dict, optional
            Mapping of granule filename stem → pd.Timestamp (from CMR).
            Used when time is not in the file.

        Returns
        -------
        xr.Dataset
            With time coordinate normalised to np.datetime64[ns].

        Raises
        ------
        RuntimeError
            If file cannot be parsed or contains no data.
        """
        if granule_times is None:
            granule_times = {}

        logger.info("Parsing granule: %s", Path(filename).name)

        try:
            root = xr.open_dataset(filename, engine="netcdf4", decode_times=False)
        except Exception as exc:
            logger.error("Failed to open %s: %s", filename, exc)
            raise

        # Inspect groups
        try:
            with nc.Dataset(filename) as f:
                groups = list(f.groups.keys())
        except Exception:
            groups = []

        logger.debug("File groups: %s", groups)

        # ─────────────────────────────────────────────────────────────────
        # TEMPO: coords at root, data in 'product' group
        # ─────────────────────────────────────────────────────────────────
        if "product" in groups:
            try:
                product = xr.open_dataset(
                    filename, group="product", engine="netcdf4", decode_times=False
                )
                coords = {}
                for coord in ("latitude", "longitude"):
                    if coord in root:
                        coords[coord] = root[coord]
                if "time" in root:
                    coords["time"] = self._decode_time(
                        root["time"], _EPOCH_SECS, unit="s"
                    )
                logger.info("Opened TEMPO-style file with 'product' group")
                return product.assign_coords(**coords)
            except Exception as exc:
                logger.warning("Failed to open 'product' group: %s", exc)

        # ─────────────────────────────────────────────────────────────────
        # MODIS AOD: HDF-EOS5 grid
        # ─────────────────────────────────────────────────────────────────
        if "HDFEOS" in groups and "product" not in groups:
            try:
                import h5py

                ds = self._parse_hdf_eos5(filename, root, granule_times)
                logger.info("Opened HDF-EOS5 grid file")
                return ds
            except Exception as exc:
                logger.warning("Failed to parse HDF-EOS5: %s", exc)

        # ─────────────────────────────────────────────────────────────────
        # OMI HCHO: multiple named groups
        # ─────────────────────────────────────────────────────────────────
        KNOWN_DATA_GROUPS = {"key_science_data", "qa_statistics", "support_data", "geolocation"}
        if any(g in KNOWN_DATA_GROUPS for g in groups) and "product" not in groups:
            try:
                ds = self._parse_grouped_dataset(filename, root, granule_times)
                logger.info("Opened grouped dataset (OMI HCHO style)")
                return ds
            except Exception as exc:
                logger.warning("Failed to parse grouped dataset: %s", exc)

        # ─────────────────────────────────────────────────────────────────
        # OMI NO2 / MODIS: flat file with numeric time axis
        # ─────────────────────────────────────────────────────────────────
        time_key = None
        for key in ("Time", "time"):
            if key in root:
                time_key = key
                break

        if time_key:
            try:
                ds = self._parse_with_time_axis(root, time_key)
                logger.info("Opened flat file with time axis")
                return ds
            except Exception as exc:
                logger.warning("Failed to parse time axis: %s", exc)

        # ─────────────────────────────────────────────────────────────────
        # TROPOMI: no time dim → synthesise from CMR or filename
        # ─────────────────────────────────────────────────────────────────
        stem = Path(filename).stem
        synth_time = granule_times.get(stem) or self._extract_time_from_filename(filename)
        if synth_time is None:
            logger.warning("Could not determine time for %s; using NaT", filename)
            synth_time = pd.NaT

        synth_time_np = (
            np.datetime64(synth_time.to_datetime64(), "ns")
            if not pd.isna(synth_time)
            else np.datetime64("NaT", "ns")
        )

        logger.info("Opened flat file, synthesised time: %s", synth_time)
        return root.expand_dims(dim={"time": [synth_time_np]})

    def _parse_hdf_eos5(
        self,
        filename: str,
        root: xr.Dataset,
        granule_times: Dict[str, pd.Timestamp],
    ) -> xr.Dataset:
        """Parse MODIS AOD HDF-EOS5 grid structure."""
        import h5py

        data_vars = {}

        with h5py.File(filename, "r") as f:
            grids = f["HDFEOS"]["GRIDS"]
            grid_name = list(grids.keys())[0]
            data_fields = grids[grid_name]["Data Fields"]
            grid_group = grids[grid_name]

            # Extract grid info
            grid_span = np.asarray(
                grid_group.attrs.get("GridSpan", b"(-180,180,-90,90)")
            ).flat[0]
            n_lon = int(
                np.asarray(
                    grid_group.attrs.get("NumberOfLongitudesInGrid", 1440)
                ).flat[0]
            )
            n_lat = int(
                np.asarray(
                    grid_group.attrs.get("NumberOfLatitudesInGrid", 720)
                ).flat[0]
            )

            span_str = (
                grid_span.decode() if isinstance(grid_span, bytes) else str(grid_span)
            )
            lon_min, lon_max, lat_min, lat_max = [
                float(x) for x in span_str.strip("()").split(",")
            ]

            lons = (
                np.linspace(lon_min, lon_max, n_lon, endpoint=False)
                + (lon_max - lon_min) / (2 * n_lon)
            )
            lats = (
                np.linspace(lat_min, lat_max, n_lat, endpoint=False)
                + (lat_max - lat_min) / (2 * n_lat)
            )

            # Read variables
            for var_name in data_fields.keys():
                data = data_fields[var_name][()]
                fill = data_fields[var_name].attrs.get("_FillValue", None)

                arr = data.astype(np.float32)
                if fill is not None:
                    try:
                        fill_val = float(np.asarray(fill).flat[0])
                        arr = np.where(
                            np.isclose(arr, fill_val, rtol=0, atol=abs(fill_val) * 1e-3),
                            np.nan,
                            arr,
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

        # Synthesise time
        stem = Path(filename).stem
        synth_time = granule_times.get(stem) or self._extract_time_from_filename(filename)
        if synth_time is None:
            synth_time = pd.NaT

        synth_time_np = (
            np.datetime64(synth_time.to_datetime64(), "ns")
            if not pd.isna(synth_time)
            else np.datetime64("NaT", "ns")
        )

        return ds.expand_dims(dim={"time": [synth_time_np]})

    def _parse_grouped_dataset(
        self,
        filename: str,
        root: xr.Dataset,
        granule_times: Dict[str, pd.Timestamp],
    ) -> xr.Dataset:
        """Parse OMI HCHO style: multiple groups merged into one dataset."""
        KNOWN_DATA_GROUPS = {"key_science_data", "qa_statistics", "support_data", "geolocation"}

        merged_vars = {}

        try:
            with nc.Dataset(filename) as f:
                groups = list(f.groups.keys())
        except Exception:
            groups = []

        for g in groups:
            if g in KNOWN_DATA_GROUPS:
                try:
                    grp_ds = xr.open_dataset(
                        filename, group=g, engine="netcdf4", decode_times=False
                    )
                    for var in grp_ds.data_vars:
                        merged_vars[var] = grp_ds[var]
                except Exception as exc:
                    logger.warning("Could not open group '%s': %s", g, exc)

        if not merged_vars:
            raise RuntimeError("No variables found in any known group")

        ds = xr.Dataset(merged_vars)
        coords = {
            coord: root[coord]
            for coord in ("latitude", "longitude")
            if coord in root
        }
        if coords:
            ds = ds.assign_coords(**coords)

        # Synthesise time
        stem = Path(filename).stem
        synth_time = granule_times.get(stem) or self._extract_time_from_filename(filename)
        if synth_time is None:
            synth_time = pd.NaT

        synth_time_np = (
            np.datetime64(synth_time.to_datetime64(), "ns")
            if not pd.isna(synth_time)
            else np.datetime64("NaT", "ns")
        )

        return ds.expand_dims(dim={"time": [synth_time_np]})

    def _parse_with_time_axis(
        self, root: xr.Dataset, time_key: str
    ) -> xr.Dataset:
        """Parse flat files with numeric time axis (OMI NO2, MODIS)."""
        units = root[time_key].attrs.get("units", "")

        if "1972" in units:
            decoded_time = self._decode_time(root[time_key], _EPOCH_DAYS, unit="D")
        elif "1990" in units:
            decoded_time = self._decode_time(root[time_key], _EPOCH_1990, unit="D")
        else:
            try:
                root = xr.decode_cf(root)
            except Exception:
                pass
            decoded_time = None

        if decoded_time is not None:
            root = root.rename({time_key: "time"}) if time_key != "time" else root
            if "Time" in root.dims:
                root = root.squeeze("Time", drop=True)
            root = root.assign_coords(time=("time", decoded_time.values))

        # Drop unnecessary coordinate bounds
        drop_vars = [
            v
            for v in ("LatitudeBounds", "LongitudeBounds", "TimeBounds", "BoundsIndex", "crs")
            if v in root
        ]
        if drop_vars:
            root = root.drop_vars(drop_vars, errors="ignore")

        return root

    @staticmethod
    def _decode_time(
        time_var: xr.DataArray, epoch: datetime, unit: str
    ) -> xr.DataArray:
        """Convert numeric time to datetime64[ns]."""
        values = time_var.values.astype("float64")
        deltas = pd.to_timedelta(values, unit=unit)
        timestamps = pd.Timestamp(epoch) + deltas

        if not hasattr(timestamps, "__len__"):
            timestamps = [timestamps]

        result = [
            np.datetime64(t.to_datetime64(), "ns")
            if not pd.isna(t)
            else np.datetime64("NaT", "ns")
            for t in timestamps
        ]

        return xr.DataArray(result, dims=time_var.dims, attrs=time_var.attrs)

    @staticmethod
    def _extract_time_from_filename(filename: str) -> Optional[pd.Timestamp]:
        """
        Guess the granule time from the filename.

        Looks for patterns like YYYYMMDD or YYYYMMDDHHMM in the filename.
        """
        import re

        stem = Path(filename).stem
        patterns = [
            (r"(\d{8})T(\d{6})", "%Y%m%dT%H%M%S"),  # YYYYMMDDTHHMMSS
            (r"(\d{8})_(\d{6})", "%Y%m%d_%H%M%S"),  # YYYYMMDD_HHMMSS
            (r"(\d{8})", "%Y%m%d"),                 # YYYYMMDD
        ]

        for pattern, fmt in patterns:
            match = re.search(pattern, stem)
            if match:
                try:
                    date_str = match.group(0).replace("_", "").replace("T", "")
                    return pd.Timestamp(date_str, format=fmt.replace("_", "").replace("T", ""))
                except (ValueError, IndexError):
                    pass

        return None