from dotenv import load_dotenv
import os
import earthaccess
import logging 
from datetime import datetime
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
from datetime import timezone
import numpy as np

DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

_EPOCH_DAYS  = datetime(1972, 1, 1, tzinfo=timezone.utc)   # OMI
_EPOCH_SECS  = datetime(1980, 1, 6, tzinfo=timezone.utc)   # TEMPO

logger = logging.getLogger(__name__)    

class DataLoader:
    def __init__(self):
        load_dotenv() 
        try:
            self.auth = earthaccess.login(strategy="environment")
            if not self.auth:
                raise RuntimeError("Authentication failed. Please check your credentials.")
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            raise  
        
        self.harmony_client = Client(env=Environment.PROD,  auth=(os.getenv("EDL_USERNAME"), os.getenv("EDL_PASSWORD")))

        

    def download_file(
            self, 
            save_dir: str,
            short_name: str, 
            temporal: Tuple[str, str],
            version: Optional[str] = None, 
            bounding_box: Optional[Tuple[float, float, float, float]] = None) -> List[str]:
        """
        Downloads data files from Earthdata based on specified parameters.

        Args:
            save_dir (str): Directory to save downloaded files.
            short_name (str): Dataset short name (e.g., 'MOD11A1)..
            temporal (Tuple[str, str]): Start and end dates in 'YYYY-MM-DD' format.
            bounding_box (Tuple[float, float, float, float], optional): Geographic bounding box defined as (min_lon, min_lat, max_lon, max_lat).

        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Search for data
            logger.info(f"Searching for {short_name} data from {temporal[0]} to {temporal[1]}")
            params = {
                "short_name": short_name,
                "temporal": temporal,
            }
            if bounding_box:
                params["bounding_box"] = bounding_box
            if version:
                params["version"] = version
            results = earthaccess.search_data(**params)
            
            if not results:
                logger.warning(f"No data found for {short_name} with given parameters")
                return []
            
            logger.info(f"Found {len(results)} granule(s)")
            
            # Download data
            downloaded_files = earthaccess.download(
                results,
                local_path=str(save_path),
            )
            
            if not downloaded_files:
                logger.warning("Download completed but no files were saved")
                return []
                
            logger.info(f"Successfully downloaded {len(downloaded_files)} file(s)")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise RuntimeError(f"Failed to download {short_name} data: {e}")
        


        
    def get_dataset(self, 
        short_name: str, 
        temporal: Tuple[str, str],
        version: Optional[str] = None, 
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        groups: Optional[List[str]] = None) -> xr.Dataset:
        """
        Loads a dataset directly into xarray.

        Returns:
            xr.Dataset: The loaded dataset.
        """
        try:
            # Search for data
            logger.info(f"Searching for {short_name} data from {temporal[0]} to {temporal[1]}")
            search_params = {
                "short_name": short_name,
                "temporal": temporal,}
            if bounding_box:
                search_params["bounding_box"] = bounding_box
            if version:
                search_params["version"] = version

            results = earthaccess.search_data(**search_params)
            
            if not results:
                logger.warning(f"No data found for {short_name} with given parameters")
                raise ValueError(f"No data found for {short_name}") 
            
            logger.info(f"Found {len(results)} granule(s)")
            
            # Open files (not download - they're accessed remotely or cached)
            file_handles = earthaccess.open(results) 
            
            if not file_handles:
                logger.warning("No file handles returned from earthaccess.open")
                raise RuntimeError("Failed to open files") 
            
            # Load into xarray

            if groups:
                root_datasets = [xr.open_dataset(f) for f in file_handles]
                group_datasets = {g: [xr.open_dataset(f, group=g) for f in file_handles] for g in groups}

                if len(file_handles) == 1:
                    # Single file - no concat needed
                    root = root_datasets[0]
                    merged_groups = xr.merge([group_datasets[g][0] for g in groups])
                else:
                    # Multiple files - concat along time
                    root = xr.concat(root_datasets, dim='time')
                    merged_groups = xr.merge([
                        xr.concat(group_datasets[g], dim='time') for g in groups
                    ])

                ds = merged_groups.assign_coords(
                    time=root.time,
                    latitude=root.latitude,
                    longitude=root.longitude
                )
            else:
                if len(file_handles) == 1:
                    ds = xr.open_dataset(file_handles[0])
                else:
                    ds = xr.open_mfdataset(file_handles, combine='nested', concat_dim='time')
                        
            logger.info(f"Successfully loaded dataset with {len(results)} granule(s)")
            return ds
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise RuntimeError(f"Failed to load {short_name} data: {e}")



    def download_dataset_harmony(
            self,
            collection_id: str,
            temporal: Tuple[str, str],
            bounding_box: Optional[Tuple[float, float, float, float]] = None,
            variables: Optional[List[str]] = None,
            max_results: int = 10,
            output_format: str = 'application/x-netcdf4',
            cache_path: str = "cache.zarr"
        ) -> xr.Dataset:
        """
        Download datasetusing Harmony client as a Zarr store and load into xarray.
        
        Args:
            collection_id (str): Harmony collection ID can be found using CMR search.
            temporal (Tuple[str, str]): Start and end dates in ISO 8601 format.
            bounding_box (Tuple[float, float, float, float], optional): Geographic bounding box defined as (min_lon, min_lat, max_lon, max_lat).
            cache_path (str): Path to Zarr cache file.
        """
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        try:
            start_dt = datetime.strptime(temporal[0], fmt)
            end_dt   = datetime.strptime(temporal[1], fmt)
        except Exception as e:
            logger.error(f"Invalid date format: {e}")
            raise ValueError("Temporal parameters must be in 'YYYY-MM-DDTHH:MM:SSZ' (ISO 8601)format")
        group_key = self.make_group_key(collection_id, temporal[0], temporal[1], bounding_box if bounding_box else ())
        logger.info(f"Cache group key: {group_key}")
        if self.is_cached(cache_path,group_key):
            logger.info(f"Cache hit — loading from Zarr at {cache_path} with group key: {group_key}")
            combined = xr.open_zarr(cache_path, group=group_key, consolidated=False)
            return combined
        else:
            logger.info("Cache miss — fetching from Harmony")
            collection = Collection(id=collection_id)

            request_params = {
                "collection": collection,
                "temporal": {
                    'start': start_dt,
                    'stop': end_dt
                },
                "max_results": max_results,
                "format": output_format
            }
            if variables:
                request_params["variables"] = variables
            if bounding_box:
                request_params["spatial"] = BBox(*bounding_box)
            
            request = Request(**request_params)
            if not request.is_valid():
                logger.error("Invalid Harmony request parameters")
                raise ValueError("Harmony request parameters are invalid")
            else:
                logger.info("Submitting Harmony request")
            job_id = self.harmony_client.submit(request)
            #status = self.harmony_client.status(job_id)
            start = time.time()
            self.harmony_client.wait_for_processing(job_id, show_progress=True)
            granule_times = self._get_granule_times(collection_id, temporal, bounding_box)

            datasets = []
            futures = self.harmony_client.download_all(job_id, directory=DOWNLOAD_DIR, overwrite=True)

            for future in concurrent.futures.as_completed(futures):
                filename = future.result()
                logger.info(f"Downloaded: {filename}")
                ds = self._open_dataset(filename, granule_times=granule_times)  # ADD granule_times
                datasets.append(ds)
            
            if not datasets:
                logger.error("No datasets were downloaded")
                raise RuntimeError("Failed to download any datasets from Harmony")
            if len(datasets) == 1:
                combined = datasets[0]
            else:
                valid = [ds for ds in datasets if "time" in ds.dims or "time" in ds.coords]
                dropped = len(datasets) - len(valid)
                if dropped:
                    logger.warning(f"{dropped} granule(s) had no time coordinate and were dropped")
                if not valid:
                    raise RuntimeError("No granules with a time coordinate — cannot concatenate")
                combined = xr.concat(valid, dim="time")
            for coord in combined.coords:
                combined[coord].attrs.pop("units", None)
                combined[coord].attrs.pop("calendar", None)

            combined.to_zarr(cache_path, group=group_key, mode='w', consolidated=True)
            logger.info(f"Cached to: {cache_path} with group key: {group_key}")
            logger.info(f"total time for Harmony download and load: {time.time() - start:.2f} seconds")
            return combined


    @staticmethod
    def make_group_key(collection_id, start_str, end_str, bbox):
        "helper function to make group key for caching for harmony downloads"
        bbox_str = "_".join(map(str, bbox))
        raw = f"{collection_id}_{start_str}_{end_str}_{bbox_str}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
    @staticmethod
    def is_cached(cache_path, group_key):
        "helper function to check if group key exists in zarr cache"
        try:
            store = zarr.open(cache_path, mode="r")
            return group_key in store
        except Exception:
            return False
        

    def _open_dataset(self, filename: str, granule_times: Optional[dict] = None) -> xr.Dataset:
        """
        Open a NASA NetCDF4 file, normalising the time coordinate.

        TEMPO   — coords at root (seconds since 1980-01-06), data in 'product' group
        OMI     — flat, time = days since 1972-01-01
        TROPOMI — flat, no time dimension → synthesised from CMR metadata or filename
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

        logger.debug(f"File groups found: {groups}")

        # --- TEMPO: coords at root, data in 'product' group 
        if "product" in groups:
            try:
                product = xr.open_dataset(filename, group="product", engine="netcdf4", decode_times=False)
                coords = {}
                for coord in ["latitude", "longitude"]:
                    if coord in root:
                        coords[coord] = root[coord]
                if "time" in root:
                    coords["time"] = self._decode_time(root["time"], _EPOCH_SECS, unit="s")
                return product.assign_coords(**coords)
            except Exception as e:
                logger.warning(f"Failed to open 'product' group, falling back: {e}")

        # --- OMI: flat, days since 1972-01-01 ---
        time_key = "Time" if "Time" in root else "time" if "time" in root else None

        if time_key:
            units = root[time_key].attrs.get("units", "")
            if "1972" in units:
                decoded_time = self._decode_time(root[time_key], _EPOCH_DAYS, unit="D")
                root = root.rename({time_key: "time"}) if time_key != "time" else root
                root = root.squeeze("Time", drop=True) if "Time" in root.dims else root
                root = root.assign_coords(time=("time", decoded_time.values))
            else:
                try:
                    root = xr.decode_cf(root)
                except Exception:
                    pass
            root = root.drop_vars(
                [v for v in ['LatitudeBounds', 'LongitudeBounds', 'TimeBounds', 'BoundsIndex', 'crs']
                if v in root],
                errors='ignore'
            )
            return root





        # --- TROPOMI: no time dimension → CMR metadata lookup, then filename fallback ---
        stem = Path(filename).stem
        synth_time = granule_times.get(stem) or self._extract_time_from_filename(filename)
        if synth_time is None:
            logger.warning(f"Could not determine time for {filename}; using NaT")
            synth_time = pd.NaT
        if pd.isna(synth_time):
            synth_time_np = np.datetime64('NaT', 'ns')
        else:
            synth_time_np = np.datetime64(synth_time.to_datetime64(), 'ns')
        return root.expand_dims(dim={"time": [synth_time_np]})


    def _get_granule_times(
        self,
        collection_id: str,
        temporal: Tuple[str, str],
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        ) -> dict:
        """
        Query earthaccess CMR metadata and return a dict mapping
        producer granule filename stem → pd.Timestamp of granule start time.
        """
        try:
            params = {"concept_id": collection_id, "temporal": temporal}
            if bounding_box:
                params["bounding_box"] = bounding_box
            results = earthaccess.search_data(**params)
            lookup = {}
            for granule in results:
                meta = granule.get("umm", {})
                identifiers = meta.get("DataGranule", {}).get("Identifiers", [])
                granule_id = next(
                    (i["Identifier"] for i in identifiers
                    if i.get("IdentifierType") == "ProducerGranuleId"),
                    None
                )
                time_str = (
                    meta.get("TemporalExtent", {})
                        .get("RangeDateTime", {})
                        .get("BeginningDateTime")
                )
                if granule_id and time_str:
                    stem = Path(granule_id).stem
                    lookup[stem] = pd.Timestamp(time_str, tz="UTC")
                    logger.debug(f"Granule time lookup: {stem} → {lookup[stem]}")
            logger.info(f"Built time lookup for {len(lookup)} granule(s)")
            return lookup
        except Exception as e:
            logger.warning(f"Failed to build granule time lookup, will fall back to filename parsing: {e}")
            return {}

    @staticmethod
    def _decode_time(time_var: xr.DataArray, epoch: datetime, unit: str) -> xr.DataArray:
        values = time_var.values.astype("float64")
        deltas = pd.to_timedelta(values, unit=unit)
        timestamps = pd.Timestamp(epoch) + deltas
        result = timestamps if hasattr(timestamps, "__len__") else [timestamps]
        result = [
            np.datetime64(t.to_datetime64(), 'ns') if not pd.isna(t)
            else np.datetime64('NaT', 'ns')
            for t in (result if hasattr(result, '__iter__') else [result])
        ]
        return xr.DataArray(result, dims=time_var.dims, attrs=time_var.attrs)
    @staticmethod
    def _extract_time_from_filename(filename: str):
        """Fallback: parse timestamp from filename when CMR metadata is unavailable."""
        stem = Path(filename).stem
        for pattern, fmt in [
            (r"(\d{8}T\d{6})",           "%Y%m%dT%H%M%S"),  # TEMPO:   20260210T172301Z
            (r"(\d{8}_\d{6})",           "%Y%m%d_%H%M%S"),  # generic: 20260210_172301
            (r"(?<!\d)(\d{8})(?!\d)",    "%Y%m%d"),          # TROPOMI: 20240810
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
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import logging
    logging.basicConfig(level=logging.DEBUG)

    data_loader = DataLoader()

    ds = data_loader.download_dataset_harmony(
        collection_id="C2215175232-GES_DISC",  # OMI NO2
        temporal=("2024-04-08T00:00:00Z", "2024-04-08T23:59:59Z"),
        bounding_box=(-106.6458, 25.8371, -93.5078, 36.5005),
        max_results=1,
        cache_path="cache_test.zarr"
    )

    print("\n=== Dataset structure ===")
    print("Data vars:", list(ds.data_vars))
    print("Coords:   ", list(ds.coords))
    print("Dims:     ", dict(ds.sizes))
    print()
    for var in ds.data_vars:
        print(f"  {var}: {ds[var].dims} {ds[var].shape}")


if __name__ == "__main__":
    main()