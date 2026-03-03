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
import json
import concurrent.futures
import netCDF4 as nc

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
            combined = xr.open_zarr(cache_path, group=group_key)
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

            datasets = []
            futures = self.harmony_client.download_all(job_id, directory='./data/downloads', overwrite=True)

            for future in concurrent.futures.as_completed(futures):
                filename = future.result()
                logger.info(f"Downloaded: {filename}")
                ds = self._open_dataset(filename)
                datasets.append(ds)
            
            if len(datasets) == 1:
                combined = datasets[0]
            elif len(datasets) > 1:
                combined = xr.concat(datasets, dim='time')
            else:
                logger.error("No datasets were downloaded")
                raise RuntimeError("Failed to download any datasets from Harmony")

            combined.to_zarr(cache_path, group=group_key, mode='w',consolidated=True)
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
        

    def _open_dataset(self, filename: str) -> xr.Dataset:
        """
        Open a NASA NetCDF4 file, handling different satellite data structures.
        
        Known structures:
            TEMPO  — coordinates at root, data variables inside 'product' group
            TROPOMI — flat structure, all variables at root level
        """
        try:
            root = xr.open_dataset(filename, engine="netcdf4", decode_times=True)
        except Exception as e:
            logger.error(f"Failed to open {filename}: {e}")
            raise

        try:
            with nc.Dataset(filename) as f:
                groups = list(f.groups.keys())
        except Exception:
            groups = []

        logger.debug(f"File groups found: {groups}")

        # --- TEMPO-style: coordinates at root, data in 'product' group ---
        if "product" in groups:
            try:
                product = xr.open_dataset(filename, group="product", engine="netcdf4")
                coords  = {}
                for coord in ["time", "latitude", "longitude"]:
                    if coord in root:
                        coords[coord] = root[coord]
                return product.assign_coords(**coords)
            except Exception as e:
                logger.warning(f"Failed to open 'product' group, falling back: {e}")

        # --- Flat structure (TROPOMI, generic) ---
        return root
        



    

def main():
    import time
    data_loader = DataLoader()
    COLLECTION_ID = "C3685896708-LARC_CLOUD"  # TEMPO NO2 L3 V04 — verify this
    temporal = ("2026-02-10T18:00:00Z","2026-02-10T18:30:00Z")
    BBOX = [-89,24,-81,32]
    start = time.time()
    ds = data_loader.download_dataset_harmony(
        collection_id=COLLECTION_ID,
        temporal=temporal,
        bounding_box=BBOX,
        variables=['product/vertical_column_troposphere'],
        max_results=10,
        output_format='application/x-netcdf4',
        cache_path="cache.zarr"
    )
    
    end = time.time()
    print(f"Dataset loaded in {end - start:.2f} seconds")
    """
    start_time = time.time()
    ds = data_loader.get_dataset(
        short_name="TEMPO_NO2_L3", 
        version="V04",
        temporal=("2026-02-10T18:00:00Z","2026-02-10T18:00:00Z"),
        groups=['product']
    )
    end_time = time.time()
    print(f"Dataset w/o bounding box loaded in {end_time - start_time:.2f} seconds")
    start_time = time.time()
    ds = data_loader.get_dataset(
        short_name="TEMPO_NO2_L3", 
        version="V04",
        temporal=("2026-02-10T18:00:00Z","2026-02-10T18:00:00Z"),
        groups=['product'],
        bounding_box=(-88,25,-81,32) # New York City bounding box
    )
    end_time = time.time()
    print(f"Dataset w bounding box loaded in {end_time - start_time:.2f} seconds")
    """


if __name__ == "__main__":
    main()