from dotenv import load_dotenv
import os
import earthaccess
import logging 
from typing import Tuple, List, Optional
from pathlib import Path
import xarray as xr

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
        
        

    def download_file(
            self, 
            save_dir: str,
            short_name: str, 
            temporal: Tuple[str, str], 
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
            results = earthaccess.search_data(
                short_name=short_name,
                temporal=temporal,
                bounding_box=bounding_box if bounding_box else None
            )
            
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
        bounding_box: Optional[Tuple[float, float, float, float]] = None) -> xr.Dataset:
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
            ds = xr.open_mfdataset(file_handles) 
            
            logger.info(f"Successfully loaded dataset with {len(results)} granule(s)")
            return ds
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise RuntimeError(f"Failed to load {short_name} data: {e}")