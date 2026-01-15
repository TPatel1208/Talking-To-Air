import numpy as np
import xarray as xr
from typing import Optional, Tuple

def get_time_index_local(
    ds: xr.Dataset,
    local_hour: int,
    tz_offset: int
) -> int:
    """
    Convert local hour to dataset UTC index.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing a time coordinate.
    local_hour : int
        Local hour in 0–23 format.
    tz_offset : int
        Local timezone offset relative to UTC (e.g., PST = -8).

    Returns
    -------
    time_idx : int
        Index into ds.time representing the requested local hour.
    """
    # Convert local hour to UTC
    if not 0 <= local_hour <= 23:
        raise ValueError(f"local_hour must be between 0 and 23, got {local_hour}")
    
    # Convert local hour to UTC
    utc_hour = (local_hour - tz_offset) % 24
    
    # Check for hour coordinate first, then time coordinate
    if 'hour' in ds.coords:
        matching_indices = np.where(ds.hour.values == utc_hour)[0]
    elif 'time' in ds.coords:
        hours = ds.time.dt.hour.values
        matching_indices = np.where(hours == utc_hour)[0]
    else:
        raise ValueError("Dataset must have either 'hour' or 'time' coordinate")
    
    if len(matching_indices) == 0:
        raise ValueError(
            f"Data not found for local hour {local_hour} "
            f"(UTC hour {utc_hour}) with timezone offset {tz_offset}"
        )
    
    return int(matching_indices[0])
    



def compute_statistic(
    data_array: xr.DataArray,
    statistic: str,
    round_digits: Optional[int] = 3
) -> float:
    """
    Compute a statistic (mean/min/max/median) for a DataArray.

    Parameters
    ----------
    data_array : xr.DataArray
        The data values to analyze.
    statistic : str
        One of {'mean', 'max', 'min', 'median'}.

    Returns
    -------
    value : float
        The computed statistic.
    """
    if not isinstance(data_array, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray, got {type(data_array)}")
    
    # Define statistic functions mapping
    stat_functions = {
        'mean': data_array.mean,
        'max': data_array.max,
        'min': data_array.min,
        'median': data_array.median
    }
    
    # Validate statistic
    if statistic not in stat_functions:
        valid = ', '.join(stat_functions.keys())
        raise ValueError(
            f"Unsupported statistic: '{statistic}'. "
            f"Must be one of: {valid}"
        )
    
    # Compute statistic
    result = stat_functions[statistic]().item()
    
    # Handle NaN
    if np.isnan(result):
        return result  # Return NaN as-is, let caller handle
    
    # Optional rounding
    if round_digits is not None:
        result = round(result, round_digits)
    
    return result



def daily_cycle_peak(
    data_array: xr.DataArray
) -> Tuple[int, float]:
    """
    Identify the hour with the highest spatial average value.

    Parameters
    ----------
    data_array : xr.DataArray
        A 24-hour variable with 'hour' dimension.

    Returns
    -------
    peak_hour : int
        Hour (0–23) with the highest spatial mean.
    peak_value : float
        The mean value at the peak hour.
    """
    if 'hour' not in data_array.dims:
        raise ValueError(
            f"DataArray must have 'hour' dimension. "
            f"Found dimensions: {list(data_array.dims)}"
        )
    
    # Compute spatial mean for each hour
    hourly_means = data_array.mean(dim=[d for d in data_array.dims if d != 'hour'])
    
    # Handle all-NaN case
    if hourly_means.isnull().all():
        return -1, float('nan')
    
    # Find peak
    peak_hour = int(hourly_means.idxmax(dim='hour').item())
    peak_value = float(hourly_means.sel(hour=peak_hour).item())
    
    return peak_hour, peak_value


