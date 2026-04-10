import requests
import time

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from shapely.geometry import box, shape, Polygon, MultiPolygon
from rasterio.features import rasterize
from affine import Affine

from typing import Optional, Tuple, Union, List


def plot_map(
    data_array: xr.DataArray,
    title: str = "",
    extent: Optional[Tuple[float, float, float, float]] = None,
    mask_geometry: Optional[Union[Polygon, MultiPolygon]] = None,
    cmap: str = "Spectral_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentile_scale: bool = True,
    add_gridlines: bool = True,
    time_slice: Optional[int] = None
):
    """
    Plot air quality data on a Cartopy map with proper extent and masking.
    
    Parameters
    ----------
    data_array : xarray.DataArray
        Variable to be plotted (e.g., NO2, O3, PM2.5)
        Can be 2D (lat, lon) or 3D (time, lat, lon)
        Must have latitude/longitude coordinates
    title : str, optional
        Plot title
    extent : tuple, optional
        Map bounds from RegionResult.bounds: (minx, miny, maxx, maxy))
        This is the CORRECT format from shapely.geometry.bounds
    mask_geometry : Polygon or MultiPolygon, optional
        Region geometry for masking data outside boundaries
        Use RegionResult.geometry here
    cmap : str, optional
        Matplotlib colormap name
        Good choices: 'viridis', 'plasma', 'YlOrRd', 'RdYlBu_r'
    vmin, vmax : float, optional
        Explicit color scale limits
        If None and percentile_scale=True, uses 2nd-98th percentile
    percentile_scale : bool, optional
        If True, automatically compute vmin/vmax from percentiles
        Ignored if vmin/vmax are explicitly provided
    add_gridlines : bool, optional
        Whether to add lat/lon gridlines with labels
    time_slice : int, optional
        For 3D data, which time index to plot
        If None, uses first time slice (index 0)
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        For further customization or saving
        
        
    """
    
    # --- 0. Handle 3D data - select time slice ---
    if data_array.ndim == 3:
        # Find the time dimension
        time_dims = ['time', 'Time', 'TIME', 't']
        time_dim = None
        
        for dim in time_dims:
            if dim in data_array.dims:
                time_dim = dim
                break
        
        if time_dim is None:
            # Assume first dimension is time if not found
            time_dim = data_array.dims[0]
        
        # Select time slice
        if time_slice is None:
            time_slice = 0
        
        print(f"Selecting time slice {time_slice} from dimension '{time_dim}'")
        time_size = data_array.sizes[time_dim]

        if time_size == 1:
            data_array = data_array.isel({time_dim: 0})  # just take the only one
        else:
            time_slice = min(time_slice, time_size - 1)  # clamp to valid range
            data_array = data_array.isel({time_dim: time_slice})
    
    # --- 1. Apply geometry mask if provided ---
    if mask_geometry is not None:
        data_array = mask_data_by_geometry(data_array, mask_geometry)
    
    # --- 1.5. Find coordinate names (handle different conventions) ---
    lat_names = ['lat', 'latitude', 'Latitude', 'LAT']
    lon_names = ['lon', 'longitude', 'Longitude', 'LON', 'long']
    
    lat_coord = None
    lon_coord = None
    
    for name in lat_names:
        if name in data_array.coords:
            lat_coord = name
            break
    
    for name in lon_names:
        if name in data_array.coords:
            lon_coord = name
            break
    
    if lat_coord is None or lon_coord is None:
        raise ValueError(
            f"Could not find lat/lon coordinates. "
            f"Available: {list(data_array.coords.keys())}"
        )
    
    # --- 2. Compute adaptive color scale ---
    if vmin is None or vmax is None:
        if percentile_scale:
            valid_data = data_array.values[~np.isnan(data_array.values)]
            if len(valid_data) > 0:
                if vmin is None:
                    vmin = np.percentile(valid_data, 2)
                if vmax is None:
                    vmax = np.percentile(valid_data, 98)
        else:
            # Fallback to min/max
            if vmin is None:
                vmin = np.nanmin(data_array.values)
            if vmax is None:
                vmax = np.nanmax(data_array.values)
    
    # --- 3. Calculate figure size based on extent aspect ratio ---
    if extent:
        lon_range = extent[2] - extent[0]  # maxx - minx   
        lat_range = extent[3] - extent[1]  # maxy - miny
        aspect_ratio = lon_range / lat_range
        
        # Base height of 6 inches, adjust width
        fig_height = 6
        fig_width = fig_height * aspect_ratio * 1.2  # 1.2 factor for map projection
        fig_width = np.clip(fig_width, 6, 14)  # Reasonable bounds
    else:
        fig_width, fig_height = 10, 6
    
    # --- 4. Create figure with Cartopy projection ---
    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height),
        dpi=150,
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    # --- 5. Plot the data ---
    # Use pcolormesh explicitly to avoid xarray auto-detection issues
    im = ax.pcolormesh(
        data_array[lon_coord].values,
        data_array[lat_coord].values,
        data_array.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto'
    )
    
    # Add colorbar manually
    cbar = plt.colorbar(
        im,
        ax=ax,
        shrink=0.7,
        extend='both',
        label=data_array.attrs.get('long_name', data_array.name or '')
    )
    
    # --- 6. Set extent (FIXED: correct bounds format) ---
    if extent:
        # Convert from shapely bounds (minx, miny, maxx, maxy)
        # to Cartopy extent [lon_max, lat_max, lon_min, lat_min]  
        cartopy_extent = [extent[0], extent[2], extent[1], extent[3]]
        ax.set_extent(cartopy_extent, crs=ccrs.PlateCarree())
    
    # --- 7. Add geographic features ---
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='black', alpha=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
    
    # Optional: add region boundary outline
    if mask_geometry is not None:
        ax.add_geometries(
            [mask_geometry],
            crs=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor='red',
            linewidth=2,
            alpha=0.8
        )
    
    # --- 8. Add gridlines with labels ---
    if add_gridlines:
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color='gray',
            alpha=0.5,
            linestyle='--'
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
    
    # --- 9. Set title ---
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    return fig, ax

def _normalize_to_2d(data_array: xr.DataArray) -> xr.DataArray:
    """
    Squeeze a DataArray down to 2D (lat, lon) by:
      1. Dropping all size-1 dimensions (handles Time=1 cleanly)
      2. Averaging over any remaining non-spatial dimensions (handles 4D layer dim)
    """
    SPATIAL = {
        'lat', 'latitude', 'Latitude', 'LAT',
        'lon', 'longitude', 'Longitude', 'LON', 'long'
    }

    # squeeze out all size-1 dims
    dims_to_squeeze = [d for d in data_array.dims if data_array.sizes[d] == 1]
    if dims_to_squeeze:
        data_array = data_array.squeeze(dims_to_squeeze)

    # average over any remaining non-spatial dims
    extra_dims = [d for d in data_array.dims if d not in SPATIAL]
    if extra_dims:
        print(f"_normalize_to_2d: averaging over {extra_dims}")
        data_array = data_array.mean(dim=extra_dims)

    return data_array

def mask_data_by_geometry(
    data_array: xr.DataArray,
    geometry: Union[Polygon, MultiPolygon]
) -> xr.DataArray:
    """
    Mask xarray data to only show values within a geometry boundary.
    Sets all points outside the geometry to NaN.
    
    Parameters
    ----------
    data_array : xarray.DataArray
        Data with latitude/longitude coordinates
        Handles common coord names: 'lat'/'latitude', 'lon'/'longitude'
    geometry : Polygon or MultiPolygon
        Boundary geometry from RegionResult
        
    Returns
    -------
    xarray.DataArray
        Masked data array (copy, original unchanged)
    """
    
    # Find lat/lon coordinates (handle different naming conventions)
    lat_names = ['lat', 'latitude', 'Latitude', 'LAT']
    lon_names = ['lon', 'longitude', 'Longitude', 'LON', 'long']
    
    lat_coord = None
    lon_coord = None
    
    for name in lat_names:
        if name in data_array.coords:
            lat_coord = name
            break
    
    for name in lon_names:
        if name in data_array.coords:
            lon_coord = name
            break
    
    if lat_coord is None or lon_coord is None:
        raise ValueError(
            f"Could not find lat/lon coordinates. "
            f"Available coords: {list(data_array.coords.keys())}"
        )
    
    # Get coordinate arrays
    lats = data_array[lat_coord].values
    lons = data_array[lon_coord].values
    
    # Calculate the affine transform for the raster
    lon_res = (lons[-1] - lons[0]) / (len(lons) - 1) if len(lons) > 1 else 1
    lat_res = (lats[-1] - lats[0]) / (len(lats) - 1) if len(lats) > 1 else 1
    
    transform = Affine.translation(lons[0] - lon_res/2, lats[0] - lat_res/2) * \
                Affine.scale(lon_res, lat_res)
    
    # Rasterize the geometry
    # 1 = inside geometry, 0 = outside
    mask_2d = rasterize(
        [(geometry, 1)],
        out_shape=(len(lats), len(lons)),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    
    # Convert to boolean (True = INSIDE geometry = keep)
    mask_2d = (mask_2d == 1)
    
    # Handle different array dimensions
    if data_array.ndim == 2:
        mask_da = xr.DataArray(mask_2d, dims=[lat_coord, lon_coord])

    elif data_array.ndim == 3:
        # Find time dimension
        time_dim = [d for d in data_array.dims if d not in [lat_coord, lon_coord]][0]
        # Broadcast mask across time dimension
        mask_3d = np.broadcast_to(mask_2d, data_array.shape)
        mask_da = xr.DataArray(mask_3d, dims=data_array.dims)

    else:
        raise ValueError(f"Unsupported array dimension: {data_array.ndim}D")
    
    # Apply mask using xarray .where() — sets outside points to NaN
    masked = data_array.where(mask_da)
    
    return masked
def plot_diff_maps(
    data_arrays: List[xr.DataArray],
    titles: List[str],
    extent: Optional[List[Tuple[float, float, float, float]]] = None,
    cmap: str = "Spectral_r",
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 150,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mask_geometries: Optional[List[Union[Polygon, MultiPolygon]]] = None
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot several difference maps side-by-side for comparison.

    Parameters
    ----------
    data_arrays : list of xr.DataArray
        Difference fields to display.
    titles : list of str
        Titles for each subplot.
    extent : tuple, optional
        Bounding box for all subplots.
    cmap : str, optional
        Diverging colormap for differences.
    figsize : tuple, optional
        Figure size. If None, computed automatically.
    dpi : int, optional
        Output resolution.
    symmetric : bool, optional
        Force symmetric colorbar around zero.
    vmin, vmax : float, optional
        Manual color limits.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    n_plots = len(data_arrays)
    if len(titles) != n_plots:
        raise ValueError(f"Number of titles ({len(titles)}) must match number of data arrays ({n_plots})")
    
    if n_plots == 0:
        raise ValueError("Must provide at least one data array to plot")
    
    # Determine layout
    if n_plots <= 3:
        nrows, ncols = 1, n_plots
    else:
        ncols = 3
        nrows = int(np.ceil(n_plots / ncols))
    
    # Auto compute figsize
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)

    # Symmetric color scale
    if vmin is None or vmax is None:
        values = np.hstack([da.values.flatten() for da in data_arrays])
        values = values[np.isfinite(values)]
        if len(values) > 0:
            abs_max = np.max(np.abs(values))
            vmin = 0 if vmin is None else vmin
            vmax = abs_max if vmax is None else vmax

    # Make figure
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        dpi=dpi,
        subplot_kw={'projection': ccrs.PlateCarree()},
        squeeze=False
    )
    axes_flat = axes.flatten()

    # Plot each map
    for idx, (da, title) in enumerate(zip(data_arrays, titles)):
        ax = axes_flat[idx]
        im = da.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False
        )

        if extent:
            cartopy_extent = [extent[idx][0], extent[idx][2], extent[idx][1], extent[idx][3]]
            ax.set_extent(cartopy_extent, crs=ccrs.PlateCarree())

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='black', alpha=0.5)
    if mask_geometries is not None:
        for idx, mask_geometry in enumerate(mask_geometries):
            axes_flat[idx].add_geometries(
                [mask_geometry],
                crs=ccrs.PlateCarree(),
                facecolor='none',
                edgecolor='red',
                linewidth=2,
                alpha=0.8
            )
    # Hide unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    name = data_arrays[0].name or "Difference"
    units = getattr(data_arrays[0], "units", "")
    cbar.set_label(f"{name} {f'({units})' if units else ''}")

    plt.tight_layout(rect=[0, 0, 0.90, 1])

    return fig, axes_flat[:n_plots].tolist()



class GeocodingService:
    """Free geocoding using Nominatim (OpenStreetMap) with polygon and bounding box"""
    
    def __init__(self):
        self.cache = {}
        self.last_request = 0
    
    def geocode(self, location_name):
        """Convert location name to coordinates, polygon, and bounding box"""
        # Check cache first
        if location_name in self.cache:
            return self.cache[location_name]
        
        # Rate limit: 1 request per second
        time_since_last = time.time() - self.last_request
        if time_since_last < 1.0:
            time.sleep(1.0 - time_since_last)
        
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': location_name,
            'format': 'json',
            'limit': 1,
            'polygon_geojson': 1  # Request polygon boundary
        }
        headers = {
            'User-Agent': '(Educational project)'
        }
        
        self.last_request = time.time()
        
        try:
            response = requests.get(url, params=params, headers=headers)
            data = response.json()
            
            if data:
                item = data[0]
                
                # Centroid
                latitude = float(item['lat'])
                longitude = float(item['lon'])
                
                # Polygon (GeoJSON)
                polygon = item.get('geojson', None)
                
                # Bounding box: [south, north, west, east] as floats
                bbox = [float(coord) for coord in item.get('boundingbox', [])]
                
                result = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'display_name': item['display_name'],
                    'polygon': polygon,  # None if not available
                    'bbox': bbox         # None if not available
                }
                
                self.cache[location_name] = result
                return result
        except Exception as e:
            print(f"Geocoding error: {e}")
        
        return None
    

class RegionResolver: 
    """resolves user location inputs into singular plot or multiple plots"""
    def __init__(self):
        self.geocoding_service = GeocodingService()

    def resolve_location(self, location_name: str):
        """Convert location name to RegionResult with geometry"""
        geo_result = self.geocoding_service.geocode(location_name)
        if geo_result is None:
            return None
        
        # Convert GeoJSON polygon to shapely geometry
        if geo_result['polygon']:
            geometry = shape(geo_result['polygon'])
        else:
            # Fallback: create small box around point
            lon, lat = geo_result['longitude'], geo_result['latitude']
            delta = 0.1  # degrees
            geometry = box(lon - delta, lat - delta, lon + delta, lat + delta)
        
        return {
            'geometry': geometry,
            'bounds': geometry.bounds,  # (minx, miny, maxx, maxy)
            'name': geo_result['display_name']
        }
        
    def plot_singular(self, data_array, location_name, **kwargs):
        """Plot data for a single location"""
        region = self.resolve_location(location_name)
        if region is None:
            raise ValueError(f"Could not find location: {location_name}")
        
        # Extract title if provided in kwargs, otherwise use default
        title = kwargs.pop('title', f"Air Quality over {region['name']}")
        
        return plot_map(
            data_array,
            title=title,
            extent=region['bounds'],
            mask_geometry=region['geometry'],
            **kwargs
        )
    def plot_multiple(self, data_array, location_names, **kwargs):
        """Plot data for multiple locations for comparison"""
        regions = []
        for name in location_names:
            region = self.resolve_location(name)
            if region:
                regions.append(region)
        
        data_arrays = [
            mask_data_by_geometry(data_array, r['geometry']) 
            for r in regions
        ]
        titles = [r['name'] for r in regions]
        extents = [r['bounds'] for r in regions]
        geometries = [r['geometry'] for r in regions]
        
        return plot_diff_maps(
            data_arrays,
            titles,
            extent=extents,
            mask_geometries=geometries,
            **kwargs
        )