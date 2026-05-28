"""
repositories/zarr_repository.py
===============================
Abstraction over xarray's Zarr backend.

Handles reading and writing datasets to a Zarr store, with support for
named groups (e.g., for different time ranges or regions).

Example
-------
    from repositories.zarr_repository import ZarrRepository

    repo = ZarrRepository("cache.zarr")

    # Write
    ds = xr.Dataset({...})
    repo.write(ds, group_key="OMI_NO2/2025-01-01T00:00:00Z_2025-01-02T00:00:00Z/-74_40_-73_41")

    # Read
    ds = repo.read(group_key="OMI_NO2/2025-01-01T00:00:00Z_2025-01-02T00:00:00Z/-74_40_-73_41")

    # Check existence
    if repo.exists(group_key):
        ...
"""

import logging
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)


class ZarrRepository:
    """Read and write datasets to a Zarr store."""

    def __init__(self, store_path: str):
        """
        Parameters
        ----------
        store_path : str
            Path to the Zarr store directory (created if it doesn't exist).
        """
        self.store_path = str(store_path)

    def exists(self, group_key: str) -> bool:
        """Check if a group exists in the Zarr store."""
        try:
            xr.open_zarr(self.store_path, group=group_key, consolidated=False)
            return True
        except (KeyError, FileNotFoundError, ValueError):
            return False

    def read(self, group_key: str) -> xr.Dataset:
        """
        Read a dataset from the Zarr store.

        Parameters
        ----------
        group_key : str
            Group name within the Zarr store.

        Returns
        -------
        xr.Dataset

        Raises
        ------
        KeyError
            If the group doesn't exist.
        """
        logger.debug("Reading Zarr group: %s from %s", group_key, self.store_path)
        ds = xr.open_zarr(self.store_path, group=group_key, consolidated=False)
        logger.info(
            "Loaded from Zarr — group=%s dims=%s",
            group_key,
            dict(ds.dims),
        )
        return ds

    def write(self, ds: xr.Dataset, group_key: str) -> None:
        """
        Write a dataset to the Zarr store.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to write.
        group_key : str
            Group name within the store (created if it doesn't exist).
        """
        logger.debug("Writing Zarr group: %s to %s", group_key, self.store_path)
        ds.to_zarr(self.store_path, group=group_key, mode="w", consolidated=True)
        logger.info(
            "Written to Zarr — group=%s dims=%s",
            group_key,
            dict(ds.dims),
        )

    def __repr__(self) -> str:
        return f"ZarrRepository({self.store_path!r})"