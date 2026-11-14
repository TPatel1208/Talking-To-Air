"""numpy ufuncs on xarray objects, typed as what they return at runtime.

``np.isfinite(da)`` dispatches through ``DataArray.__array_ufunc__`` and returns
a DataArray, but numpy's stubs declare an ndarray, so every ``.any(dims)`` /
``.sum(dims)`` / ``.groupby(...)`` chained on it fails type checking. These
wrappers change nothing at runtime; they only state the real return type.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import xarray as xr


def isfinite(da: xr.DataArray) -> xr.DataArray:
    """``np.isfinite(da)``: True where ``da`` is neither NaN nor +-inf."""
    return cast(xr.DataArray, np.isfinite(da))


def absolute(da: xr.DataArray) -> xr.DataArray:
    """``np.abs(da)``."""
    return cast(xr.DataArray, np.abs(da))
