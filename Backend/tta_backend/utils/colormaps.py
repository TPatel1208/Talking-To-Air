"""The colormap a variable maps to: the overlay renderer, the payload's LUT and the export share it."""
from __future__ import annotations

from tta_backend.domain.charts.colormaps import ColormapResolution as ColormapResolution
from tta_backend.domain.charts.colormaps import by_name as by_name
from tta_backend.domain.charts.colormaps import resolve as resolve

__all__ = ["ColormapResolution", "by_name", "resolve"]
