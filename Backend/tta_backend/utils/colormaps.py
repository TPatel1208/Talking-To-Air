"""Single source of truth mapping a variable to its colormap.

Consumed by the overlay PNG renderer, the payload's `colormap.lut`, and
`export_service`'s PNG/legend rendering so the three can never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap

_LUT_SAMPLES = 256

# Sequential variable -> matplotlib (or custom-registered) colormap name.
_SEQUENTIAL_COLORMAPS = {
    "NO2": "no2_omi",
}
_DEFAULT_SEQUENTIAL = "viridis"
_DIVERGING_COLORMAP = "RdBu_r"

_NO2_OMI_NAME = "no2_omi"
# Placeholder NASA/OMI-style tropospheric-NO2 stops (dark purple -> blue ->
# green -> yellow -> red). TODO: replace with the exact NASA/OMI NO2 LUT
# stops once a reference source is confirmed (flagged as an open build-time
# unknown in the T23 PRD).
_NO2_OMI_STOPS = [
    (0.0, (28, 28, 92)),
    (0.2, (30, 100, 180)),
    (0.4, (40, 170, 120)),
    (0.6, (190, 210, 40)),
    (0.8, (240, 130, 30)),
    (1.0, (170, 20, 20)),
]


@dataclass(frozen=True)
class ColormapResolution:
    name: str
    lut: list[list[int]]  # [N][4] RGBA, 0-255


def resolve(variable: str | None, *, diverging: bool = False) -> ColormapResolution:
    if diverging:
        return _resolve_named(_DIVERGING_COLORMAP)

    name = _SEQUENTIAL_COLORMAPS.get((variable or "").upper(), _DEFAULT_SEQUENTIAL)
    return _resolve_named(name)


def _resolve_named(name: str) -> ColormapResolution:
    cmap = mpl.colormaps[name]
    samples = cmap(np.linspace(0.0, 1.0, _LUT_SAMPLES))
    lut = [[int(round(channel * 255)) for channel in rgba] for rgba in samples]
    return ColormapResolution(name=name, lut=lut)


def _sampled_stops_to_rgba(stops, n: int) -> np.ndarray:
    positions = np.array([p for p, _ in stops])
    colors = np.array([c for _, c in stops], dtype=float) / 255.0
    xs = np.linspace(0.0, 1.0, n)
    channels = [np.interp(xs, positions, colors[:, i]) for i in range(3)]
    rgba = np.ones((n, 4))
    rgba[:, 0], rgba[:, 1], rgba[:, 2] = channels
    return rgba


def _register_no2_omi() -> None:
    if _NO2_OMI_NAME in mpl.colormaps:
        return
    rgba = _sampled_stops_to_rgba(_NO2_OMI_STOPS, _LUT_SAMPLES)
    mpl.colormaps.register(ListedColormap(rgba, name=_NO2_OMI_NAME))


_register_no2_omi()
