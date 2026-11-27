"""The one ``render`` group every render module registers its capabilities under."""
from __future__ import annotations

from tta_backend.harness.capabilities.registry import Group

GROUP = Group(
    name="render",
    description="Draw handles you already have as a map, monitor map, time series, vertical profile or table.",
    notes=(
        "Every render capability takes handle ids, stores one chart or table for the user and returns its id in "
        "charts with a short summary; it makes no new data handle and never retrieves. Rendering works on the "
        "handle's kind, not where it came from. "
        "map draws a grid; a grid with a time axis is shown as its time mean, and the result says so. Add points "
        "(such as monitor locations) and polygons (such as smoke extent) to carry them in the chart data as "
        "overlays. Use style=diverging for a difference, so zero is the centre of the colour scale. "
        "monitor_map maps monitors that carry a value, each given its pollutant's AQI category and colour: pass "
        "points with a value, or a monitor summary or exceedance table (site_id, name, lat, lon, value, unit), "
        "whose rows are reduced to one value per site with stat=max (default) or mean; give pollutant when the "
        "input's quantity does not name it. "
        "timeseries puts one or several series handles on one chart, one line per series line; lines in a second "
        "unit get a second y axis, and three or more units are refused. "
        "polygon_map draws polygons handles (such as smoke extent) when there is no grid; no polygons gives an "
        "empty map with a disclosure, and points can be added as an overlay. "
        "profile draws the table operations.vertical_profile makes. "
        "table shows a table handle, cut to max_rows with a disclosure. "
        "Disclosures say what the chart data holds; repeat them to the user. "
        "A refusal names its reason (wrong_dims, no_values, missing_columns, mixed_units, unknown_pollutant, "
        "unit_mismatch, too_many_units, too_many_lines, not_a_profile): do not retry it with the same inputs."
    ),
)
