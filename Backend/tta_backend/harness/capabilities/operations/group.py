"""The one ``operations`` group every operations module registers its capabilities under."""
from __future__ import annotations

from tta_backend.harness.capabilities.registry import Group

GROUP = Group(
    name="operations",
    description="Compute on handles you already have: reduce over time or area, statistics, differences, daily peaks, thresholds, satellite-to-monitor pairing.",
    notes=(
        "Every operation takes handle ids and returns a new handle with a small summary; it never retrieves data. "
        "aggregate_time turns a grid with a time axis into one map (mean, median, max, min or std per cell). "
        "regional_series turns a grid with a time axis into a series of the area statistic at each time step; "
        "the mean is area-weighted. statistic gives one number for a grid or a series, with the spread of the "
        "values (a standard deviation across cells or values, not an uncertainty of the number) and their count. "
        "difference takes grids a and b on the same grid cells and returns b minus a, absolute or percent, with the "
        "mean, min, max, where the max is, and the fraction of cells paired in its summary; grids are not "
        "regridded, and two products are compared only with a disclosure naming both; use over=one_period when "
        "both should cover the same dates. "
        "daily_peak lists each day's highest value, when and where. threshold lists what is above a level you give "
        "(compared as given) or a pollutant's NAAQS level (values truncated to the reporting precision first, as "
        "for the AQI); it does not compute the standard's form. "
        "pair_with_monitors takes a satellite grid, daily ground readings and the monitors' locations (from "
        "sources.monitors.find): at each monitor's nearest cell it pairs the satellite's daily value with the "
        "monitor's reading for that day, and gives the correlation r and paired days per monitor and pooled; "
        "only the pairs of satellite quantity and pollutant it lists are accepted, and the units differ, so there "
        "is no bias. "
        "A refusal names its reason (unit_mismatch, unit_missing, variable_mismatch, grid_mismatch, "
        "disjoint_periods, empty_overlap, no_time_axis): do not retry it with the same inputs. "
        "insufficient_evidence means too little valid data survived masking, or too few cells pair. "
        "Results of one operation can be passed to another."
    ),
)
