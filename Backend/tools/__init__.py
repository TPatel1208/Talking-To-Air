from .ground_sensor_tools.epa_aqs_tools import (
    find_closest_monitor,
    find_closest_monitor_by_coords,
    find_exceedance_days,
    get_annual_summary,
    get_daily_summary,
    get_quarterly_summary,
    get_sample_data,
    list_states,
)
from .satellite_tools.date_tools import (
    convert_date_to_iso as convert_date_to_iso,
    convert_temporal_range_to_iso as convert_temporal_range_to_iso,
)
from .satellite_tools.harmony_api import check_data_availability, fetch_environmental_data, geocode_location
from .satellite_tools.plot_tools import conduct_temporal_statistic, plot_multiple, plot_singular
from .satellite_tools.stat_tools import compute_statistic_tool, find_daily_peak

SATELLITE_TOOLS = [
    geocode_location,
    fetch_environmental_data,
    plot_singular,
    plot_multiple,
    compute_statistic_tool,
    conduct_temporal_statistic,
    find_daily_peak,
    check_data_availability,
]

GROUND_TOOLS = [
    find_closest_monitor,
    find_closest_monitor_by_coords,
    get_daily_summary,
    get_quarterly_summary,
    get_annual_summary,
    get_sample_data,
    find_exceedance_days,
    list_states,
    geocode_location,  # In satellite tools, but useful for ground agent too.
]











