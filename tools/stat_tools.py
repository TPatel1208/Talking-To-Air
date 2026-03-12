import json
import sys
import os
from langchain.tools import tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.statistics import compute_statistic, daily_cycle_peak, get_time_index_local
from utils.data_utils import _load_data

@tool
def compute_statistic_tool(data_json: str, statistic: str="mean") -> str:
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
    da = _load_data(data_json)
    try:
        result = compute_statistic(da, statistic)
        return json.dumps({"statistic": statistic, "value": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def conduct_temporal_statistic(data_json: str, frequency: str = "monthly") -> str:
    """
    Docstring for conduct_temporal_statistic
    
    :param data_json: Description
    :type data_json: str
    :param frequency: Description
    :type frequency: str
    :return: Description
    :rtype: str
    """
    return "0"

@tool
def find_daily_peak(data_json: str) -> str:
    """
    Docstring for find_daily_peak
    
    :param data_json: Description
    :type data_json: str
    :return: Description
    :rtype: str
    """
    return "0"