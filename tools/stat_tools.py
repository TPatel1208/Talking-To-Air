import json
import sys
import os
from langchain.tools import tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.statistics import compute_statistic, daily_cycle_peak, get_time_index_local

@tool
def compute_statistic_tool(data_json: str, statistic: str="mean") -> str:
    """
    Docstring for compute_statistic_tool
    
    :param data_json: Description
    :type data_json: str
    :param statistic: Description
    :type statistic: str
    :return: Description
    :rtype: str
    """
    return "0"


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