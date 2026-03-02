import json
import sys
import os
from langchain.tools import tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.date_time import parse_date_time, parse_temporal_range


@tool
def convert_date_to_iso(date_str: str) -> str:
    """
    Docstring for convert_date_to_iso
    
    :param date_str: Description
    :type date_str: str
    :return: Description
    :rtype: str
    """
    return "0"


@tool
def convert_temporal_range_to_iso(start_str: str, end_str: str) -> str:
    """
    Docstring for convert_temporal_range_to_iso
    
    :param start_str: Description
    :type start_str: str
    :param end_str: Description
    :type end_str: str
    :return: Description
    :rtype: str
    """
    return "0"
