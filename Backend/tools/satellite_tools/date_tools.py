import json
import sys
import os
from langchain.tools import tool
from datetime import timedelta


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.date_time import parse_date_time, parse_temporal_range


@tool
def convert_date_to_iso(date_str: str) -> dict:
    """
    Convert a natural language date/time string to an ISO 8601 range.

    Args:
        date_str (str): free‑form date/time expression.

    Returns:
        str: JSON encoded object with iso timestamps or an ``error`` key on failure.
    """
    try:
        dt = parse_date_time(date_str)
        return {
            "start_date": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_date": (dt + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except Exception as e:
        return{"error": str(e)}



@tool
def convert_temporal_range_to_iso(start_str: str, end_str: str) -> dict:
    """
    Turn two natural language date expressions into a full-day ISO 8601 date range.
    Always spans from 00:00:00Z on the start date to 23:59:59Z on the end date.

    Args:
        start_str (str): free‑form start date expression.
        end_str (str): free‑form end date expression.

    Returns:
        dict: Keys 'start_date' and 'end_date' (ISO 8601 strings) or 'error'.
    """
    try:
        start_date, end_date = parse_temporal_range(start_str, end_str)
        return {
            "start_date": f"{start_date[:10]}T00:00:00Z",
            "end_date":   f"{end_date[:10]}T23:59:59Z",
        }
    except Exception as e:
        return {"error": str(e)}


