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

    This tool is intended for user inputs like "tomorrow at 3pm" or
    "2026-02-10 18:00".  The returned value is a JSON string containing
    ``start_date`` and ``end_date`` where ``end_date`` is exactly one
    hour after ``start_date``.  The timestamps are formatted with a
    trailing ``Z`` to indicate UTC, which is what the downstream Harmony
    API expects.

    Example return value::
        "{\"start_date\": \"YYYY-MM-DDT00:00:00Z\", \"end_date\": \"YYYY-MM-DDT01:00:00Z\"}"

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
    Turn two natural language date expressions into an ISO 8601 date range.

    Uses :func:`utils.date_time.parse_temporal_range` to normalize the
    inputs to ``YYYY-MM-DD`` strings.  The returned JSON encodes
    ``start_date`` and ``end_date`` timestamps suitable for the Harmony
    API.  We interpret the start of a day at midnight UTC and the end of a
    day at 23:59:59Z so that a range like "january 1 to january 5" covers
    the full five days.

    Example::
        "{\"start_date\": \"2026-01-01T00:00:00Z\", \"end_date\": \"2026-01-05T23:59:59Z\"}"

    Args:
        start_str (str): free‑form start date expression.
        end_str (str): free‑form end date expression.

    Returns:
        dict: Keys 'start_date' and 'end_date' (ISO 8601 strings) or 'error'.
    """
    try:
        start_date, end_date = parse_temporal_range(start_str, end_str)
        return {
            "start_date": f"{start_date}Z",
            "end_date": f"{end_date}Z",
        }
    except Exception as e:
        return {"error": str(e)}


