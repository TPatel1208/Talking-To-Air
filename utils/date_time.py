import parsedatetime as pdt
from datetime import datetime
from typing import Tuple

cal = pdt.Calendar()

def parse_date_time(date_time_str: str) -> datetime:
    time_struct, parse_status = cal.parse(date_time_str)
    if parse_status == 0:
        raise ValueError(f"Could not parse date/time: '{date_time_str}'")
    return datetime(*time_struct[:6])  # returns datetime object

def parse_temporal_range(start_str: str, end_str: str) -> Tuple[str, str]:
    start = parse_date_time(start_str)
    end = parse_date_time(end_str)
    if end < start:
        raise ValueError(f"End date {end} is before start date {start}")
    return (start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S"))


def main():
    # Example usage
    print(parse_date_time("february 1 2024 at 3pm"))
    print(parse_temporal_range("january 1", "january 5"))

if __name__ == "__main__":
    main()