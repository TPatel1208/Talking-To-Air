import parsedatetime as pdt
from datetime import datetime
from typing import Tuple

cal = pdt.Calendar()

def parse_date_time(date_time_str: str) -> datetime:
    time_struct, parse_status = cal.parse(date_time_str)
    if parse_status == 0:
        return "Invalid date/time string"
    return datetime(*time_struct[:6])

def parse_temporal_range(start_str: str, end_str: str) -> Tuple[str, str]:
    start = parse_date_time(start_str)
    end = parse_date_time(end_str)

    if end < start:
        raise ValueError(f"End date {end} is before start date {start}")
    
    return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


print(parse_date_time("next Monday at 3pm"))