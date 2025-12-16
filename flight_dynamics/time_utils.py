import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import spiceypy as spice

sys.path.append(str(Path(__file__).parent.parent))
from flight_dynamics import Constants

def generate_time_grid(initial_utc_str: str, 
                       time_span_sec: float,
                       delta_t: float) -> List[str]:
    """
    Generate time array of UTC time strings at discrete time intervals.
    """

    initial_dt = datetime.strptime(initial_utc_str, Constants.UTC_FORMAT)

    time_grid = []
    steps = int(time_span_sec // delta_t) + 1

    for i in range(steps):
        t = initial_dt + timedelta(seconds=i*delta_t)
        time_grid.append(t.strftime(Constants.UTC_FORMAT))

    return time_grid