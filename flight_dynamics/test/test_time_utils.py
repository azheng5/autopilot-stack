import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import time_utils

def test_generate_time_grid():

    initial_utc_str = "2025-01-01T00:00:00.000"
    time_span_sec = 300
    delta_t = 30

    time_grid = time_utils.generate_time_grid(initial_utc_str, time_span_sec, delta_t)

    print(time_grid)

if __name__ == "__main__":
    test_generate_time_grid()