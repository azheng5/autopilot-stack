import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics.Eclipse import Eclipse

def test_check_eclipse():

    r_sc = np.array([7000,0,0])
    curr_utc_time = '2000-01-01T11:58:55.816073'

    is_eclipse = Eclipse.check_eclipse(r_sc, curr_utc_time)

# Debugging mode
if __name__ == "__main__":
    test_check_eclipse()