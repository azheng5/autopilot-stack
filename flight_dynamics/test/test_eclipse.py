import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics import eclipse_utils

def test_check_eclipse():

    r_sc = np.array([7000,0,0])
    curr_utc_time = '2000-01-01T11:58:55.816073'

    is_eclipse = eclipse_utils.check_eclipse(r_sc, curr_utc_time)

def test_brute_force_eclipse_angles():

    # At low altitudes and eccentricities, the cylindrical shadow assumption means 
    # that the orbit within the cylinder at all times, which means there are 
    # no intersections with the cylinder. Need to be careful that no entry/exit points
    # doesn't imply no eclipse.

    sma = Constants.R_EARTH + 500
    ecc = 0.01
    
    # inc = astro_utils.compute_sso_inc(sma, ecc)
    inc = 0.0 * (np.pi/180)
    raan = 0.0 * (np.pi / 180)
    aop = 0.0 * (np.pi / 180)
    ta = 0.0 * (np.pi / 180)
    kep_state = np.array([sma, ecc, inc, raan, aop, ta])
    utc_str = '2025-01-01T00:00:00.000000'

    ta_entry, ta_exit = eclipse_utils.brute_force_eclipse_angles(kep_state, utc_str)

def test_compute_eclipse_angles():

    # At low altitudes and eccentricities, the cylindrical shadow assumption means 
    # that the orbit within the cylinder at all times, which means there are 
    # no intersections with the cylinder. Need to be careful that no entry/exit points
    # doesn't imply no eclipse.

    sma = Constants.R_EARTH + 500
    ecc = 0.01
    
    # inc = astro_utils.compute_sso_inc(sma, ecc)
    inc = 0.0 * (np.pi/180)
    raan = 0.0 * (np.pi / 180)
    aop = 0.0 * (np.pi / 180)
    ta = 0.0 * (np.pi / 180)
    kep_state = np.array([sma, ecc, inc, raan, aop, ta])
    utc_str = '2025-01-01T00:00:00.000000'

    start_time = time.perf_counter()
    ta_entry, ta_exit = eclipse_utils.compute_eclipse_angles(kep_state, utc_str)
    end_time = time.perf_counter()
    print(f"{end_time-start_time}")

    # start_time = time.perf_counter()
    # ta_entry, ta_exit = eclipse_utils.brute_force_eclipse_angles(kep_state, utc_str)
    # end_time = time.perf_counter()
    # print(f"{end_time-start_time}")

# Debugging mode
if __name__ == "__main__":
    # test_check_eclipse()
    # test_brute_force_eclipse_angles()
    test_compute_eclipse_angles()