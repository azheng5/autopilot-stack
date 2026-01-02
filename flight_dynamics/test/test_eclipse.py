import sys
import time
from pathlib import Path

import numpy as np
import spiceypy as spice

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics import eclipse_utils
from flight_dynamics import time_utils

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
    ecc = 1e-6
    
    # inc = astro_utils.compute_sso_inc(sma, ecc)
    inc = 0.0 * (np.pi/180)
    raan = 0.0 * (np.pi / 180)
    aop = 0.0 * (np.pi / 180)
    ta = 0.0 * (np.pi / 180)
    kep_state = np.array([sma, ecc, inc, raan, aop, ta])
    utc_str = '2025-01-01T00:00:00.000000'

    # start_time = time.perf_counter()
    # ta_entry, ta_exit = eclipse_utils.brute_force_eclipse_angles(kep_state, utc_str)
    # end_time = time.perf_counter()
    # print(f"`brute_force_eclipse_angles` runtime: {end_time-start_time} seconds")

    kep_state= np.array([7.30000000e+03, 1.00000000e-03, 1.74532925e-01, 1.74532925e-01,
       1.74532925e-01, 8.72664626e-02])
    utc_str = "2024-12-16T00:00:00.000000"
    start_time = time.perf_counter()
    ta_entry, ta_exit = eclipse_utils.brute_force_eclipse_angles(kep_state, utc_str)
    end_time = time.perf_counter()
    print(f"`brute_force_eclipse_angles` runtime: {end_time-start_time} seconds")

def test_compute_eclipse_angles():

    #this worked
    # ecc = 0.002
    # sma = (1.029*Constants.R_EARTH)/(1-ecc**2)
    # inc = 63.4*(np.pi/180)# 1.68859134e+00
    # raan = 0.0 * (np.pi / 180)
    # aop = 0.0 * (np.pi / 180)
    # ta = 0.0 * (np.pi / 180)

    inc = 0.1# 1.68859134e00
    kep_state = np.array([ 6.80569765e+03,  1.19063860e-06,  inc,  1.74813883e-02,
       -3.10533920e+00,  3.08785781e+00])
    # utc_str = '2025-01-01T00:00:00.000000'
    utc_str = '2025-01-03T08:46:05.466712'
    # utc_str = '2025-01-18T00:00:00.000000'
    
    start_time = time.perf_counter()
    ta_entry, ta_exit = eclipse_utils.brute_force_eclipse_angles(kep_state, utc_str)
    end_time = time.perf_counter()
    print(f"`brute_force_eclipse_angles` runtime: {end_time-start_time} seconds")

    start_time = time.perf_counter()
    ta2_entry, ta2_exit = eclipse_utils.compute_eclipse_angles(kep_state, utc_str)
    end_time = time.perf_counter()
    print(f"`compute_eclipse_angles` runtime: {end_time-start_time} seconds")

    # Check eclipse-free orbit. #TODO

    if ta_entry is not None and ta_exit is not None:
        assert astro_utils.shortest_angular_dist(ta_entry, ta2_entry) <= 1e-1 and astro_utils.shortest_angular_dist(ta_exit, ta2_exit) <= 1e-1
    else:
        raise ValueError("Entry and exit angles are None.")

def test_sort_eclipse_angles():

    DEG2RAD = np.pi/180
    
    x_en, x_ex = eclipse_utils.sort_eclipse_angles(30*DEG2RAD,60*DEG2RAD)
    assert x_en == 30*DEG2RAD and x_ex == 60*DEG2RAD
    x_en, x_ex = eclipse_utils.sort_eclipse_angles(60*DEG2RAD,30*DEG2RAD)
    assert x_en == 30*DEG2RAD and x_ex == 60*DEG2RAD

    x_en, x_ex = eclipse_utils.sort_eclipse_angles(30*DEG2RAD,90*DEG2RAD)
    assert x_en == 30*DEG2RAD and x_ex == 90*DEG2RAD
    x_en, x_ex = eclipse_utils.sort_eclipse_angles(90*DEG2RAD,30*DEG2RAD)
    assert x_en == 30*DEG2RAD and x_ex == 90*DEG2RAD
    
    x_en, x_ex = eclipse_utils.sort_eclipse_angles(30*DEG2RAD,180*DEG2RAD)
    assert x_en == 30*DEG2RAD and x_ex == 180*DEG2RAD
    x_en, x_ex = eclipse_utils.sort_eclipse_angles(180*DEG2RAD,30*DEG2RAD)
    assert x_en == 30*DEG2RAD and x_ex == 180*DEG2RAD

    x_en, x_ex = eclipse_utils.sort_eclipse_angles(180*DEG2RAD,0*DEG2RAD)
    assert x_en == 0*DEG2RAD and x_ex == 180*DEG2RAD
    x_en, x_ex = eclipse_utils.sort_eclipse_angles(0*DEG2RAD,180*DEG2RAD)
    assert x_en == 0*DEG2RAD and x_ex == 180*DEG2RAD

    x_en, x_ex = eclipse_utils.sort_eclipse_angles(30*DEG2RAD,330*DEG2RAD)
    assert x_en == 330*DEG2RAD and x_ex == 30*DEG2RAD
    x_en, x_ex = eclipse_utils.sort_eclipse_angles(330*DEG2RAD,30*DEG2RAD)
    assert x_en == 330*DEG2RAD and x_ex == 30*DEG2RAD

    x_en, x_ex = eclipse_utils.sort_eclipse_angles(0*DEG2RAD,180*DEG2RAD)
    assert x_en == 0*DEG2RAD and x_ex == 180*DEG2RAD
    x_en, x_ex = eclipse_utils.sort_eclipse_angles(180*DEG2RAD,0*DEG2RAD)
    assert x_en == 0*DEG2RAD and x_ex == 180*DEG2RAD

    x_en, x_ex = eclipse_utils.sort_eclipse_angles(0*DEG2RAD,179.9*DEG2RAD)
    assert x_en == 0*DEG2RAD and x_ex == 179.9*DEG2RAD
    x_en, x_ex = eclipse_utils.sort_eclipse_angles(179.9*DEG2RAD,0*DEG2RAD)
    assert x_en == 0*DEG2RAD and x_ex == 179.9*DEG2RAD

def test_get_earth_sun_vector():

    utc_str = spice.et2utc(0.0, 'ISOC', 6)
    r_earth_sun = eclipse_utils.get_earth_sun_vector(utc_str)

    r_earth_sun = eclipse_utils.get_earth_sun_vector('2000-01-01T00:00:00.000000')


def test_compare_eclipse_angle_functions():

    sma = 7300
    ecc_grid = [0.03, 1e-3, 1e-6]
    inc_grid = [85.0 * (np.pi/180)]
    # ecc_grid = [1e-6, 1e-3, 0.01, 0.1, 0.2]
    
    # inc = astro_utils.compute_sso_inc(sma, ecc)
    raan =0.0 * (np.pi / 180)
    aop = 0.0 * (np.pi / 180)
    ta = 0.0 * (np.pi / 180)

    utc_str_grid = time_utils.generate_time_grid('2024-01-01T00:00:00.000000',
                                                86400*365,
                                                3*86400)
    
    for inc in inc_grid:
        for ecc in ecc_grid:
            for utc_str in utc_str_grid:

                kep_state = np.array([sma, ecc, inc, raan, aop, ta])

                print(f"Evaluating {utc_str}")
                ta_entry, ta_exit = eclipse_utils.brute_force_eclipse_angles(kep_state, utc_str)
                ta2_entry, ta2_exit = eclipse_utils.compute_eclipse_angles(kep_state, utc_str)

                if ta_entry is not None and ta_exit is not None:
                    assert astro_utils.shortest_angular_dist(ta_entry, ta2_entry) <= 1e-1 and astro_utils.shortest_angular_dist(ta_exit, ta2_exit) <= 1e-1
                else:
                    raise ValueError("Entry and exit angles are None.")

# Debugging mode
if __name__ == "__main__":
    # test_check_eclipse()
    # test_brute_force_eclipse_angles()
    # test_compute_eclipse_angles()
    # test_sort_eclipse_angles()
    test_compare_eclipse_angle_functions()
    # test_get_earth_sun_vector()