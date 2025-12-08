import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
import flight_dynamics.astro_utils as astro_utils
from flight_dynamics import Constants
from flight_dynamics.Propagator import Propagator
from flight_dynamics.Time import Time

def test_propagate():

    sma = Constants.R_EARTH + 500
    ecc = 0.001
    inc = astro_utils.compute_sso_inc(sma, ecc)
    raan = 30 * (np.pi / 180)
    aop = 40 * (np.pi / 180)
    ma = 10 * (np.pi / 180)

    initial_mass = 1000
    initial_kep_state = np.array([sma, ecc, inc, raan, aop, ma])
    Isp = 300
    t_final = 1*24*60*60
    initial_time = Time(utc_string="2025-01-01T00:00:00Z")
    delta_t = 10
    out_file_name = "test_output.csv"
    out_columns = ["t","m","rx","ry","rz","vx","vy","vz","sma","ecc","inc","raan","aop","ma"]
    plot_timescale = "days"

    propagator = Propagator(initial_time, initial_kep_state, initial_mass, Isp, t_final, delta_t, out_file_name, out_columns, plot_timescale)


# Debugging mode
if __name__ == "__main__":
    test_propagate()