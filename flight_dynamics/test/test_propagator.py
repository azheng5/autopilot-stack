import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics import time_utils
from flight_dynamics.OrbitLogger import OrbitLogger
from flight_dynamics.Propagator import Propagator
from flight_dynamics.Spacecraft import Spacecraft

def test_propagate():

    sma = 7.30000000e+03
    ecc = 1.00000000e-03
    inc = 10*(np.pi/180)
    raan = 1.74532925e-01
    aop = 1.74532925e-01
    ma = 45*(np.pi/180)

    id = '01'
    wet_mass = 1000
    dry_mass = 1000
    Isp = 3000
    Cd = 1
    A_ref = 1
    spacecraft = Spacecraft(id, wet_mass, dry_mass, Isp, Cd, A_ref)

    initial_kep_state= np.array([sma, ecc, ecc, raan, aop, ma])
    raan_dot, aop_dot = astro_utils.compute_j2_drift_effect(sma, ecc, inc)
    utc_str = "2024-12-16T00:00:00.000000"
    # utc_str = "2025-01-01T00:00:00.000000"
    initial_mass = wet_mass
    time_grid = time_utils.generate_time_grid(utc_str, 86400/2, 10)
    out_file_name = "test_output.csv"

    propagator = Propagator(spacecraft)
    logged_df, _ = propagator.propagate(initial_kep_state, 
                                    initial_mass,
                                    time_grid,
                                    phase_number=2)
    
    orbit_logger = OrbitLogger()
    orbit_logger.save_to_csv([logged_df], out_file_name)
    orbit_logger.plot_results([logged_df], "hours")

# Debugging mode
if __name__ == "__main__":
    test_propagate()