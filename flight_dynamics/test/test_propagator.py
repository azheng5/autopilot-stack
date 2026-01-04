import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics import time_utils
from flight_dynamics.OrbitLogger import OrbitLogger
from flight_dynamics.Propagator import Propagator, PropagatorTerminator
from flight_dynamics.Spacecraft import Spacecraft

def test_propagate():

    sma = Constants.R_EARTH + 300
    ecc = 0.01
    inc = 45*(np.pi/180)
    raan = 0.0
    aop = 0.0
    ma = 0.0

    id = '01'
    wet_mass = 1000
    dry_mass = 50
    Isp = 3000
    Cd = 2.2
    A_ref = 3e-6
    spacecraft = Spacecraft(id, wet_mass, dry_mass, Isp, Cd, A_ref)

    initial_kep_state= np.array([sma, ecc, inc, raan, aop, ma])
    raan_dot, aop_dot = astro_utils.compute_j2_drift_effect(sma, ecc, inc)
    utc_str = "2024-12-16T00:00:00.000000"
    # utc_str = "2025-01-01T00:00:00.000000"
    initial_mass = wet_mass
    time_grid = time_utils.generate_time_grid(utc_str, 30*86400, 30)
    out_file_name = "test_output.csv"

    propagator = Propagator(spacecraft)
    logged_df, _ = propagator.propagate(initial_kep_state, 
                                    initial_mass,
                                    time_grid,
                                    terminators=[PropagatorTerminator.ATMOS_ENTRY],
                                    show_progress_bar=False)
    
    orbit_logger = OrbitLogger()
    orbit_logger.save_to_csv([logged_df], out_file_name)
    orbit_logger.plot_results([logged_df], "hours")

# Debugging mode
if __name__ == "__main__":
    test_propagate()