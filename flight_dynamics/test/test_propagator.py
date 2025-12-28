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

    sma = Constants.R_EARTH + 500
    ecc = 0.001
    inc = astro_utils.compute_sso_inc(sma, ecc)
    raan = 30 * (np.pi / 180)
    aop = 40 * (np.pi / 180)
    ma = 10 * (np.pi / 180)

    id = '01'
    wet_mass = 1000
    dry_mass = 1000
    Isp = 3000
    Cd = 1
    A_ref = 1
    spacecraft = Spacecraft(id, wet_mass, dry_mass, Isp, Cd, A_ref)

    initial_kep_state = np.array([sma, ecc, inc, raan, aop, ma])
    initial_mass = wet_mass
    time_grid = time_utils.generate_time_grid("2025-01-01T00:00:00.000", 86400/2, 10)
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