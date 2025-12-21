import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics.DirectSolver import DirectSolver
from flight_dynamics.OrbitLogger import OrbitLogger
from flight_dynamics.Spacecraft import Spacecraft

def test_orbit_averaged_propagation():
    id = '01'
    wet_mass = 1000
    dry_mass = 1000
    Isp = 300
    Cd = 1
    A_ref = 1
    spacecraft = Spacecraft(id, wet_mass, dry_mass, Isp, Cd, A_ref)

    sma = Constants.R_EARTH + 500
    ecc = 0.001
    inc = astro_utils.compute_sso_inc(sma, ecc)
    raan = 30 * (np.pi / 180)
    aop = 40 * (np.pi / 180)
    ma = 10 * (np.pi / 180)
    initial_kep_state = np.array([sma, ecc, inc, raan, aop, ma])

    target_sma = sma + 100
    initial_utc_str  = "2025-01-01T00:00:00.000"

    sma_grid = np.linspace(sma, target_sma, 10)
    a_lambda_a_grid = np.linspace(0, 0, 10)
    lambda_e_grid = np.linspace(0, 0, 10)
    lambda_i_grid = np.linspace(0, 0, 10)
    tf = 86400

    solver = DirectSolver(spacecraft)
    solver.orbit_averaged_propagation(initial_kep_state,
                                      wet_mass,
                                      initial_utc_str,
                                      sma_grid,
                                      a_lambda_a_grid,
                                      lambda_e_grid,
                                      lambda_i_grid,
                                      tf)


def test_solve():

    id = '01'
    wet_mass = 1000
    dry_mass = 1000
    Isp = 300
    Cd = 1
    A_ref = 1
    spacecraft = Spacecraft(id, wet_mass, dry_mass, Isp, Cd, A_ref)

    sma = Constants.R_EARTH + 500
    ecc = 0.001
    inc = astro_utils.compute_sso_inc(sma, ecc)
    raan = 30 * (np.pi / 180)
    aop = 40 * (np.pi / 180)
    ma = 10 * (np.pi / 180)
    initial_kep_state = np.array([sma, ecc, inc, raan, aop, ma])

    target_sma = sma + 100
    initial_utc_str  = "2025-01-01T00:00:00.000"

    solver = DirectSolver(spacecraft)
    control_law_handle = solver.generate_control_law(initial_kep_state,
                                                    spacecraft.wet_mass,
                                                    initial_utc_str,
                                                    target_sma,
                                                    3)
    print("done")
    # orbit_logger = OrbitLogger()
    # orbit_logger.plot_results(out_df_list, "hours")

if __name__ == "__main__":
    # test_orbit_averaged_propagation()
    test_solve()