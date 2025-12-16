import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics.IndirectSolver import IndirectSolver
from flight_dynamics.OrbitLogger import OrbitLogger
from flight_dynamics.Spacecraft import Spacecraft

def test_single_rev_program():

    id = '01'
    wet_mass = 1000
    dry_mass = 1000
    Isp = 300
    Cd = 1
    A_ref = 1
    spacecraft = Spacecraft(id, wet_mass, dry_mass, Isp, Cd, A_ref)

    solver = IndirectSolver(spacecraft)
    _, alpha = solver.single_rev_program(np.pi/2)

    plt.plot(np.arange(len(alpha)), alpha)
    plt.show()

def test_raise_circular_orbit():

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

    solver = IndirectSolver(spacecraft)
    orbit_logger = OrbitLogger()
    out_df_list = solver.raise_circular_orbit(initial_kep_state,
                                spacecraft.wet_mass, 
                                target_sma,
                                initial_utc_str)
    orbit_logger.plot_results(out_df_list, "hours")

if __name__ == "__main__":
    # test_single_rev_program()
    test_raise_circular_orbit()