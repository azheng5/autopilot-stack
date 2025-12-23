import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics.DirectSolver import DirectSolver
from flight_dynamics.OldDirectSolver import OldDirectSolver
from flight_dynamics.OrbitLogger import OrbitLogger
from flight_dynamics.Spacecraft import Spacecraft

#TODO test harness
id = '01'
wet_mass = 200
dry_mass = 200
Isp = 300
Cd = 1
A_ref = 1
spacecraft = Spacecraft(id, wet_mass, dry_mass, Isp, Cd, A_ref)

def test_orbit_averaged_propagation():

    sma = Constants.R_EARTH + 500
    ecc = 0.1
    inc = astro_utils.compute_sso_inc(sma, ecc)
    raan = 30 * (np.pi / 180)
    aop = 40 * (np.pi / 180)
    ma = 10 * (np.pi / 180)
    initial_kep_state = np.array([sma, ecc, inc, raan, aop, ma])

    target_sma = sma + 10
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

    sma = 7000
    ecc = 0.1
    inc = astro_utils.compute_sso_inc(sma, ecc)
    raan = 30 * (np.pi / 180)
    aop = 40 * (np.pi / 180)
    ma = 10 * (np.pi / 180)
    initial_kep_state = np.array([sma, ecc, inc, raan, aop, ma])

    target_sma = sma + 50
    target_ecc = 0.0
    initial_utc_str  = "2025-01-01T00:00:00.000"

    A_mag = 2.943e-4 * 1e-3

    num_costate_nodes = 3

    solver = DirectSolver(spacecraft)
    # control_law_handle = solver.generate_control_law(initial_kep_state,
    #                                                 spacecraft.wet_mass,
    #                                                 initial_utc_str,
    #                                                 target_sma,
    #                                                 num_costate_nodes)
    control_law_handle = solver.generate_control_law(initial_kep_state,
                                                    spacecraft.wet_mass,
                                                    initial_utc_str,
                                                    target_sma,
                                                    target_ecc,
                                                    num_costate_nodes,
                                                    A_mag)
    print("done")
    # orbit_logger = OrbitLogger()
    # orbit_logger.plot_results(out_df_list, "hours")

def test_slow_equinoctial_diff_eq():

    solver = DirectSolver(spacecraft)

    sma = 7000
    ecc = 0.01
    inc = astro_utils.compute_sso_inc(sma, ecc)
    raan = 30 * (np.pi / 180)
    aop = 40 * (np.pi / 180)
    E = 0.1 * (np.pi / 180)
    kep_state = np.array([sma, ecc, inc, raan, aop, E])
    equin_state = astro_utils.classical_to_equinoctial(kep_state)

    # Validate zero acceleration leads to zero rate of change
    slow_equin_state_dot = solver.slow_equinoctial_diff_eq(equin_state, 0.0, 0.0)
    assert np.array_equal(slow_equin_state_dot, np.array([0,0,0,0,0]))

    # Compare to slow keplerian diff eq
    slow_equin_state_dot = solver.slow_equinoctial_diff_eq(equin_state, 0.001, 0.0)
    slow_kep_state_dot = solver.slow_kep_diff_eq(kep_state, 0.001, 0.0)
    dt = 0.1
    next_slow_equin_state = equin_state[0:5] + dt * slow_equin_state_dot
    next_slow_kep_state = kep_state[0:5] + dt * slow_kep_state_dot

    next_kep_state = np.concatenate((next_slow_kep_state,[0.0]))
    next_equin_state = np.concatenate((next_slow_equin_state,[0.0]))
    res_next_slow_equin_state = astro_utils.classical_to_equinoctial(next_kep_state)
    res_next_slow_kep_state = astro_utils.equinoctial_to_classical(next_equin_state)

    # NOTE: If SMA=7000->T=5828, need resolution of 360/(Hz*T) degrees to maintain desired 
    # error (desired error stems from the decided dt)
    # 0.1 Hz: worst case e-06 error -> 0.62 deg res -> linspace 580
    # 1 Hz: worst case e-08 error -> 0.06 degree res ->linspace 6000
    # 10Hz: worst case e-11 error -> 0.006 deg res -> linsapce 60000

    # The errors due to numerical roundoffs, which get propagated forward to a larger 
    # degree the larger the timestep.
    assert np.allclose(next_slow_equin_state, res_next_slow_equin_state[0:5], atol=1e-12)
    assert np.allclose(next_slow_kep_state, res_next_slow_kep_state[0:5], atol=1e-12)

# Debugging mode
if __name__ == "__main__":
    # test_orbit_averaged_propagation()
    test_solve()
    # test_slow_equinoctial_diff_eq()