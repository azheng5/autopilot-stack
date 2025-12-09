import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics.IndirectSolver import IndirectSolver

def test_single_rev_program():

    sma = Constants.R_EARTH + 500
    ecc = 0.001
    inc = astro_utils.compute_sso_inc(sma, ecc)
    raan = 30 * (np.pi / 180)
    aop = 40 * (np.pi / 180)
    ma = 10 * (np.pi / 180)
    initial_kep_state = np.array([sma, ecc, inc, raan, aop, ma])

    final_sma = sma + 10

    solver = IndirectSolver(initial_kep_state, final_sma)

    alpha = solver.single_rev_program(np.pi/2)

    plt.plot(np.arange(len(alpha)), alpha)
    plt.show()

if __name__ == "__main__":
    test_single_rev_program()