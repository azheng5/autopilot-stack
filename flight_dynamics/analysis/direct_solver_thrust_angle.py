import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics.solver.DirectSolver import DirectSolver
from flight_dynamics.OrbitLogger import OrbitLogger
from flight_dynamics.Spacecraft import Spacecraft

"""
Building intuition about the relationship and effect of SMA and ECC costates
on thrust angle alpha.

Can we determine a set of costate bounds such that it becomes impossible for 
the thrust angle to effect a negative SMA rate? This makes it much easier for the
solver to converge.
"""

def run_direct_solver_thrust_angle(a_lambda_a):

    id = '01'
    wet_mass = 200
    dry_mass = 0
    Isp = 3000
    Cd = 1
    A_ref = 1
    spacecraft = Spacecraft(id, wet_mass, dry_mass, Isp, Cd, A_ref)

    solver = DirectSolver(spacecraft)

    sma = Constants.R_EARTH + 400
    ecc = 0.001
    inc = 45
    raan = 0 * (np.pi / 180)
    aop = 0 * (np.pi / 180)
    ta = 0 * (np.pi / 180)
    kep_state = np.array([sma, ecc, inc, raan, aop, ta])
    equin_state = astro_utils.classical_to_equinoctial(kep_state)

    r = astro_utils.compute_radius(sma, ecc, 0)
    v = astro_utils.vis_viva(sma, r)

    # Showing that if ecc costate is zero and sma costate is negative, then 
    # thrust angle is zero
    # For the case ofcontinuous thrust and nearly circular orbit, tangent 
    # steering should result in a zero net eccentricity change

    # a_lambda_a = -1
    # a_lambda_a = -7.5
    lambda_a = a_lambda_a / sma
    # lambda_e_grid = [-1, -0.1, 0, 0.5, 1, 1.06, 2, 1e6]
    lambda_e_grid = [0, 0.5, 1, 2, 3, 5, 10, 20]
    # lambda_e_grid = [9.0]
    nu_grid = np.linspace(-np.pi,np.pi,100)
    # nu_grid = np.linspace(0,2*np.pi,100)

    alpha_grid = np.zeros((100,len(lambda_e_grid)))
    for i in range(len(lambda_e_grid)):
        for j in range(len(nu_grid)):
            alpha = solver.compute_thrust_angle(sma, ecc, 
                                                nu_grid[j], lambda_a, lambda_e_grid[i])
            alpha_grid[j,i]=alpha
        

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    nu_grid_deg = nu_grid *(180/np.pi)
    ax.plot(nu_grid_deg, alpha_grid[:,0]*(180/np.pi), label="lambda_e=-0")
    ax.plot(nu_grid_deg, alpha_grid[:,1]*(180/np.pi), label="lambda_e=0.5")
    ax.plot(nu_grid_deg, alpha_grid[:,2]*(180/np.pi), label="lambda_e=1")
    ax.plot(nu_grid_deg, alpha_grid[:,3]*(180/np.pi), label="lambda_e=2")
    ax.plot(nu_grid_deg, alpha_grid[:,4]*(180/np.pi), label="lambda_e=3")
    ax.plot(nu_grid_deg, alpha_grid[:,5]*(180/np.pi), label="lambda_e=5")
    ax.plot(nu_grid_deg, alpha_grid[:,6]*(180/np.pi), label="lambda_e=10")
    ax.plot(nu_grid_deg, alpha_grid[:,7]*(180/np.pi), label="lambda_e=20")
    ax.set_xlabel("nu (deg)")
    ax.set_ylabel("Thrust Angle (deg)")
    plt.title(f"a_lambda_a={a_lambda_a}")

    plt.legend()

    plt.savefig(f"flight_dynamics/analysis/data/direct_solver/ds_plot_{a_lambda_a}.png", dpi=300)
    # plt.show()


    # print("fin")

if __name__ == "__main__":
    run_direct_solver_thrust_angle(-10)
    run_direct_solver_thrust_angle(-7.5)
    run_direct_solver_thrust_angle(-5)
    run_direct_solver_thrust_angle(-2.5)
    run_direct_solver_thrust_angle(-1)