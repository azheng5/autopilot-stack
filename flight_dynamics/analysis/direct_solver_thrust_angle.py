import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics.direct_solver.DirectSolver import DirectSolver
from flight_dynamics.OldDirectSolver import OldDirectSolver
from flight_dynamics.OrbitLogger import OrbitLogger
from flight_dynamics.Spacecraft import Spacecraft

"""
Building intuition about the relationship and effect of SMA and ECC costates
on thrust angle alpha.

Can we determine a set of costate bounds such that it becomes impossible for 
the thrust angle to effect a negative SMA rate? This makes it much easier for the
solver to converge.
"""

id = '01'
wet_mass = 200
dry_mass = 0
Isp = 3000
Cd = 1
A_ref = 1
spacecraft = Spacecraft(id, wet_mass, dry_mass, Isp, Cd, A_ref)

solver = DirectSolver(spacecraft)

sma = 7378
ecc = 0.05
inc = astro_utils.compute_sso_inc(sma, ecc)
raan = 30 * (np.pi / 180)
aop = 40 * (np.pi / 180)
ta = 0.1 * (np.pi / 180)
kep_state = np.array([sma, ecc, inc, raan, aop, ta])
equin_state = astro_utils.classical_to_equinoctial(kep_state)

r = astro_utils.compute_radius(sma, ecc, 0)
v = astro_utils.vis_viva(sma, r)

# Showing that if ecc costate is zero and sma costate is negative, then 
# thrust angle is zero
# For the case ofcontinuous thrust and nearly circular orbit, tangent 
# steering should result in a zero net eccentricity change

a_lambda_a = -1
lambda_a = a_lambda_a / sma
# lambda_e_grid = [-1, -0.1, 0, 0.5, 1, 1.06, 2, 1e6]
lambda_e_grid = [0, 0.5, 1, 1.06, 2, 1e6]
nu_grid = np.linspace(-np.pi,np.pi,100)

alpha_grid = np.zeros((100,len(lambda_e_grid)))
for i in range(len(lambda_e_grid)):
    for j in range(len(nu_grid)):
        alpha = solver.compute_thrust_angle(sma, ecc, 
                                            nu_grid[j], lambda_a, lambda_e_grid[i])
        alpha_grid[j,i]=alpha
    

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
nu_grid_deg = nu_grid *(180/np.pi)
# ax.plot(nu_grid_deg, alpha_grid[:,0]*(180/np.pi), label="lambda_e=-1")
# ax.plot(nu_grid_deg, alpha_grid[:,1]*(180/np.pi), label="lambda_e=-0.1")
ax.plot(nu_grid_deg, alpha_grid[:,0]*(180/np.pi), label="lambda_e=-0")
ax.plot(nu_grid_deg, alpha_grid[:,1]*(180/np.pi), label="lambda_e=0.5")
ax.plot(nu_grid_deg, alpha_grid[:,2]*(180/np.pi), label="lambda_e=1")
ax.plot(nu_grid_deg, alpha_grid[:,3]*(180/np.pi), label="lambda_e=1.06")
ax.plot(nu_grid_deg, alpha_grid[:,4]*(180/np.pi), label="lambda_e=2")
ax.plot(nu_grid_deg, alpha_grid[:,5]*(180/np.pi), label="lambda_e=1e6")
ax.set_xlabel("nu (deg)")
ax.set_ylabel("Thrust Angle (deg)")
plt.legend()
plt.show()

print("fin")