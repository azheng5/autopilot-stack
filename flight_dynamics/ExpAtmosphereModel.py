import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics.Spacecraft import Spacecraft

class ExpAtmosphereModel:
    """
    Class for computing atmospheric drag on LEO satellites

    References:
        - Vallado, Fundamentals of Astrodynamics and Applications, Ch 8.6.2
    """
    
    def __init__(self, spacecraft: Spacecraft) -> None:
        self.spacecraft = spacecraft

        self.exp_atmosphere_table = pd.read_csv("flight_dynamics/exp_atmosphere_table.csv")
        self.exp_atmosphere_table["rho0"] *= 1e9

    def compute_specific_drag_force(self,
                                    cart_state: np.ndarray,
                                    curr_mass: float) -> np.ndarray:

        rx, ry, rz, vx, vy, vz = cart_state
        r_norm = np.linalg.norm(cart_state[0:3])

        # Velocity relative to atmosphere, neglect wind effects
        v_rel = np.array([
            vx + ry*Constants.EARTH_ROT_RATE,
            vy - rx*Constants.EARTH_ROT_RATE,
            vz
        ])
        v_rel_norm = np.linalg.norm(v_rel)

        # Get current atmospheric density
        curr_rho = self.density_exponential_model(r_norm)

        # Compute ballistic coefficient
        curr_bc = curr_mass/(self.spacecraft.Cd*self.spacecraft.A_ref)

        return -0.5 * (1/curr_bc) * curr_rho * (v_rel_norm**2) * (v_rel/v_rel_norm)

    def density_exponential_model(self, r: float) -> float:
        """
        Get atmospheric density rho from static expoential model, 
        valid for 0-1000 km.

        Valid for rough simulation of drag effects for high level 
        design studies, but not for high accurate studies.
        """

        # Actual altitude above the ellipsoid
        h_ellp = r - Constants.R_EARTH

        # Divide atmosphere into altitude bands, each with its own reference values
        rho0, h0, H = self.sample_exp_atmosphere_table(h_ellp)

        return rho0 * np.exp(-(h_ellp - h0)/H)
    
    def sample_exp_atmosphere_table(self, h: float) -> Tuple[float, float, float]:

        if h > 1000:
            raise ValueError("Altitude cannot exceed 1000 km.")
        
        # Find the row where h falls within the [start, end) interval
        condition = (self.exp_atmosphere_table['h0'] <= h) & (h < self.exp_atmosphere_table['hf'])
        row = self.exp_atmosphere_table[condition]

        # Extract values as floats
        rho0 = float(row['rho0'].iloc[0])
        h0 = float(row['h0'].iloc[0])
        H = float(row['H'].iloc[0])
        return rho0, h0, H