import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics.Spacecraft import Spacecraft

@dataclass
class DirectSolverSettings:
    spacecraft: Spacecraft
    initial_kep_state: np.ndarray
    initial_mass: float
    initial_utc_str: str
    target_sma: float
    target_ecc: float
    num_costate_nodes: int
    A_mag: float
    sma_tol: float
    ecc_tol: float
    tf_tol: float

    # Safety checks
    def __post_init__(self):

        if self.initial_kep_state[0] < Constants.R_EARTH:
            raise ValueError(f"SMA is lower than Earth radius: {self.initial_kep_state[0]}")

        if self.initial_mass < self.spacecraft.dry_mass:
            raise ValueError(f"Initial mass is less than spacecraft dry mass: {self.initial_mass}")
        
        if self.initial_mass > self.spacecraft.wet_mass:
            raise ValueError(f"Initial mass is more than spacecraft wet mass: {self.initial_mass}")
        
        if self.initial_kep_state[1] < 0:
            raise ValueError(f"Eccentricity is negative: {self.initial_kep_state[1]}")

        if self.initial_kep_state[1] > 1:
            raise ValueError(f"Eccentricity is greater than 1 (only elliptical orbits allowed): {self.initial_kep_state[1]}")

    # Derived properties
    @property
    def initial_equin_state(self) -> np.ndarray:
        initial_E = astro_utils.true2eccentric(self.initial_kep_state[-1], self.initial_kep_state[1])
        return astro_utils.classical_to_equinoctial(
            np.concatenate((self.initial_kep_state[0:5],[initial_E]))
        )
    
    @property
    def sma_grid(self) -> np.ndarray:
        return np.linspace(self.initial_kep_state[0], self.target_sma, self.num_costate_nodes)