import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics.ExpAtmosphereModel import ExpAtmosphereModel
from flight_dynamics.Spacecraft import Spacecraft

def test_sample_exp_atmosphere_table():
    id = '01'
    wet_mass = 200
    dry_mass = 50
    Isp = 3000
    Cd = 2.2
    A_ref = 3
    spacecraft=Spacecraft(id, wet_mass, dry_mass, Isp, Cd, A_ref)

    atm_model = ExpAtmosphereModel(spacecraft)
    rho0, h0, H = atm_model.sample_exp_atmosphere_table(310)
    assert rho0 == 2.418E-11 and h0 == 300 and H == 53.628

if __name__ == "__main__":
    test_sample_exp_atmosphere_table()