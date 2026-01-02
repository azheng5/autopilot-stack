import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spiceypy as spice
from scipy.optimize import minimize
from typing import Tuple

sys.path.append(str(Path(__file__).parent.parent.parent))
import flight_dynamics.timestepper_utils as timestepper_utils
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics import eclipse_utils
from flight_dynamics import time_utils
from flight_dynamics.Propagator import Propagator, PropagatorTerminator
from flight_dynamics.solver.DirectSolverSettings import DirectSolverSettings
from flight_dynamics.solver.DirectSolverLogger import DirectSolverLogEntry, DirectSolverLogger, DirectSolverResult, MeanPropLogEntry
from flight_dynamics.Spacecraft import Spacecraft

class ClosedLoopSolver:
    """
    Closed loop low thrust guidance law by resolving the fuel-optimal
    low-thrust trajectory throughout overall transfer lifespan.
    
    Given an initial injection orbit, and target semi-major axis, 
    eccentricity, and RAAN, this tool solves and simulates 
    a near-optimal feedback trajectory. 
    
    The SMA and ECC targets are posed as terminal constraints in 
    the `DirectSolver`, and the RAAN target is achieved by 
    computing a fuel-optimal intermediate orbit and station time.
    """
    def __init__(self, 
                 cfg: ClosedLoopSolverSettings) -> None:
        self.cfg = cfg
        self.cls_logger = ClosedLoopSolverLogger()
        #TODO