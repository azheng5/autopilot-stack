import sys
from enum import Enum, auto
from pathlib import Path
from tqdm import tqdm
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spiceypy as spice

sys.path.append(str(Path(__file__).parent.parent))
import flight_dynamics.astro_utils as astro_utils
from flight_dynamics import Constants
from flight_dynamics.Eclipse import Eclipse
from flight_dynamics.OrbitLogger import OrbitLogger
from flight_dynamics.Spacecraft import Spacecraft

spice.furnsh("meta_kernel.tm")

class PropagatorTerminator(Enum):
    """Termination conditions for Propagator class"""
    MAX_PROP_TIME = auto()
    PHASE_COUNT_LIMIT = auto()
    ECLIPSE_FALLING_EDGE = auto()
    ECLIPSE_RISING_EDGE = auto()
    NONE = auto()

class Propagator:
    """
    Propagates 3DOF spacecraft trajectories.

    #TODO: add validators (conservation of angmom, orbit error, covariances)
    #TODO add covariance propagation
    """

    def __init__(self,
                 spacecraft: Spacecraft) -> None:
        self.spacecraft = spacecraft

    def propagate(self,
                  initial_kep_state: np.ndarray,
                  initial_mass: float,
                  time_grid: list,
                  phase_number: int|None = None,
                  thrust_profile: np.ndarray|None = None,
                  thrust_frame: str|None = None,
                  terminators: list = [],
                  show_progress_bar: bool = True) -> tuple[pd.DataFrame, PropagatorTerminator]:
        """
        Main propagation tool

        Arguments:
            create_plots: Whether or not to create plots of the results
        """


        # Set up propagator
        orbit_logger = OrbitLogger()
        prop_time = spice.utc2et(time_grid[-1]) - spice.utc2et(time_grid[0])
        initial_et = spice.utc2et(time_grid[0])
        initial_cart_state = astro_utils.kep2cart(initial_kep_state, initial_et)
        phase_counter = 0 # number of phasing orbits that have been completed
        prev_eclipse_status = None # invalid status 
        prev_cart_state = None
        prev_kep_state = None
        prev_phase_increment_et = initial_et
        x_curr = np.concatenate(([initial_mass], initial_cart_state)) #["m","rx","ry","rz","vx","vy","vz"]
        t_curr = 0.0 # seconds since start time
        if show_progress_bar:
            progress_bar = tqdm(total=100, desc="Propagating", unit="%")
            curr_progress = 0.0
        initial_kep_state = astro_utils.cart2kep(initial_cart_state, initial_et)
        termination_cause = PropagatorTerminator.NONE

        # Set up data to be logged during propagation process. Additional data
        # may be computed during post-processing
        logged_data = []

        # Main propagation loop
        for step_counter in range(len(time_grid)):

            # Compute current sim state
            curr_et = initial_et + t_curr
            curr_utc_str = spice.et2utc(curr_et, 'ISOC', 6)
            curr_cart_state = x_curr[1:7]
            curr_cart_pos = x_curr[1:4]
            curr_cart_vel = x_curr[4:7]
            curr_ang_mom = np.cross(curr_cart_pos, curr_cart_vel)
            curr_h_norm = np.linalg.norm(curr_ang_mom)
            curr_kep_state = astro_utils.cart2kep(curr_cart_state, curr_et)
            eclipse_status = Eclipse.check_eclipse(curr_cart_state[0:3], curr_utc_str)
            curr_orbit_period = astro_utils.get_orbit_period(curr_kep_state[0])

            # Log data in current iteration
            #TODO add "log data entry struct" to prevent errors
            out_entry = [t_curr,
                         *x_curr, 
                         *curr_kep_state,
                         eclipse_status,
                         *curr_ang_mom,
                         curr_h_norm,
                         curr_orbit_period,
                         curr_utc_str,
                         curr_et]
            logged_data.append(out_entry)

            # TODO: need to abstract event detection functions (eclipse, phasing loop, hitting periapsis/apoapsis)
            # Termination logic
            # We want to log the final state, but terminate if we are at the 
            # final time, so logging goes first. This way we can also see the state
            # that triggered a termination

            # Propagation final time reached
            if curr_et >= spice.utc2et(time_grid[-1]):
                termination_cause = PropagatorTerminator.MAX_PROP_TIME
                break

            # Detect eclipse falling and rising edges
            if prev_eclipse_status is not None:
                # Falling edge
                if (PropagatorTerminator.ECLIPSE_FALLING_EDGE in terminators) and eclipse_status == 0 and prev_eclipse_status > 0:
                    termination_cause = PropagatorTerminator.ECLIPSE_FALLING_EDGE
                    break
                # Rising edge
                if (PropagatorTerminator.ECLIPSE_RISING_EDGE in terminators) and eclipse_status > 0 and prev_eclipse_status == 0:
                    termination_cause = PropagatorTerminator.ECLIPSE_RISING_EDGE
                    break

            # Detect phasing loop increment
            # This is at least a several second error
            if curr_et - prev_phase_increment_et >= curr_orbit_period:
                prev_phase_increment_et = curr_et
                phase_counter += 1


            # Its wrong to detect phasing loop increments using osculating mean anomaly....
            # if (prev_kep_state is not None) and (curr_et - prev_phase_increment_et >= curr_orbit_period/2):

            #     # Handle edge case when initial mean anomaly at 2pi border
            #     if prev_kep_state[5] <= initial_mean_anomaly and curr_kep_state[5] <= prev_kep_state[5]:
            #         prev_phase_increment_et = curr_et
            #         phase_counter += 1

            #     # Handle edge case when initial mean anomaly at 0 border
            #     if prev_kep_state[5] >= initial_mean_anomaly and curr_kep_state[5] <= :
            #         prev_phase_increment_et = curr_et
            #         phase_counter += 1

            if phase_counter == phase_number:
                termination_cause = PropagatorTerminator.PHASE_COUNT_LIMIT
                break

            # Sample control input
            if thrust_profile is None:
                u_curr = np.array([0,0,0])
            else:
                if eclipse_status > 0:
                    raise ValueError("Attempting to apply thrust during eclipse")
                
                #TODO add linear interpolation?
                if thrust_frame == "ECI":
                    u_curr = thrust_profile[step_counter,:].reshape(3,1)          
                elif thrust_frame == "LVLH":
                    R_ECI_LVLH = astro_utils.lvlh_to_eci_matrix(curr_cart_pos,
                                                                curr_cart_vel)
                    T_LVLH = thrust_profile[step_counter,:].reshape(3,1)
                    u_curr = R_ECI_LVLH @ T_LVLH
                else:
                    raise ValueError("Invalid thrust frame")

            # Compute RK4 step
            delta_t = spice.utc2et(time_grid[step_counter+1]) - spice.utc2et(time_grid[step_counter])
            x_next = self.rk4_step(self.eom, t_curr, x_curr, u_curr, delta_t)

            # Update progress bar
            if show_progress_bar:
                sim_progress = (t_curr / prop_time) * 100
                delta_progress = sim_progress - curr_progress
                progress_bar.update(delta_progress)

            # Prepare for next iteration (k -> k+1)
            x_curr = x_next
            t_curr = t_curr + delta_t
            if show_progress_bar:
                curr_progress = sim_progress
            prev_eclipse_status = eclipse_status
            prev_cart_state = curr_cart_state
            prev_kep_state = curr_kep_state

        # Process output
        # print("Propagation complete.")
        if show_progress_bar:
            progress_bar.close()

        logged_df = pd.DataFrame(logged_data, columns=orbit_logger.logged_columns)

        if termination_cause == PropagatorTerminator.NONE:
            raise ValueError("No termination condition occurred.")

        # print(f"Process ended due to {termination_cause} event.")

        return logged_df, termination_cause

    def eom(self,
            t: float,
            x: np.ndarray,
            u: np.ndarray) -> np.ndarray:
        
        # Extract state vector
        m, rx, ry, rz, vx, vy, vz = x
        r = np.array([rx, ry, rz])
        v = np.array([vx, vy, vz])
        r_norm = np.linalg.norm(r)

        # Extract control vector
        T_mag = np.linalg.norm(u)

        # Compute 2-body gravity acceleration
        a_g = -(Constants.EARTH_MU / r_norm**3) * r

        # Compute J2 perturbing acceleration
        j2_unit_vec = np.array([(1 - (5*(rz/r_norm)**2)) * (rx/r_norm),
                                (1 - (5*(rz/r_norm)**2)) * (ry/r_norm),
                                (3 - (5*(rz/r_norm)**2)) * (rz/r_norm) ])
        a_j2 = -1.5 * Constants.J2 * (Constants.EARTH_MU / r_norm**2) * (Constants.R_EARTH/r_norm)**2 * j2_unit_vec
        # a_j2 = np.array([0,0,0])

        # Compute atmospheric drag perturbing acceleration
        a_drag = np.array([0,0,0])

        # Compute thrust acceleration
        a_thrust = (u / m).ravel()

        # Compute mass derivative
        m_dot = -T_mag / (self.spacecraft.Isp * Constants.G0)

        a_tot = a_g + a_j2 + a_drag + a_thrust
        x_dot = np.array([m_dot, 
                          vx, vy, vz,
                          a_tot[0], a_tot[1], a_tot[2]])
        
        return x_dot

    def rk4_step(self,fn: Callable[[float, np.ndarray, np.ndarray],np.ndarray],
                t: float,
                x: np.ndarray,
                u: np.ndarray,
                delta_t: float) -> np.ndarray:
        
        k1 = fn(t,x,u)
        k2 = fn(t + delta_t/2, x + (delta_t/2)*k1, u)
        k3 = fn(t + delta_t/2, x + (delta_t/2)*k2, u)
        k4 = fn(t + delta_t, x + delta_t*k3, u)

        return x + (delta_t/6) * (k1 + 2*k2 + 2*k3 + k4)