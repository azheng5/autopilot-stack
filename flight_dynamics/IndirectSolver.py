import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spiceypy as spice
from scipy.optimize import newton
from scipy.integrate import quad

sys.path.append(str(Path(__file__).parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics import time_utils
from flight_dynamics.Eclipse import Eclipse
from flight_dynamics.Propagator import Propagator, PropagatorTerminator
from flight_dynamics.OrbitLogger import OrbitLogger
from flight_dynamics.Spacecraft import Spacecraft

class IndirectSolver:
    """
    Indirect thrust solver for coplanar circle-to-circle low thrust transfers.

    Takes an initial Cartesian state and a target semi-major axis, and generates
    an optimal thrust profile to achieve the transfer.

    Assumes 2-body point mass dynamics, so only used for preliminary studies.

    References:
        - Colasurdo and Casalino, https://arc.aiaa.org/doi/epdf/10.2514/6.2004-5087
    
    """

    def __init__(self, spacecraft: Spacecraft) -> None:
        # Configuration settings for indirect solver here...
        self.spacecraft = spacecraft

    def raise_circular_orbit(self, 
                             initial_kep_state: np.ndarray, 
                             initial_mass: float,
                             target_sma: float,
                             initial_utc_str: str,
                             A_mag: float = 0.1) -> list[pd.DataFrame]:
        """
        Given an initial state, target SMA, and fixed thrust magnitude,
        solves for a trajectory and thrust profile to hit the target SMA

        Arguments:
            - initial_kep_state: Initial keplerian state
            - initial_mass: Initial mass
            - target_sma: Target SMA
            - initial_utc_str: Initial UTC string
            - A_mag: Constant thrust acceleration (N)

        """
        
        # if A_mag > 0.01:
            # raise ValueError("Solver does not work for thrust accelerations greater than 0.01")

        # Set up solver
        propagator = Propagator(self.spacecraft)

        # This is not wasteful as it can be planned as the final orbit segment of LEOPs
        rising_edge_kep_state, rising_edge_mass, rising_edge_utc_str, _, rising_edge_df = self.propagate_indirect_segment(propagator,
                                                                                                        initial_kep_state,
                                                                                                        initial_mass,
                                                                                                        initial_utc_str,
                                                                                                        PropagatorTerminator.ECLIPSE_RISING_EDGE)
        out_df_list = [rising_edge_df]



        # Iterate through single revolution profiles until target SMA is hit
        while rising_edge_kep_state[0] < target_sma:

            # Freely propagate until eclipse falling edge or 1 rev to 
            # determine orbit period and shadow angle
            falling_edge_kep_state, falling_edge_mass, falling_edge_utc_str, termination_cause, falling_edge_df = self.propagate_indirect_segment(propagator,
                                                                                                                            rising_edge_kep_state,
                                                                                                                            rising_edge_mass,
                                                                                                                            rising_edge_utc_str,
                                                                                                                            PropagatorTerminator.ECLIPSE_FALLING_EDGE)
            falling_edge_et = spice.utc2et(falling_edge_utc_str)
            out_df_list.append(falling_edge_df)

            ## Compute shadow angle for current revolution
            # NOTE: This isn't the true shadow angle of the orbit because the orbit itself changes 
            # during a low thrust arc. The approximation here is that the next eclipse entry point for the 
            # sc under low thrust is close the entry point if no thrust were applied under a single revolution
            if termination_cause == PropagatorTerminator.PHASE_COUNT_LIMIT:
                # One full rev was completed without entering eclipse, so current revolution doesn't require shadow arc
                shadow_angle = 0
            elif termination_cause == PropagatorTerminator.ECLIPSE_FALLING_EDGE:
                # Compute shadow angle using mean anomaly differences
                shadow_angle = (falling_edge_kep_state[5] - rising_edge_kep_state[5]) % (2*np.pi)
            else:
                raise ValueError("Invalid termination condition occurred.")

            # Compute thrust profile for current rev (assume outputted true anomaly grid = mean anomaly grid)
            delta_ma_grid, alpha = self.single_rev_program(shadow_angle)

            # Convert mean anomaly grid to ephemeris time grid
            thrust_arc_time_grid = []
            for delta_ma in delta_ma_grid:
                delta_t = astro_utils.mean_anomaly_to_time(falling_edge_kep_state[5]+delta_ma, 
                                                           falling_edge_kep_state[5], 
                                                           falling_edge_kep_state[0])
                thrust_arc_time_grid.append(spice.et2utc(falling_edge_et + delta_t, 'ISOC', 6))

            # Compute 3d thrust components
            A_mag_kN = A_mag * 1e-3
            T_LVLH_x = A_mag_kN*np.sin(alpha)
            T_LVLH_y = A_mag_kN*np.cos(alpha)
            T_LVLH_z = np.zeros(len(T_LVLH_x))
            T_LVLH = np.column_stack((T_LVLH_x, T_LVLH_y, T_LVLH_z))

            # Propagate thrust profile over current revolution until rising edge
            inter_df, termination_cause = propagator.propagate(falling_edge_kep_state,
                                                            falling_edge_mass,
                                                            thrust_arc_time_grid,
                                                            thrust_profile=T_LVLH,
                                                            thrust_frame="LVLH",
                                                            terminators=[PropagatorTerminator.ECLIPSE_RISING_EDGE],
                                                            show_progress_bar=False)
            out_df_list.append(inter_df)
            if termination_cause == PropagatorTerminator.MAX_PROP_TIME:
                rising_edge_kep_state, rising_edge_mass, rising_edge_utc_str, termination_cause, rising_edge_df = self.propagate_indirect_segment(propagator,
                                                                                                                                np.array([inter_df.iloc[-1]["sma"],
                                                                                                                                        inter_df.iloc[-1]["ecc"],
                                                                                                                                        inter_df.iloc[-1]["inc"],
                                                                                                                                        inter_df.iloc[-1]["raan"],
                                                                                                                                        inter_df.iloc[-1]["aop"],
                                                                                                                                        inter_df.iloc[-1]["ma"]]),
                                                                                                                                inter_df.iloc[-1]["m"],
                                                                                                                                inter_df.iloc[-1]["utc_str"],
                                                                                                                                PropagatorTerminator.ECLIPSE_RISING_EDGE)
                out_df_list.append(rising_edge_df)
                # if termination_cause != PropagatorTerminator.ECLIPSE_RISING_EDGE:
                #     raise ValueError("Could not reach eclipse rising edge from applied thrust profile.")
            elif termination_cause == PropagatorTerminator.ECLIPSE_RISING_EDGE:
                rising_edge_kep_state = np.array([inter_df.iloc[-1]["sma"],
                                                inter_df.iloc[-1]["ecc"],
                                                inter_df.iloc[-1]["inc"],
                                                inter_df.iloc[-1]["raan"],
                                                inter_df.iloc[-1]["aop"],
                                                inter_df.iloc[-1]["ma"]])
                rising_edge_mass = inter_df.iloc[-1]["m"]
                rising_edge_utc_str = inter_df.iloc[-1]["utc_str"]
            else:
                raise ValueError("Could not reach eclipse rising edge from applied thrust profile.")
            # orbit_logger = OrbitLogger()
            # orbit_logger.plot_results([inter_df], "hours")
            print(f"Current SMA: {rising_edge_kep_state[0]}")

        print(f"Solver finished with target accuracy of {abs(rising_edge_kep_state[0] - target_sma)} km.")

        return out_df_list

    def propagate_indirect_segment(self,
                                    propagator: Propagator,
                                    kep_state: np.ndarray,
                                    mass: float,
                                    utc_str: str,
                                    termination_condition: PropagatorTerminator|None = None
                                    ):
        """Helper function for propagating defined orbit segments in the solver."""

        sim_hz = 0.1
        orbit_period = astro_utils.get_orbit_period(kep_state[0])
        max_prop_time = 1.2*orbit_period # arbitrary max prop time that has to be at least 1 orbit period

        out_df, termination_cause = propagator.propagate(kep_state,
                                                        mass,
                                                        time_utils.generate_time_grid(utc_str, max_prop_time, 1/sim_hz),
                                                        phase_number=1,
                                                        terminators=[termination_condition],
                                                        show_progress_bar=False)
                                                        # show_plots=True,
                                                        # plot_timescale="hours")

        if termination_cause == PropagatorTerminator.MAX_PROP_TIME:
            raise ValueError("Propagation unexpectedly lasted longer than an orbit period.")
        
        final_mass = out_df.iloc[-1]["m"]
        final_utc_str = out_df.iloc[-1]["utc_str"]
        final_kep_state = np.array([out_df.iloc[-1]["sma"],
                                   out_df.iloc[-1]["ecc"],
                                   out_df.iloc[-1]["inc"],
                                   out_df.iloc[-1]["raan"],
                                   out_df.iloc[-1]["aop"],
                                   out_df.iloc[-1]["ma"]])

        return final_kep_state, final_mass, final_utc_str, termination_cause, out_df

    def single_rev_program(self, shadow_angle: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute thrust angle for a single revolution coplanar transfer. Seeks to 
        maximize change in semi-major axis while maintaining zero eccentricity at end of revolution.

        Alpha defined as angle from tangential (LVLH y-axis) direction, and positive angle is directed outward
        
        Arguments:
            - shadow_angle: Amplitude of shadow arc angle for the current revolution (delta_s) (radians)
        """

        if shadow_angle > 0 and shadow_angle <= np.pi:

            # First, compute start and end points of the thrusting arc. For a single revolution, 
            # the thrusting arc begins at an angle of nu=-delta and ends at an angle of 
            # nu=+delta, where the angular position is defined wrt the Earth-Sun line.
            delta = -0.5 * (shadow_angle - 2*np.pi)

            # Define longitude nu over thrusting arc
            # grid of 1000 points is much more refined than expected MA propagator in LEO
            delta_nu = np.linspace(0, 2*delta, 1000)
            nu = np.linspace(-delta, delta, 1000)

            # Solve for K2 integration constant
            K2 = newton(lambda K2: self.K2_root_fn(K2, delta), x0=0.0)

            # Obtain optimal thrust direction alpha with true anomaly nu as independent variable
            tan_alpha = (K2 * np.sin(nu)) / ( 2*(1 + K2 * np.cos(nu)) )
            alpha = np.arctan(tan_alpha)

            return delta_nu, alpha

        elif shadow_angle == 0:
            # Always apply thrust tangentially
            delta_nu = np.linspace(0, 2*np.pi, 1000)
            alpha = np.zeros(len(delta_nu))
            return delta_nu, alpha

        else:
            raise ValueError("Eclipse arc cannot exceed 180 degrees or be negative.")
        
    def K2_root_fn(self, K2: float, delta: float) -> float:
        """Helper function for K2 root finding."""
        value, _ = quad(lambda nu: self.K2_integrand(K2,nu), -delta, delta)
        return value

    def K2_integrand(self, K2: float, nu: float) -> float:
        """Helper function for K2 root finding."""
        num = K2 * np.sin(nu)**2 + 4 * (1 + K2 * np.cos(nu)) * np.cos(nu)
        den = np.sqrt( ( K2 * np.sin(nu) )**2 + 4*( 1 + K2 * np.cos(nu) )**2 )
        return num / den