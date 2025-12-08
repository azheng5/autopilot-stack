import sys
from pathlib import Path
from tqdm import tqdm
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.resolve()))
import flight_dynamics.astro_utils as astro_utils
from flight_dynamics import Constants
from flight_dynamics.Spacecraft import Spacecraft
from flight_dynamics.Time import Time

class Propagator:
    """
    Propagates 3DOF spacecraft trajectories.
    """

    def __init__(self,
                 initial_time: Time,
                 initial_kep_state: np.ndarray,
                 initial_mass: float,
                 Isp: float,
                 t_final: float,
                 delta_t: float,
                 out_file_name: str,
                 out_columns: list[str],
                 plot_timescale: str) -> None:
        """
        Arguments:
            initial_kep_state: Initial keplerian state [sma (km), ecc, inc (rad), raan (rad), aop (rad), ma (rad)]
        """
        
        self.initial_time = initial_time
        self.initial_mass = initial_mass
        self.initial_kep_state = initial_kep_state
        self.Isp = Isp
        self.t_final = t_final
        self.delta_t = delta_t
        self.out_file_name = out_file_name
        self.out_columns = out_columns
        self.plot_timescale = plot_timescale

        self.propagate()

    def propagate(self) -> None:
        """
        Main propagation tool

        Arguments:
            create_plots: Whether or not to create plots of the results
        """

        # Initialization
        terminate = False
        initial_cart_state = astro_utils.kep2cart(self.initial_kep_state, self.initial_time.et) 
        step_counter = 0
        x_curr = np.concatenate(([self.initial_mass], initial_cart_state)) #["m","rx","ry","rz","vx","vy","vz"]
        t_curr = 0.0 # seconds since start time
        progress_bar = tqdm(total=100, desc="Propagating", unit="%")
        curr_progress = 0.0

        # Set up data to be logged during propagation process. Additional data
        # may be computed during post-processing
        logged_columns = ["t","m","rx","ry","rz","vx","vy","vz","sma","ecc","inc","raan","aop","ma"]
        logged_data = np.empty((0, len(logged_columns)))

        # Main propagation loop
        while not terminate:

            # Log data in current iteration
            curr_et = self.initial_time.et + t_curr
            curr_cart_state = x_curr[1:7]
            curr_kep_state = astro_utils.cart2kep(curr_cart_state, curr_et)
            out_entry = np.concatenate(([t_curr], x_curr, curr_kep_state))
            logged_data = np.vstack((logged_data, out_entry))

            # Compute control input
            u_curr = np.array([0,0,0])

            # Compute RK4 step
            x_next = self.rk4_step(self.eom, t_curr, x_curr, u_curr)

            # Termination logic
            terminate = self.check_termination_conditions(t_curr)

            # Update progress bar
            sim_progress = (t_curr / self.t_final) * 100
            delta_progress = sim_progress - curr_progress
            progress_bar.update(delta_progress)

            # Prepare for next iteration
            x_curr = x_next
            t_curr = t_curr + self.delta_t
            step_counter = step_counter + 1
            curr_progress = sim_progress

        # Process output
        print("Propagation complete.")
        progress_bar.close()
        logged_df = pd.DataFrame(logged_data, columns=logged_columns)
        self.process_output(logged_df)

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
        a_thrust = u / m

        # Compute mass derivative
        m_dot = -T_mag / (self.Isp * Constants.G0)

        a_tot = a_g + a_j2 + a_drag + a_thrust
        x_dot = np.array([m_dot, 
                          vx, vy, vz,
                          a_tot[0], a_tot[1], a_tot[2]])
        
        return x_dot


    def rk4_step(self,fn: Callable[[float, np.ndarray, np.ndarray],np.ndarray],
                t: float,
                x: np.ndarray,
                u: np.ndarray) -> np.ndarray:
        
        k1 = fn(t,x,u)
        k2 = fn(t + self.delta_t/2, x + (self.delta_t/2)*k1, u)
        k3 = fn(t + self.delta_t/2, x + (self.delta_t/2)*k2, u)
        k4 = fn(t + self.delta_t, x + self.delta_t*k3, u)

        return x + (self.delta_t/6) * (k1 + 2*k2 + 2*k3 + k4)

    def check_termination_conditions(self,
                                     t: float):

        if (t >= self.t_final):
            return True
        return False

    def drag_model(self):
        pass
    
    def process_output(self,
                       logged_df: pd.DataFrame) -> None:

        # Save processed output to csv
        logged_df.to_csv(Constants.OUT_PATH / self.out_file_name,index=False)

        # Plot final result
        self.plot_results(logged_df)
    
    def plot_results(self, 
                     out_df: pd.DataFrame) -> None:

        # Convert angles to degrees for plotting
        out_df["inc"] = out_df["inc"] * (180/np.pi)
        out_df["raan"] = out_df["raan"] * (180/np.pi)
        out_df["aop"] = out_df["aop"] * (180/np.pi)
        out_df["ma"] = out_df["ma"] * (180/np.pi)

        # Select timescale for x-axis
        if self.plot_timescale == "days":
            out_df["t"] = out_df["t"] / (24*60*60)
        elif self.plot_timescale == "hours":
            out_df["t"] = out_df["t"] / (60*60)
        elif self.plot_timescale == "minutes":
            out_df["t"] = out_df["t"] / 60
        elif self.plot_timescale == "seconds":
            out_df["t"] = out_df["t"]
        else:
            raise ValueError(f"Invalid plot_timescale: {self.plot_timescale}. Choose from 'seconds', 'minutes', 'hours', or 'days'.")


        fig, axes = plt.subplots(3, 1)

        axes[0].plot(out_df["t"],out_df["rx"])
        axes[0].set_ylabel("rx (km)")
        axes[0].grid(True)

        axes[1].plot(out_df["t"],out_df["ry"])
        axes[1].set_ylabel("ry (km)")
        axes[1].grid(True)

        axes[2].plot(out_df["t"],out_df["rz"])
        axes[2].set_ylabel("rz (km)")
        axes[2].set_xlabel(f"Time ({self.plot_timescale})")
        axes[2].grid(True)

        plt.suptitle("Cartesian Position")
        plt.tight_layout()
        # plt.savefig(Constants.OUT_PATH / "position.jpg", dpi=300, bbox_inches="tight")


        fig, axes = plt.subplots(3, 1)

        axes[0].plot(out_df["t"],out_df["vx"])
        axes[0].set_ylabel("vx (km/s)")
        axes[0].grid(True)

        axes[1].plot(out_df["t"],out_df["vy"])
        axes[1].set_ylabel("vy (km/s)")
        axes[1].grid(True)

        axes[2].plot(out_df["t"],out_df["vz"])
        axes[2].set_ylabel("vz (km/s)")
        axes[2].set_xlabel(f"Time ({self.plot_timescale})")
        axes[2].grid(True)

        plt.suptitle("Cartesian Velocity")
        plt.tight_layout()
        # plt.savefig(Constants.OUT_PATH / "velocity.jpg", dpi=300, bbox_inches="tight")

        fig, axes = plt.subplots(3, 2)
        axes[0,0].plot(out_df["t"],out_df["sma"])
        axes[0,0].set_ylabel("sma (km)")
        axes[0,0].grid(True)

        axes[1,0].plot(out_df["t"],out_df["ecc"])
        axes[1,0].set_ylabel("ecc")
        axes[1,0].grid(True)

        axes[2,0].plot(out_df["t"],out_df["inc"])
        axes[2,0].set_ylabel("inc (deg)")
        axes[2,0].grid(True)

        axes[0,1].plot(out_df["t"],out_df["raan"])
        axes[0,1].set_ylabel("raan (deg)")
        axes[0,1].grid(True)

        axes[1,1].plot(out_df["t"],out_df["aop"])
        axes[1,1].set_ylabel("aop (deg)")
        axes[1,1].grid(True)

        axes[2,1].plot(out_df["t"],out_df["ma"])
        axes[2,1].set_ylabel("ma (deg)")
        axes[2,1].grid(True)

        plt.suptitle("Cartesian Velocity")
        fig.supxlabel(f"Time ({self.plot_timescale})")
        plt.tight_layout()


        plt.show()