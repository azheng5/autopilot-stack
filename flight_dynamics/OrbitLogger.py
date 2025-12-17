import sys
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from flight_dynamics import Constants

@dataclass
class LogEntry:
    t: float
    m: float
    rx: float
    ry: float
    rz: float
    vx: float
    vy: float
    vz: float
    sma: float
    ecc: float
    inc: float
    raan: float
    aop: float
    ma: float
    eclipse_status: float
    hx: float
    hy: float
    hz: float
    hmag: float
    orbit_period: float
    utc_str: str
    et: float
    T_ECI_x: float
    T_ECI_y: float
    T_ECI_z: float
    T_LVLH_x: float
    T_LVLH_y: float
    T_LVLH_z: float
    T_mag: float
    thrust_angle: float

class OrbitLogger:

    def __init__(self):
        self.logged_columns = [f.name for f in fields(LogEntry)]
        
    def stitch_dataframes(self, out_df_list: list[pd.DataFrame]) -> pd.DataFrame:

        # Stitch dataframes together
        df = out_df_list[0].copy(deep=True)
        prev_df = out_df_list[0].copy(deep=True)
        if len(out_df_list) > 1:
            for ind in range(1,len(out_df_list)):
                curr_df = out_df_list[ind].copy(deep=True)
                curr_df["t"] = curr_df["t"] + prev_df.iloc[-1]["t"]
                df = pd.concat([df, curr_df], ignore_index=True)
                prev_df = curr_df.copy(deep=True)
        return df
        
    def save_to_csv(self, out_df_list: list[pd.DataFrame], out_file_name: str) -> None:
        
        # Stitch dataframes together
        df = self.stitch_dataframes(out_df_list)

        # Save processed output to csv
        df.to_csv(Constants.OUT_PATH / out_file_name,index=False)
        
    def plot_results(self, 
                     out_df_list: list[pd.DataFrame],
                     plot_timescale: str) -> None:
        """
        Plot propagated orbit dataframes.

        Arguments:
            - out_df_list: If only one dataframe is provided, then only
            that dataframe is plotted. If a list of dataframes are provided,
            then they will be stitched together and assume ascending time order.
            - plot_timescale: Timescale of the plots
        """
        
        # Stitch dataframes together
        df = self.stitch_dataframes(out_df_list)

        # Convert angles to degrees for plotting
        df["inc"] = df["inc"] * (180/np.pi)
        df["raan"] = df["raan"] * (180/np.pi)
        df["aop"] = df["aop"] * (180/np.pi)
        df["ma"] = df["ma"] * (180/np.pi)

        # Select timescale for x-axis
        if plot_timescale == "days":
            df["t"] = df["t"] / (24*60*60)
        elif plot_timescale == "hours":
            df["t"] = df["t"] / (60*60)
        elif plot_timescale == "minutes":
            df["t"] = df["t"] / 60
        elif plot_timescale == "seconds":
            df["t"] = df["t"]
        else:
            raise ValueError(f"Invalid plot_timescale: {plot_timescale}. Choose from 'seconds', 'minutes', 'hours', or 'days'.")

        # Plot cartesian position
        fig, axes = plt.subplots(3, 1)
        axes[0].plot(df["t"],df["rx"])
        axes[0].set_ylabel("rx (km)")
        axes[0].grid(True)
        axes[1].plot(df["t"],df["ry"])
        axes[1].set_ylabel("ry (km)")
        axes[1].grid(True)
        axes[2].plot(df["t"],df["rz"])
        axes[2].set_ylabel("rz (km)")
        axes[2].set_xlabel(f"Time ({plot_timescale})")
        axes[2].grid(True)
        plt.suptitle("Cartesian Position")
        plt.tight_layout()

        # Plot cartesian velocity
        fig, axes = plt.subplots(3, 1)
        axes[0].plot(df["t"],df["vx"])
        axes[0].set_ylabel("vx (km/s)")
        axes[0].grid(True)
        axes[1].plot(df["t"],df["vy"])
        axes[1].set_ylabel("vy (km/s)")
        axes[1].grid(True)
        axes[2].plot(df["t"],df["vz"])
        axes[2].set_ylabel("vz (km/s)")
        axes[2].set_xlabel(f"Time ({plot_timescale})")
        axes[2].grid(True)
        plt.suptitle("Cartesian Velocity")
        plt.tight_layout()

        # Plot keplerian elements
        fig, axes = plt.subplots(3, 2)
        axes[0,0].plot(df["t"],df["sma"])
        axes[0,0].set_ylabel("sma (km)")
        axes[0,0].grid(True)
        axes[1,0].plot(df["t"],df["ecc"])
        axes[1,0].set_ylabel("ecc")
        axes[1,0].grid(True)
        axes[2,0].plot(df["t"],df["inc"])
        axes[2,0].set_ylabel("inc (deg)")
        axes[2,0].grid(True)
        axes[0,1].plot(df["t"],df["raan"])
        axes[0,1].set_ylabel("raan (deg)")
        axes[0,1].grid(True)
        axes[1,1].plot(df["t"],df["aop"])
        axes[1,1].set_ylabel("aop (deg)")
        axes[1,1].grid(True)
        axes[2,1].plot(df["t"],df["ma"])
        axes[2,1].set_ylabel("ma (deg)")
        axes[2,1].grid(True)
        plt.suptitle("Keplerian Elements")
        fig.supxlabel(f"Time ({plot_timescale})")
        plt.tight_layout()

        # Plot eclipse status
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(df["t"], df["eclipse_status"])
        ax.set_xlabel(f"Time ({plot_timescale})")
        ax.set_ylabel("Eclipse Status")

        # Plot angular momentum components
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(df["t"], df["hx"])
        ax.plot(df["t"], df["hy"])
        ax.plot(df["t"], df["hz"])
        ax.set_xlabel(f"Time ({plot_timescale})")
        ax.set_ylabel("Angular Momentum (kg-m^2)/s)")

        # Plot orbit period
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(df["t"], df["orbit_period"])
        ax.set_xlabel(f"Time ({plot_timescale})")
        ax.set_ylabel("Orbit Period")

        # Plot thrust components
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(df["t"], df["T_LVLH_x"]*1000)
        ax.plot(df["t"], df["T_LVLH_y"]*1000)
        ax.plot(df["t"], df["T_LVLH_z"]*1000)
        ax.set_xlabel(f"Time ({plot_timescale})")
        ax.set_ylabel("Thrust (N)")

        # Plot thrust angle
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(df["t"], df["thrust_angle"]*(180/np.pi))
        ax.set_xlabel(f"Time ({plot_timescale})")
        ax.set_ylabel("Thrust Angle (deg)")

        plt.show()