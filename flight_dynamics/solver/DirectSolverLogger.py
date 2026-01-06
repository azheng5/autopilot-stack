import sys
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants

@dataclass
class DirectSolverLogEntry:
    iteration: int
    final_sma: float
    final_ecc: float
    final_inc: float
    final_raan: float
    final_aop: float
    final_mass: float
    tf: float

@dataclass
class MeanPropLogEntry:
    t: float
    sma: float
    h: float
    k: float
    p: float
    q: float
    m: float

@dataclass
class DirectSolverResult:
    tf: float
    sma_grid: np.ndarray
    a_lambda_a_grid: np.ndarray
    lambda_e_grid: np.ndarray
    lambda_i_grid: np.ndarray

class DirectSolverLogger:
    """
    Support class for `DirectSolver` for logging internal direct solver 
    performance metrics. Namely, information over solver iterations and 
    mean propagated orbits.
    """

    def __init__(self):
        self.logged_iter_columns = [f.name for f in fields(DirectSolverLogEntry)]
        self.logged_prop_columns = [f.name for f in fields(MeanPropLogEntry)]

        # List of log entry objects
        self.logged_iter_data: list = []
        self.logged_prop_data: list = []

        # Counter for number of iterations an optimizer has taken
        self.iter_count: int|None = None

        # Direct solver result
        self.ds_result: DirectSolverResult|None = None

    def log_current_iter_entry(self, log_entry: DirectSolverLogEntry) -> None:
        self.logged_iter_data.append([getattr(log_entry, f.name) for f in fields(DirectSolverLogEntry)])

    def convert_to_iter_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.logged_iter_data, columns=self.logged_iter_columns)

    def log_current_prop_entry(self, log_entry: MeanPropLogEntry) -> None:
        self.logged_prop_data.append([getattr(log_entry, f.name) for f in fields(MeanPropLogEntry)])

    def convert_to_prop_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.logged_prop_data, columns=self.logged_prop_columns)

    def plot_iterations(self) -> None:
        """
        Plot iterations
        """

        out_df = self.convert_to_iter_df()

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(out_df["iteration"], out_df["final_sma"])
        ax.set_xlabel(f"Iteration")
        ax.set_ylabel("Final SMA")

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(out_df["iteration"], out_df["final_ecc"])
        ax.set_xlabel(f"Iteration")
        ax.set_ylabel("Final ECC")

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(out_df["iteration"], out_df["final_inc"])
        ax.set_xlabel(f"Iteration")
        ax.set_ylabel("Final INC")

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(out_df["iteration"], out_df["final_raan"])
        ax.set_xlabel(f"Iteration")
        ax.set_ylabel("Final RAAN")

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(out_df["iteration"], out_df["final_aop"])
        ax.set_xlabel(f"Iteration")
        ax.set_ylabel("Final AOP")

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(out_df["iteration"], out_df["final_mass"])
        ax.set_xlabel(f"Iteration")
        ax.set_ylabel("Final mass")

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(out_df["iteration"], out_df["tf"])
        ax.set_xlabel(f"Iteration")
        ax.set_ylabel("Final Time")

    def plot_mean_propagation(self) -> None:
        """
        Plot mean propagated trajectory and costates
        """

        out_df = self.convert_to_prop_df()

        # Convert seconds to days
        out_df["t"] = out_df["t"] / (24*60*60)
        equin_data = np.column_stack((
            out_df[["sma","h","k","p","q"]].to_numpy(), np.zeros(len(out_df["sma"]))
        ))
        kep_data = astro_utils.equinoctial_to_classical(
            equin_data
        )

        fig, axes = plt.subplots(3, 2)
        axes[0,0].plot(out_df["t"],out_df["sma"])
        axes[0,0].set_ylabel("sma (km)")
        axes[0,0].grid(True)
        axes[1,0].plot(out_df["t"],out_df["h"])
        axes[1,0].set_ylabel("h")
        axes[1,0].grid(True)
        axes[2,0].plot(out_df["t"],out_df["k"])
        axes[2,0].set_ylabel("k")
        axes[2,0].grid(True)
        axes[0,1].plot(out_df["t"],out_df["p"])
        axes[0,1].set_ylabel("p")
        axes[0,1].grid(True)
        axes[1,1].plot(out_df["t"],out_df["q"])
        axes[1,1].set_ylabel("q")
        axes[1,1].grid(True)
        plt.suptitle("Mean Equinoctial Elements")
        fig.supxlabel(f"Time (days)")
        plt.tight_layout()

        fig, axes = plt.subplots(3, 2)
        axes[0,0].plot(out_df["t"],kep_data[:,0])
        axes[0,0].set_ylabel("sma (km)")
        axes[0,0].grid(True)
        axes[1,0].plot(out_df["t"],kep_data[:,1])
        axes[1,0].set_ylabel("ecc")
        axes[1,0].grid(True)
        axes[2,0].plot(out_df["t"],kep_data[:,2])
        axes[2,0].set_ylabel("inc")
        axes[2,0].grid(True)
        axes[0,1].plot(out_df["t"],kep_data[:,3])
        axes[0,1].set_ylabel("raan")
        axes[0,1].grid(True)
        axes[1,1].plot(out_df["t"],kep_data[:,4])
        axes[1,1].set_ylabel("aop")
        axes[1,1].grid(True)
        plt.suptitle("Mean Keplerian Elements")
        fig.supxlabel(f"Time (days)")
        plt.tight_layout()

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(out_df["t"], out_df["m"])
        ax.set_xlabel(f"Time (days)")
        ax.set_ylabel("Mass (kg)")

    def plot_costates(self):
        
        if self.ds_result is None:
            raise ValueError("No result was logged.")

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.ds_result.sma_grid, self.ds_result.a_lambda_a_grid,label="a_lambda_a")
        ax.plot(self.ds_result.sma_grid, self.ds_result.lambda_e_grid,label="lambda_e")
        ax.plot(self.ds_result.sma_grid, self.ds_result.lambda_i_grid,label="lambda_i")
        ax.set_xlabel(f"SMA (km)")
        plt.legend()