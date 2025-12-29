import sys
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
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

class DirectSolverLogger:
    """
    Support class for `DirectSolver` for logging internal direct solver 
    performance metrics. Namely, information over solver iterations and 
    mean propagated orbits.
    """

    def __init__(self):
        self.logged_columns = [f.name for f in fields(DirectSolverLogEntry)]

        # List of log entry objects
        self.logged_data: list = []

        # Counter for number of iterations an optimizer has taken
        self.iter_count: int|None = None

    def log_current_entry(self, log_entry: DirectSolverLogEntry) -> None:
        self.logged_data.append([getattr(log_entry, f.name) for f in fields(DirectSolverLogEntry)])

    def convert_to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.logged_data, columns=self.logged_columns)

    def plot_iterations(self) -> None:
        """
        Plot iterations
        """

        out_df = self.convert_to_df()

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

        plt.show()