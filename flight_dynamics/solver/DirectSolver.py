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

class DirectSolver:
    """
    Standalone low thrust solver tool that applies a direct optimization technique 
    to compute LEO transfers.

    References:
        - Spacecraft Trajectory Optmization
    """
    
    def __init__(self, 
                 cfg: DirectSolverSettings) -> None:
        self.cfg = cfg
        self.ds_logger = DirectSolverLogger()

        # Default universal SLSQP solver function tolerance setting
        self.ftol = 1e-6

    def perform_control_parameterization(self) -> DirectSolverResult:
        """
        Parameterize controls to drive low thrust spacecraft from an
        initial keplerian state to a final target SMA and ECC

        Can choose to use either equinoctial or keplerian elements to solve.
        
        Returns:
            - Decision vector z
        """

        # Generate initial guess for decision vector and absolute final time guess
        initial_guess = self.generate_initial_guess()
        
        # Define objective function
        objective_handle = lambda z: self.minimum_time_objective(z)

        # Define bounds on orbital elements and final time
        # When lambda_a and lambda_e are both 0, the 
        # optimal thrust angle equation hits a singularity,
        # so numerically we avoid using 0.
        #TODO 5 is still hardcoded
        # a_lambda_a_bounds = [(-20,-0.0001)]*self.cfg.num_costate_nodes
        # lambda_e_bounds = [(-0.01,10.0)]*self.cfg.num_costate_nodes
        a_lambda_a_bounds = [(None,None)]*self.cfg.num_costate_nodes
        lambda_e_bounds = [(None,None)]*self.cfg.num_costate_nodes
        lambda_i_bounds = [(0,0)]*self.cfg.num_costate_nodes
        rel_tf_bounds = (0.0,50*initial_guess[-1])
        bounds = [*a_lambda_a_bounds,
                  *lambda_e_bounds,
                  *lambda_i_bounds,
                  rel_tf_bounds]
        
        # Construct terminal constraints denoting final target SMA
        terminal_eq_handle = lambda z: self.terminal_constraints(z)["eq_constraint"]
        terminal_ineq_handle = lambda z: self.terminal_constraints(z)["ineq_constraint"]

        # Define SLSQP solver configuration
        slsqp_options = {
                "max_iter": 5,
                "disp": True,
                "ftol": self.ftol,
            }
        
        # Define callback function
        optimizer_callback_handle = lambda z: self.optimizer_callback(z)

        # Run minimizer
        start_time = time.perf_counter()
        opt_result = minimize(
            fun= objective_handle,
            x0= initial_guess,
            method="SLSQP", #sequential least squares programming
            bounds=bounds,
            constraints=[
                {"type": "eq","fun": terminal_eq_handle},
                {"type": "ineq","fun": terminal_ineq_handle}
            ],
            callback=optimizer_callback_handle,
            options=slsqp_options
        )
        end_time = time.perf_counter()

        #TODO put all this stuff in a txt file output
        print(f"-------------------- SOLVER COMPLETED IN {end_time-start_time:.3f} SECONDS --------------------")
        a_lambda_a_grid, lambda_e_grid, lambda_i_grid, rel_tf = self.unpack_z(opt_result.x)

        final_eq_x_bar = self.mean_equinoctial_propagation(opt_result.x, log=True)
        final_mean_equin_state = final_eq_x_bar[0:5]
        final_mass = final_eq_x_bar[5]
        final_mean_kep_state = astro_utils.equinoctial_to_classical(np.concatenate((final_mean_equin_state,[0.0])))
        tf = rel_tf*self.cfg.tf_tol
        #TODO: this is debug stuff remove
        print(f"Final SMA: {final_mean_kep_state[0]}")
        print(f"Final ECC: {final_mean_kep_state[1]}")
        print(f"Final INC: {final_mean_kep_state[2]}")
        print(f"Final RAAN: {final_mean_kep_state[3]}")
        print(f"Final AOP: {final_mean_kep_state[4]}")
        print(f"Final mass: {final_mass}")
        print(f"Final time: {tf}")
        print(f"SMA constraint: {self.sma_constraint(final_mean_kep_state[0])}")
        print(f"Eccentricity constraint: {self.ecc_constraint(final_mean_kep_state[1])}")
        print(f"ftol: {self.ftol}")

        # Log final result
        self.ds_logger.ds_result = DirectSolverResult(
            tf=tf,
            sma_grid=self.cfg.sma_grid,
            a_lambda_a_grid=a_lambda_a_grid,
            lambda_e_grid=lambda_e_grid,
            lambda_i_grid=lambda_i_grid
        )

        return self.ds_logger.ds_result

    def pack_z(self, 
               a_lambda_a_grid: np.ndarray, 
               lambda_e_grid: np.ndarray,
               lambda_i_grid: np.ndarray,
               rel_tf: float):
        """Generate decision vector from costate grids and final time"""
        return np.concatenate([
            a_lambda_a_grid,
            lambda_e_grid,
            lambda_i_grid,
            np.array([rel_tf])
        ])

    def unpack_z(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Unpack decision vector into costate grids and final time"""
        num_costate_nodes = int((len(z)-1)/3)
        a_lambda_a_grid = z[0:num_costate_nodes]
        lambda_e_grid = z[num_costate_nodes:2*num_costate_nodes]
        lambda_i_grid = z[2*num_costate_nodes:3*num_costate_nodes]
        rel_tf = z[-1]
        return a_lambda_a_grid, lambda_e_grid, lambda_i_grid, rel_tf

    def generate_initial_guess(self) -> np.ndarray:
        """
        Generates costate grid and final time guesses by assuming
        thrust follows a near-tangent thrusting profile over 
        quasi-circular orbit.

        References:
            - Edelbaum, Propulsion Requirements for Controllable Satellites
        """

        # Combo of a_lambda_a=-1 and low ecc is near-tangent thrusting
        a_lambda_a_grid_guess = -1*np.ones(self.cfg.num_costate_nodes)

        # Keep ecc costate low and near zero
        lambda_e_grid_guess = 0.01*np.ones(self.cfg.num_costate_nodes)

        #TODO defaulted to zero for now
        lambda_i_grid_guess = np.zeros(self.cfg.num_costate_nodes)
       
        # Generate final time guess using rocket equation, mfr,
        # quasi-circular tangent constant thrust assumptions
        sma = self.cfg.initial_kep_state[0]
        ecc = self.cfg.initial_kep_state[1]
        ta = self.cfg.initial_kep_state[5]
        r0 = astro_utils.compute_radius(sma, ecc, ta)
        v0 = astro_utils.vis_viva(sma, r0)
        delta_v = 0.5 * ((self.cfg.target_sma - sma)/sma) * v0
        T = self.cfg.A_mag*self.cfg.initial_mass
        ve = self.cfg.spacecraft.Isp * Constants.G0
        tf = ((self.cfg.initial_mass*ve)/T) * (1 - np.exp(-delta_v/ve))
        # Scale by factor of 1.2 to account for eclipse times
        #minor TODO 1.2 is arbitrary, scaling could be more accurate 
        # by getting eclipse arc time of initial orbit
        tf_guess = 1.2 * tf

        rel_tf_guess = tf_guess / self.cfg.tf_tol

        return np.concatenate([
            a_lambda_a_grid_guess,
            lambda_e_grid_guess,
            lambda_i_grid_guess,
            np.array([rel_tf_guess])
        ])

    def minimum_time_objective(self, z: np.ndarray) -> float:
        """
        Performance index specifed to achieve minimum time. 
        Normalizes flight time by some reference time (i.e 
        initial guess for the solved time) so that returned
        value of the objective function is OOM ~1
        """
        return (float(z[-1]) / self.cfg.tf_tol) * self.ftol
    
    # def minimum_fuel_objective(self, z: np.ndarray) -> float:
    #     """Performance index specifying minimum time"""
    #     #TODO make it -m(tf), but they should be nearly identical, but comp time doubels....
    #     #TODO this function still not usable
    #     return 0.0
    
    def sma_constraint(self, final_sma: float) -> float:
        """
        Semi-major axis terminal constraint.
        """
        # sma_relative_error_tol = 1e-5
        # sma_relative_error = sma_relative_error_tol - abs((final_sma - target_sma)/target_sma)
        # return sma_relative_error

        # return (final_sma - self.cfg.target_sma)/self.cfg.target_sma
        return ((final_sma - self.cfg.target_sma)/(self.cfg.sma_tol)) * self.ftol

    def ecc_constraint(self, final_ecc: float) -> float:
        """
        Eccentrcity terminal constraint.
        """
        # return 1e-3 - abs(final_ecc - target_ecc)
        # return final_ecc - self.cfg.target_ecc
        return ((final_ecc - self.cfg.target_ecc)/(self.cfg.ecc_tol)) * self.ftol

    def terminal_constraints(self,
                             z: np.ndarray):
        """
        Terminal constraint callback function for evaluating residual from 
        target keplerian states.

        Arguments:
            - z: Decision vector
            - initial_equin_state: Initial equinoctial state
            - initial_mass: Initial mass
            - initial_utc_str: Initial UTC string
            - sma_grid: Grid of SMA values from initial SMA to target SMA
        """
        _, _, _, rel_tf = self.unpack_z(z)
        final_eq_x_bar = self.mean_equinoctial_propagation(z)
        final_mean_equin_state = final_eq_x_bar[0:5]
        final_mass = final_eq_x_bar[5]

        # NOTE: Dummy value of 0 for F inserted, so resulting E is also a dummy value with no meaning
        final_mean_kep_state = astro_utils.equinoctial_to_classical(np.concatenate((final_mean_equin_state,[0.0])))

        final_sma = final_mean_kep_state[0]
        final_ecc = final_mean_kep_state[1]
        final_inc = final_mean_kep_state[2]
        final_raan = final_mean_kep_state[3]
        final_aop = final_mean_kep_state[4]


        # sma_constraint = 1e-3**2 - ((final_sma - sma_grid[-1]) / sma_grid[-1])**2
        # sma_target = 1 - abs(final_sma - sma_grid[-1])
        sma_relative_error = self.sma_constraint(final_sma)
        # ecc_relative_error_tol = 0.01
        # ecc_relative_error = ecc_relative_error_tol - abs((final_ecc - target_ecc)/target_ecc)
        ecc_target = self.ecc_constraint(final_ecc)
        # negative_ecc_constraint = final_ecc
        # lower_sma_constraint = self.cfg.initial_equin_state[0]-10
        dry_mass = self.cfg.spacecraft.dry_mass #TODO fix this
        mass_constraint = final_mass - dry_mass

        constraint_dict = {
            "eq_constraint": np.array([sma_relative_error,
                                       ecc_target]),
            "ineq_constraint": np.array([
                mass_constraint
            ]),
        }

        #TODO delete and move
        print(
            f"SMA: {final_sma:12.6f} | "
            f"ECC: {final_ecc:12.6f} | "
            f"INC: {final_inc:12.6f} | "
            f"RAAN: {final_raan:12.6f} | "
            f"AOP: {final_aop:12.6f} | "
            f"MASS: {final_mass:12.6f} | "
            f"SMA_REL: {sma_relative_error:12.6f} | "
            f"TF: {rel_tf*self.cfg.tf_tol:12.6f}"
        )

        return constraint_dict
    
    def optimizer_callback(self,
                           z: np.ndarray) -> None:

        # Initialize counter if optimizer callback being called for the first time
        if self.ds_logger.iter_count is None:
            self.ds_logger.iter_count = 1
        
        print(f"--- Callback {self.ds_logger.iter_count} ---")
        a_lambda_a_grid, lambda_e_grid, lambda_i_grid, rel_tf = self.unpack_z(z)
        final_eq_x_bar = self.mean_equinoctial_propagation(z)
        final_mean_equin_state = final_eq_x_bar[0:5]
        # NOTE: Dummy value of 0 for F inserted, so resulting E is also a dummy value with no meaning
        final_mean_kep_state = astro_utils.equinoctial_to_classical(np.concatenate((final_mean_equin_state,[0.0])))

        # Construct and append log entry
        log_entry = DirectSolverLogEntry(
            iteration=self.ds_logger.iter_count,
            final_sma=final_mean_kep_state[0],
            final_ecc=final_mean_kep_state[1],
            final_inc=final_mean_kep_state[2],
            final_raan=final_mean_kep_state[3],
            final_aop=final_mean_kep_state[4],
            final_mass=final_eq_x_bar[5],
            tf=rel_tf*self.cfg.tf_tol
        )
        self.ds_logger.log_current_iter_entry(log_entry)

        self.ds_logger.iter_count = self.ds_logger.iter_count + 1
        return None

    def mean_equinoctial_propagation(self,
                                    z: np.ndarray,
                                    log: bool = False) -> np.ndarray:
        """
        Given initial state and grid of costates variables,
        propagates averaged state to final time.
        """

        if log and self.ds_logger.logged_prop_data != []:
            raise ValueError("Log is not empty.")

        a_lambda_a_grid, lambda_e_grid, lambda_i_grid, rel_tf = self.unpack_z(z)
        tf = rel_tf * self.cfg.tf_tol

        # Initialize propagation
        t_curr = 0.0 # seconds since start time
        x_bar_curr = np.array([
            self.cfg.initial_equin_state[0],
            self.cfg.initial_equin_state[1],
            self.cfg.initial_equin_state[2],
            self.cfg.initial_equin_state[3],
            self.cfg.initial_equin_state[4],
            self.cfg.initial_mass
        ])

        mean_equinoctial_derivative_handle = lambda t, x: self.mean_equinoctial_derivative(t,
                                                                                           x,
                                                                                           a_lambda_a_grid, 
                                                                                           lambda_e_grid)

        # Enter propagation loop
        num_steps = 40 #TODO arbitrary
        delta_t = tf/num_steps
        for step_ind in range(num_steps+1):
    
            if log:
                logged_entry = MeanPropLogEntry(
                    t=t_curr,
                    sma=x_bar_curr[0],
                    h=x_bar_curr[1],
                    k=x_bar_curr[2],
                    p=x_bar_curr[3],
                    q=x_bar_curr[4],
                    m=x_bar_curr[5]
                )
                self.ds_logger.log_current_prop_entry(logged_entry)

            if x_bar_curr[0] < Constants.R_EARTH + 160:
                print(f"WARNING: Current solver iteration targeted an infeasible SMA: {x_bar_curr[0]}.")

            if t_curr >= tf:
                break

            x_bar_next = timestepper_utils.heun_step(mean_equinoctial_derivative_handle,
                                                    t_curr,
                                                    x_bar_curr,
                                                    delta_t)
            
            # Prepare for next iteration (k -> k+1)
            x_bar_curr = x_bar_next
            t_curr = t_curr + delta_t

        return x_bar_curr
    
    def mean_equinoctial_derivative(self, 
                                  t: float,
                                  mean_state: np.ndarray,
                                  a_lambda_a_grid: np.ndarray,
                                  lambda_e_grid: np.ndarray) -> np.ndarray:
        """
        Computes each state element's mean time rate of change.

        Arguments:
            - t: time
            - mean_state: Mean slow equinoctal states and mass [sma h k p q m]
            - lambda_a: SMA costate
            - lambda_e: ECC costate
            
        Returns:
            - mean state derivative [sma h k p q m]
        """

        curr_sma = mean_state[0]
        m = mean_state[5]

        initial_et = spice.utc2et(self.cfg.initial_utc_str)

        # Interpolate costate values
        a_lambda_a = np.interp(curr_sma, self.cfg.sma_grid, a_lambda_a_grid)
        lambda_e = np.interp(curr_sma, self.cfg.sma_grid, lambda_e_grid)
        lambda_a = a_lambda_a / curr_sma

        mean_equin_state = mean_state[0:5]
        #NOTE: arbitrary eccentric anomaly set to zero
        kep_state = astro_utils.equinoctial_to_classical(
            np.concatenate((mean_equin_state, [0.0]))
        )
        ecc = kep_state[1]
        raan = kep_state[3]
        aop = kep_state[4]

        #DEBUG #TODO remove
        if ecc == 0:
            print("WARNING ECC IS ZERO!")
        
        # Generate integration limits
        # NOTE: Need to be careful here, SPICE defines angles as [0,2pi]
        curr_et = initial_et + t
        curr_utc_str = spice.et2utc(curr_et, 'ISOC', 6)
        if isinstance(curr_utc_str,str):
            ta_en, ta_ex = eclipse_utils.compute_eclipse_angles(kep_state, curr_utc_str)
        else:
            raise ValueError("`curr_utc_str` is not a string.")
        
        if ta_en ==0 and ta_ex == 0:
            F_en = np.pi
            F_ex = -np.pi
        else:
            if ta_en > np.pi:
                ta_en = ta_en - 2*np.pi
            if ta_ex > np.pi:
                ta_ex = ta_ex - 2*np.pi
            E_en = astro_utils.true2eccentric(ta_en, ecc)
            F_en = np.mod(raan + aop + E_en, 2*np.pi)
            E_ex = astro_utils.true2eccentric(ta_ex, ecc)
            F_ex = np.mod(raan + aop + E_ex, 2*np.pi)
        
        # For some reason (0,2pi) bounds dont work
        F_en = np.pi
        F_ex = -np.pi
        F_grid = np.linspace(F_ex, F_en, 20) #TODO 20 is arbitrary still
        #TODO assume thruster power and hence thurst is zero when sc in in shadow. If no shadowing conditions exist
        # for a prtcly osculating orbit, then limits are from E_ex=-pi to E_en=pi

        # Compute mean rate of change of orbital elements from thrust and 2-body point mass effects
        # NOTE: We do not integrate over the eclipse arc b/c mean sma, ecc, inc, and mass
        # should not change over that regime
        integrand_grid = np.zeros((len(F_grid),5))
        for ind in range(len(F_grid)):
            F_i = F_grid[ind]
            full_equin_state = np.concatenate((mean_state[0:5],[F_i]))
            integrand_grid[ind,:]= self.equinoctial_integrand(full_equin_state, lambda_a, lambda_e)
        mean_equin_state_dot = np.trapezoid(integrand_grid, F_grid, axis=0)

        # Compute averaged rates of change from J2 effect
        mean_equin_state_dot_j2 = self.compute_equinoctial_j2_rate(mean_equin_state)
        # mean_equin_state_dot_j2 = np.array([0,0,0,0,0])

        total_mean_equin_state_dot = mean_equin_state_dot + mean_equin_state_dot_j2

        # Compute mass derivative
        m_dot = -(self.cfg.A_mag*m) / (self.cfg.spacecraft.Isp * Constants.G0) #TODO

        return np.concatenate((total_mean_equin_state_dot, [m_dot]))

    def equinoctial_integrand(self, 
                            equin_state: np.ndarray,
                            lambda_a: float,
                            lambda_e: float) -> np.ndarray:
        """
        Helper callback function for computing the orbit integrand term.

        Arguments:
            - equin_state: [a h k p q F]

        References:
            - Eq 5.11, Spacecraft Trajectory Optimization
        """

        # Extract elements
        sma, h, k, _, _, F= equin_state

        # Intermediate terms
        _, ecc, _, _, _, E = astro_utils.equinoctial_to_classical(equin_state)
        n = astro_utils.compute_mean_motion(sma)
        ta = astro_utils.eccentric2true(E, ecc)
        r = astro_utils.compute_radius(sma, ecc, ta)
        v = astro_utils.vis_viva(sma, r)

        alpha = self.compute_thrust_angle(sma,ecc,ta,
                                          lambda_a,lambda_e)
        
        if alpha > np.pi or alpha < -np.pi:
            print("WARNING: thrust angle has a negative tangential component")
        
        # Conmpute instantaneous state derivative
        slow_state_dot = self.slow_equinoctial_diff_eq(equin_state, alpha)

        # Change of variable term: (dt/dF)_T
        dt_dF_T = (1 - k * np.cos(F) - h * np.sin(F) ) / (2*np.pi)

        return slow_state_dot * dt_dF_T


    def slow_equinoctial_diff_eq(self,
                                equin_state: np.ndarray,
                                alpha: float) -> np.ndarray:
        """
        Compute slow equinoctial state derivatives using the full 
        equinoctial state.

        Arguments:
            - equin_state: [a h k p q F]
            - alpha: in plane thrust pitch angle

        References:
            - Kluever, Direct Approach for Computing Near-Optimal Low-Thrust 
            Earth-Orbit Transfers
        """

        # Unpack elements
        a, h, k, p, q, F = equin_state
        
        # Intermediate variables
        G = np.sqrt(1.0 - h**2 - k**2)
        b = 1.0 / (1.0 + G)
        n = np.sqrt(Constants.EARTH_MU / a**3)
        r = a * (1.0 - k*np.cos(F) - h*np.sin(F))
        K = 1.0 + p**2 + q**2
        cF = np.cos(F)
        sF = np.sin(F)
        X  = a * (cF * (1 - b*h**2) + h*k*b*sF - k)
        Y  = a * (sF * (1 - b*k**2) + h*k*b*cF - h)
        Xd = ((n*a**2)/r) * (h*k*b*cF - (1 - (h**2)*b)*sF)
        Yd = ((n*a**2)/r) * ((1 - k*k*b)*cF - h*k*b*sF)
        dX_dh = a * ( -(h*cF - k*sF)*(b + ((h**2)*(b**3))/(1-b)) - (a/r)*cF*(h*b-sF) )
        dX_dk = -a * ( (h*cF - k*sF)*((h*k*b**3)/(1-b)) + 1 + (a/r)*sF*(sF - h*b) )
        dY_dh = a * ( (h*cF - k*sF)*((h*k*b**3)/(1-b)) - 1 + (a/r)*cF*(k*b-cF) )
        dY_dk = a * ( (h*cF - k*sF)*(b + ((k**2)*(b**3))/(1-b)) + (a/r)*sF*(cF - k*b) )

        # Matrix elements
        M11 = ((2*a)/(n*r)) * (h*k*b*cF - (1 - h*h*b)*sF)
        M12 = ((2*a)/(n*r)) * ((1 - k*k*b)*cF - h*k*b*sF)
        M13 = 0.0
        M21 = (G/(n*a*a)) * (dX_dk - h*b*Xd/n)
        M22 = (G/(n*a*a)) * (dY_dk - h*b*Yd/n)
        M23 = (k/(G*n*a*a)) * (q*Y - p*X)
        M31 = -(G/(n*a*a)) * (dX_dh + k*b*Xd/n)
        M32 = -(G/(n*a*a)) * (dY_dh + k*b*Yd/n)
        M33 = -(h/(G*n*a*a))*(q*Y - p*X)
        M41 = 0.0
        M42 = 0.0
        M43 = (K*Y)/(2*G*n*a*a)
        M51 = 0.0
        M52 = 0.0
        M53 = (K*X)/(2*G*n*a*a)

        # Construct matrix
        M = np.array([
            [M11, M12, M13],
            [M21, M22, M23],
            [M31, M32, M33],
            [M41, M42, M43],
            [M51, M52, M53],
        ])

        # Convert from NTH frame to equinoctial frame (Eq. 43-45)
        a_nth = self.compute_nth_thrust_components(alpha)
        a_hat_x = np.cos(alpha) * (Xd)/(np.sqrt(Xd**2 + Yd**2)) + np.sin(alpha) * (Yd)/(np.sqrt(Xd**2 + Yd**2))
        a_hat_y = -np.sin(alpha) * (Xd)/(np.sqrt(Xd**2 + Yd**2)) + np.cos(alpha) * (Yd)/(np.sqrt(Xd**2 + Yd**2))
        a_hat_z = 0.0
        a_pert = np.linalg.norm(a_nth) * np.array([a_hat_x, a_hat_y, a_hat_z])

        # Compute state derivative
        equin_state_dot = M @ (a_pert.reshape(3,1))

        return equin_state_dot.ravel()
    
    def compute_equinoctial_j2_rate(self, mean_equin_state: np.ndarray) -> np.ndarray:
        """
        Computes mean equinoctal element rates from purely J2 effect.

        Arguments: 
            - mean_equin_state: Mean slow equinoctial state [a h k p q]

        Returns:
            - equin_ob_dot
        """

        a, h, k, p, q = mean_equin_state

        n = astro_utils.compute_mean_motion(a)

        ob_factor = 3 * Constants.EARTH_MU * (Constants.R_EARTH**2) * Constants.J2
        Gamma_common_den = 2*n*(a**5) * ((1 - h**2 - k**2)**2) * ((1 + p**2 + q**2)**2)
        Gamma_hk_num = ob_factor * ( 1 - 6*(p**2 + q**2) + 3*(p**2 + q**2)**2 )
        Gamma_pq_num = ob_factor * (1 - p**2 - q**2)

        Gamma_hk = Gamma_hk_num / Gamma_common_den
        Gamma_pq = Gamma_pq_num / Gamma_common_den

        a_ob_dot = 0
        h_ob_dot = Gamma_hk * k
        k_ob_dot = - Gamma_hk * h
        p_ob_dot = - Gamma_pq * q
        q_ob_dot = Gamma_pq * p

        return np.array([a_ob_dot, h_ob_dot, k_ob_dot, p_ob_dot, q_ob_dot])
    
    def mean_keplerian_propagation(self,
                                    z: np.ndarray,
                                   init_kep_state: np.ndarray,
                                   initial_mass: float,
                                   initial_utc_string: str,
                                   sma_grid: np.ndarray,
                                   A_mag: float):
        """
        Given initial state and grid of costates variables,
        propagates averaged state to final time.

        Arguments:
            - init_kep_state: Initial keplerian state [sma ecc inc raan aop ma]
            - initial_mass: Initial mass (kg)
            - initial_utc_string: Initial UTC string
            - sma_grid: Grid of SMA values from initial SMA to target SMA
            - a_lambda_a_grid: Grid of a*lambda_a values
            - lambda_e_grid: Grid of lambda_e_values
            - lambda_i_grid: Grid of lambda_i values
            - tf: Final time

        Returns
        """

        a_lambda_a_grid, lambda_e_grid, lambda_i_grid, tf = self.unpack_z(z)

        # Initialize propagation
        initial_et = spice.utc2et(initial_utc_string)
        t_curr = 0.0 # seconds since start time
        x_bar_curr = np.array([
            init_kep_state[0],
            init_kep_state[1],
            init_kep_state[2],
            init_kep_state[3],
            init_kep_state[4],
            initial_mass
        ])

        mean_keplerian_derivative_handle = lambda t, x: self.mean_keplerian_derivative(t,
                                                                                        x,
                                                                                        a_lambda_a_grid, 
                                                                                        lambda_e_grid)

        # Enter propagation loop
        num_steps = 40 #TODO arbitrary
        logged_data = np.zeros((num_steps+1, 7))
        delta_t = tf/num_steps
        for step_ind in range(num_steps+1):

            logged_data[step_ind,0] = t_curr
            logged_data[step_ind,1:7] = x_bar_curr

            if x_bar_curr[0] < Constants.R_EARTH + 160:
                print(f"WARNING: Current solver iteration targeted an infeasible SMA: {x_bar_curr[0]}.")

            if t_curr >= tf:
                break

            x_bar_next = timestepper_utils.heun_step(mean_keplerian_derivative_handle,
                                                    t_curr,
                                                    x_bar_curr,
                                                    delta_t)
            
            # if x_bar_next[1] <0:
            #     x_bar_next[1] = 0

            # Prepare for next iteration (k -> k+1)
            # print(x_bar_next)
            x_bar_curr = x_bar_next
            t_curr = t_curr + delta_t

        return x_bar_curr, logged_data
    
    def mean_keplerian_derivative(self, 
                                  t: float,
                                  mean_state: np.ndarray,
                                  a_lambda_a_grid: np.ndarray,
                                  lambda_e_grid: np.ndarray):
        """
        Computes each state element's mean time rate of change.

        Arguments:
            - t: time
            - mean_state: Mean slow equinoctal states and mass [sma h k p q m]
            - lambda_a: SMA costate
            - lambda_e: ECC costate
            
        Returns:
            - mean state derivative [sma h k p q m]
        """

        curr_sma = mean_state[0]

        initial_et = spice.utc2et(self.cfg.initial_utc_str)

        # Interpolate costate values
        a_lambda_a = np.interp(curr_sma, self.cfg.sma_grid, a_lambda_a_grid)
        lambda_e = np.interp(curr_sma, self.cfg.sma_grid, lambda_e_grid)
        lambda_a = a_lambda_a / curr_sma

        sma = mean_state[0]
        ecc = mean_state[1]
        inc = mean_state[2]# np.mod(mean_state[2], 2*np.pi)
        raan = mean_state[3]# np.mod(mean_state[3], 2*np.pi)
        aop =  mean_state[4]#np.mod(mean_state[4], 2*np.pi)
        m = mean_state[5]

        # if ecc < 0:
        #     print(f"WARNING: Setting negative eccentricity {ecc} to zero.")
        #     ecc = 0.0
        
        # Generate integration limits
        # NOTE: Need to be careful here, SPICE defines angles as [0,2pi]
        curr_et = initial_et + t
        E_en = np.pi # TODO actually compute it for now just trying to see if solver works for eclipseless case tho
        E_ex = -np.pi
        E_grid = np.linspace(E_ex, E_en, 100) #TODO 100 is arbitrary still
        #TODO assume thruster power and hence thurst is zero when sc in in shadow. If no shadowing conditions exist
        # for a prtcly osculating orbit, then limits are from E_ex=-pi to E_en=pi

        # Compute mean rate of change of orbital elements from thrust and 2-body point mass effects
        # NOTE: We do not integrate over the eclipse arc b/c mean sma, ecc, inc, and mass
        # should not change over that regime
        integrand_grid = np.zeros((len(E_grid),5))
        for ind in range(len(E_grid)):
            E_i = E_grid[ind]
            full_kep_state = np.array([sma, ecc, inc, raan, aop, E_i])
            integrand_grid[ind,:] = self.keplerian_integrand(full_kep_state, lambda_a, lambda_e)
        T = astro_utils.get_orbit_period(sma)
        mean_kep_state_dot = (1/T) * np.trapezoid(integrand_grid, E_grid, axis=0)

        # Compute averaged rates of change from J2 effect
        mean_motion = astro_utils.compute_mean_motion(sma)
        num_j2_term = 3*mean_motion*(Constants.R_EARTH**2)*Constants.J2
        den_j2_term = (sma**2)*(1-ecc**2)**2
        mean_raan_dot_j2_term = ((-num_j2_term)/(2*den_j2_term)) * np.cos(inc)
        mean_aop_dot_j2_term = ((num_j2_term)/(4*den_j2_term)) * (4 - 5 * np.sin(inc)**2)
        # mean_kep_state_dot_j2 = np.array([0,0,0,mean_raan_dot_j2_term,mean_aop_dot_j2_term])
        mean_kep_state_dot_j2 = np.array([0,0,0,0,0])

        total_mean_kep_state_dot = mean_kep_state_dot + mean_kep_state_dot_j2

        # Compute mass derivative
        m_dot = -(self.cfg.A_mag*m) / (self.cfg.spacecraft.Isp * Constants.G0) #TODO

        return np.concatenate((total_mean_kep_state_dot, [m_dot]))

    def keplerian_integrand(self, 
                        kep_state: np.ndarray,
                        lambda_a: float,
                        lambda_e: float) -> np.ndarray:
        """
        Helper callback function for computing the orbit integrand term.

        Arguments:
            - equin_state: [a h k p q F]

        Returns:
            

        References:
            - Eq 5.11, Spacecraft Trajectory Optimization
        """

        # Extract elements
        sma = kep_state[0]
        ecc = kep_state[1]
        inc = np.mod(kep_state[2], 2*np.pi)
        raan = np.mod(kep_state[3], 2*np.pi)
        aop = np.mod(kep_state[4], 2*np.pi)
        E = kep_state[5]

        # if ecc < 0:
        #     print(f"WARNING: Setting negative eccentricity {ecc} to zero.")
        #     ecc = 0.0

        # Intermediate terms
        n = astro_utils.compute_mean_motion(sma)
        ta = astro_utils.eccentric2true(E, ecc)
        r = astro_utils.compute_radius(sma, ecc, ta)
        v = astro_utils.vis_viva(sma, r)
        kep_state_sanitized = np.array([sma, ecc, inc, raan, aop, ta])

        # Compute thrust acceleration components
        alpha = self.compute_thrust_angle(sma,ecc,ta,
                                          lambda_a,lambda_e)
        # print(alpha*(180/np.pi))

        # Conmpute instantaneous state derivative
        slow_state_dot = self.slow_kep_diff_eq(kep_state_sanitized, alpha)

        # Change of variable term dt/dE
        dt_dE = r/(n*sma)

        return slow_state_dot * dt_dE
    
    def slow_kep_diff_eq(self,
                         kep_state: np.ndarray,
                         alpha: float) -> np.ndarray:
        """
        Computes instantaneous keplerian state and mass derivatives using 
        Gauss form of Lagrange's planetary equations and mass flow rate equation.

        Arguments:
            - t: time
            - kep_state: keplerian state [sma ecc inc raan aop ta]
            - A_mag: acceleration perturbation magnitude
            - alpha: in plane thrust pitch angle
        
        Returns:
            - np.ndarray: slow kep state derivatives [sma ecc inc raan aop]
        """

        
        # Extract keplerian elements
        sma = kep_state[0]
        ecc = kep_state[1]
        if ecc < 0:
            raise ValueError("Eccentricity is negative.")
        inc = np.mod(kep_state[2], 2*np.pi)
        # raan = full_state[3]
        aop = np.mod(kep_state[4], 2*np.pi)
        ea = kep_state[5]
        ta = astro_utils.eccentric2true(ea, ecc)

        # Extract thrust components in NTH frame
        a_nth = self.compute_nth_thrust_components(alpha)
        a_n = a_nth[0]
        a_t = a_nth[1]
        a_h = a_nth[2]

        r = astro_utils.compute_radius(sma, ecc, ta)
        v = astro_utils.vis_viva(sma, r)
        h = astro_utils.compute_ang_mom_norm(sma, ecc)

        sma_dot = (2*(sma**2)*v)/(Constants.EARTH_MU) * a_t
        ecc_dot = (1/v) * ( 2*(ecc + np.cos(ta))*a_t + (r/sma)*a_n*np.sin(ta) )
        inc_dot = (r/h) * a_h * np.cos(aop + ta)

        # RAAN not defined at zero inclination
        if inc != 0:
            raan_dot = (r/(h*np.sin(inc))) * a_h * np.sin(aop + ta)
        else:
            raan_dot = 0.0

        # AOP not defined at zero eccentricity
        if ecc > 0:
            aop_dot = (1/(ecc*v)) * ( 2*a_t*np.sin(ta) - (2*ecc+(r/sma)*np.cos(ta))*a_n ) - (r/(h*np.sin(inc))) * a_h * np.sin(aop + ta) * np.cos(inc)
        else:
            aop_dot = 0.0

        return np.array([sma_dot, ecc_dot, inc_dot, raan_dot, aop_dot])

    def compute_thrust_angle(self, 
                             sma: float,
                             ecc: float,
                             nu: float,
                             lambda_a: float, 
                             lambda_e: float) -> float:
        """
        Compute optimal feeback thrust angle given current state and costate.

        Alpha is defined as the angle from the velocity vector to projection
        of the thrust vector onto the orbital plane.

        Arguments:
            - sma: semi-major axis (km)
            - ecc: eccentricity
            - r: orbit radius (km)
            - v: orbit velocity (km)
            - nu: true anomaly (rad)
            - lambda_a: sma costate grid
            - lambda_e: ecc costate grid
        """

        r = astro_utils.compute_radius(sma, ecc, nu)
        v = astro_utils.vis_viva(sma, r)

        sin_alpha_num = -lambda_e * (r/sma) * np.sin(nu)
        cos_alpha_num = -2 * ( lambda_a * ((sma**2)*(v**2))/(Constants.EARTH_MU) + lambda_e * (ecc + np.cos(nu)) )
        den1 = lambda_a*((sma**2 * v**2)/Constants.EARTH_MU) + lambda_e*(ecc+np.cos(nu))
        den2 = lambda_e * (r/sma) * np.sin(nu)
        denominator = np.sqrt( 4*den1**2 + den2**2 )
        
        sin_alpha = sin_alpha_num / denominator
        cos_alpha = cos_alpha_num / denominator
        alpha = np.arctan2(sin_alpha, cos_alpha)
        return alpha
        # return np.asin(sin_alpha)

    def compute_nth_thrust_components(self,
                                      alpha: float) -> np.ndarray:
        """
        Compute thrust acceleration vector in the NTH frame.

        Currently only restricted to in-plane thrust vectors (i.e. no inc change)

        Arguments:
            - alpha: In plane pitch thrust steering angle (rad)
        """
        return self.cfg.A_mag * np.array([np.sin(alpha),
                                    np.cos(alpha),
                                    0]) # [a_n a_t a_h]