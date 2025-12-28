import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spiceypy as spice
from scipy.optimize import minimize
from typing import Tuple

sys.path.append(str(Path(__file__).parent))
import flight_dynamics.timestepper_utils as timestepper_utils
from flight_dynamics import astro_utils
from flight_dynamics import Constants
from flight_dynamics import time_utils
from flight_dynamics.Eclipse import Eclipse
from flight_dynamics.Propagator import Propagator, PropagatorTerminator
from flight_dynamics.OrbitLogger import OrbitLogger
from flight_dynamics.Spacecraft import Spacecraft

class DirectSolver:
    """
    Applies a direct optimization technique to compute low thrust trajectories.

    References:
        - Spacecraft Trajectory Optmization
    """
    
    def __init__(self, spacecraft: Spacecraft) -> None:
        self.spacecraft = spacecraft

    # def solve():
        
    #     control_law_handle = self.generate_control_law()

    #     #TODO modify propagator to be able to take in control laws
    #     propagator = Propagator(self.spacecraft)

    def generate_control_law(self, 
                            initial_kep_state: np.ndarray, 
                            initial_mass: float,
                            initial_utc_str: str,
                            target_sma: float,
                            target_ecc: float,
                            num_costate_nodes: int,
                            A_mag: float,
                            elements: str = "equinoctial",
                            perf_ind: str = "min_time"):
        """
        Generates a feedback optimal control law to drive low thrust spacecraft from an
        initial keplerian state to a final target SMA and ECC

        Can choose to use either equinoctial or keplerian elements to solve.
        
        Arguments:
            - initial_kep_state: [sma ecc inc raan aop ta]
            - initial_mass: Initial spacecraft mass (kg)
            - initial_utc_str: Initial UTC string format (Constants.UTC_FORMAT)
            - target_sma: Target SMA (km)
            - num_costate_nodes: Resolution of costate grid
            - A_mag: thrust acceleration magnitude (km/s**2)
        """

        if initial_kep_state[0] < Constants.R_EARTH:
            raise ValueError(f"SMA is lower than Earth radius: {initial_kep_state[0]}")

        if initial_kep_state[1] < 0:
            raise ValueError(f"Eccentricity is negative: {initial_kep_state[1]}")
        
        #TOOD write this logic
        if elements == "equinoctial":
            pass
        elif elements == "classical":
            pass
        else:
            raise ValueError("Invalid orbital elements classification `elements` specified.")
        
        initial_E = astro_utils.true2eccentric(initial_kep_state[-1], initial_kep_state[1])
        initial_equin_state = astro_utils.classical_to_equinoctial(
            np.concatenate((initial_kep_state[0:5],[initial_E]))
        )

        sma_grid = np.linspace(initial_kep_state[0], target_sma, num_costate_nodes)

        # Generate initial guess for decision vector and absolute final time guess
        initial_guess, ref_tf = self.generate_initial_guess(initial_kep_state, 
                                                    initial_mass, 
                                                    target_sma, 
                                                    num_costate_nodes, 
                                                    A_mag)
        
        if perf_ind == "min_time":
            objective_handle = lambda z: self.minimum_time_objective(z, ref_tf)
        elif perf_ind == "min_fuel":
            objective_handle = self.minimum_fuel_objective
        else:
            raise ValueError("Invalid performance index `perf_ind` specified.")

        # Define bounds on orbital elements and final time
        # When lambda_a and lambda_e are both 0, the 
        # optimal thrust angle equation hits a singularity,
        # so numerically we avoid using 0.
        a_lambda_a_bounds = [(-5,-0.0001)]*num_costate_nodes
        lambda_e_bounds = [(-0.01,1.0)]*num_costate_nodes
        lambda_i_bounds = [(0,0)]*num_costate_nodes
        rel_tf_bounds = (0.0,100.0)
        bounds = [*a_lambda_a_bounds,
                  *lambda_e_bounds,
                  *lambda_i_bounds,
                  rel_tf_bounds]
        
        # Construct terminal constraints denoting final target SMA
        # terminal_eq_handle = lambda z: self.terminal_constraints(z,
        #                                                         initial_equin_state,
        #                                                         initial_mass,
        #                                                         initial_utc_str,
        #                                                         sma_grid,
        #                                                         target_ecc,
        #                                                         A_mag)["eq_constraint"]
        terminal_ineq_handle = lambda z: self.terminal_constraints(z,
                                                                initial_equin_state,
                                                                initial_mass,
                                                                initial_utc_str,
                                                                sma_grid,
                                                                target_ecc,
                                                                A_mag,
                                                                ref_tf)["ineq_constraint"]

        # Define SLSQP solver configuration
        slsqp_options = {
                "max_iter": 10,
                "disp": True,
                "ftol": 1e-6,
            }

        # Run minimizer
        opt_result = minimize(
            fun= objective_handle,
            x0= initial_guess,
            method="SLSQP", #sequential least squares programming
            bounds=bounds,
            constraints=[
                # {"type": "eq","fun": terminal_eq_handle},
                {"type": "ineq","fun": terminal_ineq_handle}
            ],
            callback=self.optimizer_callback,
            options=slsqp_options
        )

        #TODO put all this stuff in a txt file output
        print("-------------------- SOLVER COMPLETED --------------------")
        a_lambda_a_grid, lambda_e_grid, lambda_i_grid, rel_tf = self.unpack_z(opt_result.x)

        final_eq_x_bar, _ = self.mean_equinoctial_propagation(opt_result.x,
                                                            initial_equin_state,
                                                            initial_mass,
                                                            initial_utc_str,
                                                            sma_grid,
                                                            A_mag,
                                                            ref_tf)
        final_mean_equin_state = final_eq_x_bar[0:5]
        final_mass = final_eq_x_bar[5]
        final_mean_kep_state = astro_utils.equinoctial_to_classical(np.concatenate((final_mean_equin_state,[0.0])))
        print(f"Final SMA: {final_mean_kep_state[0]}")
        print(f"Final ECC: {final_mean_kep_state[1]}")
        print(f"Final INC: {final_mean_kep_state[2]}")
        print(f"Final RAAN: {final_mean_kep_state[3]}")
        print(f"Final AOP: {final_mean_kep_state[4]}")
        print(f"Final mass: {final_mass}")
        print(f"Final time: {rel_tf*ref_tf}")
        print(f"Relative final time: {rel_tf}")
        print(f"SMA constraint: {self.sma_constraint(final_mean_kep_state[0], target_sma)}")
        print(f"Eccentricity constraint: {self.ecc_constraint(final_mean_kep_state[1], target_ecc)}")

        #TODO create a lambda for the optimal control law
        control_law_handle = lambda kep_state: self.control_law_skeleton(kep_state)

        return control_law_handle

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
    
    def control_law_skeleton(self, kep_state: np.ndarray):
        return None

        # sma = kep_state[0]
        # ecc = kep_state[1]
        # ta = kep_state[5]
        # r = astro_utils.compute_radius(sma, ecc, ta)
        # v = astro_utils.vis_viva(sma, r)

        # lambda_a = 
        # lambda_e = 
        # lambda_i = 


        # alpha = self.compute_thrust_angle(self, sma, ecc, r, v, ta, lambda_a, lambda_e, lambda_i)
        # beta = 0
        # A_mag = 

        # a_thurst = self.compute_nth_thrust_components(A_mag, alpha, beta)

        # return

    def generate_initial_guess(self, 
                               initial_kep_state: np.ndarray,
                               initial_mass: float,
                               target_sma: float,
                               num_costate_nodes: int,
                               A_mag: float) -> Tuple[np.ndarray, float]:
        """
        Generates costate grid and final time guesses by assuming
        thrust follows a near-tangent thrusting profile over 
        quasi-circular orbit.

        References:
            - Edelbaum, Propulsion Requirements for Controllable Satellites
        """

        # Combo of a_lambda_a=-1 and low ecc is near-tangent thrusting
        a_lambda_a_grid_guess = -1*np.ones(num_costate_nodes)

        # Keep ecc costate low and near zero
        lambda_e_grid_guess = 0.01*np.ones(num_costate_nodes)

        #TODO defaulted to zero for now
        lambda_i_grid_guess = np.zeros(num_costate_nodes)
       
        # Generate final time guess using rocket equation, mfr,
        # quasi-circular tangent constant thrust assumptions
        sma = initial_kep_state[0]
        ecc = initial_kep_state[1]
        ta = initial_kep_state[5]
        r0 = astro_utils.compute_radius(sma, ecc, ta)
        v0 = astro_utils.vis_viva(sma, r0)
        delta_v = 0.5 * ((target_sma - sma)/sma) * v0
        T = A_mag*initial_mass
        ve = self.spacecraft.Isp * Constants.G0
        tf = ((initial_mass*ve)/T) * (1 - np.exp(-delta_v/ve))
        # Scale by factor of 1.2 to account for eclipse times
        #minor TODO 1.2 is arbitrary, scaling could be more accurate 
        # by getting eclipse arc time of initial orbit
        tf_guess = 1.2 * tf

        rel_tf_guess = 1

        return np.concatenate([
            a_lambda_a_grid_guess,
            lambda_e_grid_guess,
            lambda_i_grid_guess,
            np.array([rel_tf_guess])
        ]), tf_guess

    def minimum_time_objective(self, z: np.ndarray, ref_tf: float) -> float:
        """
        Performance index specifed to achieve minimum time. 
        Normalizes flight time by some reference time (i.e 
        initial guess for the solved time) so that returned
        value of the objective function is OOM ~1
        """
        return float(z[-1]) / ref_tf
    
    def minimum_fuel_objective(self, z: np.ndarray) -> float:
        """Performance index specifying minimum time"""
        #TODO make it -m(tf), but they should be nearly identical, but comp time doubels....
        #TODO this function still not usable
        return 0.0
    
    def sma_constraint(self, final_sma: float, target_sma: float) -> float:
        sma_relative_error_tol = 1e-5
        sma_relative_error = sma_relative_error_tol - abs((final_sma - target_sma)/target_sma)
        return sma_relative_error
    
    def ecc_constraint(self, final_ecc: float, target_ecc: float) -> float:
        return 1e-3 - abs(final_ecc - target_ecc)

    def terminal_constraints(self,
                             z: np.ndarray,
                             initial_equin_state: np.ndarray,
                             initial_mass: float,
                             initial_utc_str: str,
                             sma_grid: np.ndarray,
                             target_ecc: float,
                             A_mag: float,
                             ref_tf):
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
        a_lambda_a_grid, lambda_e_grid, lambda_i_grid, rel_tf = self.unpack_z(z)

        # initial_kep_state = astro_utils.equinoctial_to_classical(initial_equin_state)

        final_eq_x_bar, _ = self.mean_equinoctial_propagation(z,
                                                                initial_equin_state,
                                                                initial_mass,
                                                                initial_utc_str,
                                                                sma_grid,
                                                                A_mag,
                                                                ref_tf)

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
        sma_relative_error = self.sma_constraint(final_sma, sma_grid[-1])
        # ecc_relative_error_tol = 0.01
        # ecc_relative_error = ecc_relative_error_tol - abs((final_ecc - target_ecc)/target_ecc)
        ecc_target = self.ecc_constraint(final_ecc, target_ecc)
        elliptic_constraint = 1- final_mean_equin_state[1]**2 + final_mean_equin_state[2]**2
        negative_ecc_constraint = final_ecc
        lower_sma_constraint = initial_equin_state[0]-10
        dry_mass = self.spacecraft.dry_mass #TODO fix this
        mass_constraint = final_mass - dry_mass

        # print(sma_target)

        # constraint_dict = {
        #     "eq_constraint": np.array([sma_constraint, ecc_constraint]),
        # }

        # constraint_dict = {
        #     "eq_constraint": np.array([]),
        #     "ineq_constraint": np.array([elliptic_constraint,
        #                                  negative_ecc_constraint,
        #                                  mass_constraint,
        #                                  sma_target,
        #                                  lower_sma_constraint]),
        # }
        constraint_dict = {
            "eq_constraint": np.array([]),
            "ineq_constraint": np.array([
                sma_relative_error,
                mass_constraint,
                ecc_target,
            ]),
        }

        print(
            f"SMA: {final_sma:12.6f} | "
            f"ECC: {final_ecc:12.6f} | "
            f"INC: {final_inc:12.6f} | "
            f"RAAN: {final_raan:12.6f} | "
            f"AOP: {final_aop:12.6f} | "
            f"MASS: {final_mass:12.6f} | "
            f"SMA_REL: {sma_relative_error:12.6f} | "
            f"TF: {rel_tf*ref_tf:12.6f}"
        )

        return constraint_dict
    
    def optimizer_callback(self,z):
        print("Iteration")
    
    def mean_equinoctial_propagation(self,
                                    z: np.ndarray,
                                   init_equin_state: np.ndarray,
                                   initial_mass: float,
                                   initial_utc_string: str,
                                   sma_grid: np.ndarray,
                                   A_mag: float,
                                   ref_tf: float):
        """
        Given initial state and grid of costates variables,
        propagates averaged state to final time.

        Arguments:
            - init_equin_state: Initial equinoctial state
            - initial_mass: Initial mass (kg)
            - initial_utc_string: Initial UTC string
            - sma_grid: Grid of SMA values from initial SMA to target SMA
            - a_lambda_a_grid: Grid of a*lambda_a values
            - lambda_e_grid: Grid of lambda_e_values
            - lambda_i_grid: Grid of lambda_i values
            - tf: Final time

        Returns
        """
        a_lambda_a_grid, lambda_e_grid, lambda_i_grid, rel_tf = self.unpack_z(z)
        tf = rel_tf * ref_tf

        # Initialize propagation
        initial_et = spice.utc2et(initial_utc_string)
        t_curr = 0.0 # seconds since start time
        x_bar_curr = np.array([
            init_equin_state[0],
            init_equin_state[1],
            init_equin_state[2],
            init_equin_state[3],
            init_equin_state[4],
            initial_mass
        ])

        mean_equinoctial_derivative_handle = lambda t, x: self.mean_equinoctial_derivative(t,
                                                                                           x, 
                                                                                           A_mag,
                                                                                           sma_grid,
                                                                                           a_lambda_a_grid, 
                                                                                           lambda_e_grid,
                                                                                           initial_et)

        
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

            x_bar_next = timestepper_utils.heun_step(mean_equinoctial_derivative_handle,
                                                    t_curr,
                                                    x_bar_curr,
                                                    delta_t)
            

            # Prepare for next iteration (k -> k+1)
            x_bar_curr = x_bar_next
            t_curr = t_curr + delta_t

        

        return x_bar_curr, logged_data
    
    def mean_equinoctial_derivative(self, 
                                  t: float,
                                  mean_state: np.ndarray,
                                  A_mag: float,
                                  sma_grid: np.ndarray,
                                  a_lambda_a_grid: np.ndarray,
                                  lambda_e_grid: np.ndarray,
                                  initial_et: float) -> np.ndarray:
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

        # Interpolate costate values
        a_lambda_a = np.interp(curr_sma, sma_grid, a_lambda_a_grid)
        lambda_e = np.interp(curr_sma, sma_grid, lambda_e_grid)
        lambda_a = a_lambda_a / curr_sma

        mean_equin_state = mean_state[0:5]
        
        # Generate integration limits
        # NOTE: Need to be careful here, SPICE defines angles as [0,2pi]
        curr_et = initial_et + t
        F_en = np.pi # TODO actually compute it for now just trying to see if solver works for eclipseless case tho
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
            integrand_grid[ind,:] = self.equinoctial_integrand(full_equin_state, A_mag, lambda_a, lambda_e)
        mean_equin_state_dot = np.trapezoid(integrand_grid, F_grid, axis=0)

        # Compute averaged rates of change from J2 effect
        # mean_equin_state_dot_j2 = self.compute_equinoctial_j2_rate(mean_equin_state)
        mean_equin_state_dot_j2 = np.array([0,0,0,0,0])

        total_mean_equin_state_dot = mean_equin_state_dot + mean_equin_state_dot_j2

        # Compute mass derivative
        m_dot = -(A_mag*m) / (self.spacecraft.Isp * Constants.G0) #TODO

        return np.concatenate((total_mean_equin_state_dot, [m_dot]))

    def equinoctial_integrand(self, 
                            equin_state: np.ndarray,
                            A_mag: float,
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
        
        # Conmpute instantaneous state derivative
        slow_state_dot = self.slow_equinoctial_diff_eq(equin_state, A_mag, alpha)

        # Change of variable term: (dt/dF)_T
        dt_dF_T = (1 - k * np.cos(F) - h * np.sin(F) ) / (2*np.pi)

        return slow_state_dot * dt_dF_T

    def slow_equinoctial_diff_eq(self,
                                equin_state: np.ndarray,
                                A_mag: float,
                                alpha: float,) -> np.ndarray:
        """
        Compute slow equinoctial state derivatives using the full 
        equinoctial state.

        Arguments:
            - equin_state: [a h k p q F]
            - A_mag: acceleration perturbation magnitude
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
        a_nth = self.compute_nth_thrust_components(A_mag, alpha)
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
                                                                                        A_mag,
                                                                                        sma_grid,
                                                                                        a_lambda_a_grid, 
                                                                                        lambda_e_grid,
                                                                                        initial_et)

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
                                  A_mag: float,
                                  sma_grid: np.ndarray,
                                  a_lambda_a_grid: np.ndarray,
                                  lambda_e_grid: np.ndarray,
                                  initial_et: float):
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

        # Interpolate costate values
        a_lambda_a = np.interp(curr_sma, sma_grid, a_lambda_a_grid)
        lambda_e = np.interp(curr_sma, sma_grid, lambda_e_grid)
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
            integrand_grid[ind,:] = self.keplerian_integrand(full_kep_state, A_mag, lambda_a, lambda_e)
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
        m_dot = -(A_mag*m) / (self.spacecraft.Isp * Constants.G0) #TODO

        return np.concatenate((total_mean_kep_state_dot, [m_dot]))

    def keplerian_integrand(self, 
                        kep_state: np.ndarray,
                        A_mag: float,
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
        slow_state_dot = self.slow_kep_diff_eq(kep_state_sanitized, A_mag, alpha)

        # Change of variable term dt/dE
        dt_dE = r/(n*sma)

        return slow_state_dot * dt_dE
    
    def slow_kep_diff_eq(self,
                         kep_state: np.ndarray,
                         A_mag: float,
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
        a_nth = self.compute_nth_thrust_components(A_mag, alpha)
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
                                      A_mag: float,
                                      alpha: float) -> np.ndarray:
        """
        Compute thrust acceleration vector in the NTH frame.

        Currently only restricted to in-plane thrust vectors (i.e. no inc change)

        Arguments:
            - A_mag: Acceleration magnitude
            - alpha: In plane pitch thrust steering angle (rad)
        """
        return A_mag * np.array([np.sin(alpha),
                                    np.cos(alpha),
                                    0]) # [a_n a_t a_h]