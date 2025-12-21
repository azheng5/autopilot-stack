import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spiceypy as spice
from scipy.optimize import minimize

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
                            num_costate_nodes: int):
        """
        Generates a feedback optimal control law
        """

        sma_grid = np.linspace(initial_kep_state[0], target_sma, num_costate_nodes)

        # TODO make a better guess...
        a_lambda_a_grid_guess = np.zeros(num_costate_nodes)
        lambda_e_grid_guess = np.zeros(num_costate_nodes)
        lambda_i_grid_guess = np.zeros(num_costate_nodes)
        tf_guess = 100

        initial_guess = np.concatenate([
            a_lambda_a_grid_guess,
            lambda_e_grid_guess,
            lambda_i_grid_guess,
            np.array([tf_guess])
        ])

        # Define bounds on orbital elements and final time
        a_lambda_a_bounds = [(-5,0)]*num_costate_nodes
        lambda_e_bounds = [(-5,5)]*num_costate_nodes
        lambda_i_bounds = [(0,0)]*num_costate_nodes
        tf_bounds = (1,60*60*24*365)
        bounds = [*a_lambda_a_bounds,
                  *lambda_e_bounds,
                  *lambda_i_bounds,
                  tf_bounds]
        
        # Construct terminal constraints denoting final target SMA
        terminal_constraints_handle = lambda z: self.terminal_constraints(z,
                                                                        initial_kep_state,
                                                                        initial_mass,
                                                                        initial_utc_str,
                                                                        sma_grid)

        # Run minimizer
        opt_result = minimize(
            fun= self.objective,
            x0= initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints={
                "type": "eq",
                "fun": terminal_constraints_handle
            },
            options={
                "disp": True
            }
        )

        a_lambda_a_grid = opt_result.x[0:num_costate_nodes]
        lambda_e_grid = opt_result.x[num_costate_nodes:2*num_costate_nodes]
        lambda_i_grid = opt_result.x[2*num_costate_nodes:3*num_costate_nodes]
        tf = opt_result.x[-1]

        #TODO create a lambda for the optimal control law
        control_law_handle = lambda kep_state: self.control_law_skeleton(kep_state)

        return control_law_handle
    
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
        # A_mag_kN = 

        # a_thurst = self.compute_nth_thrust_components(A_mag_kN, alpha, beta)

        # return

    def objective(self, z: np.ndarray) -> float:
        """Performance index specifying minimum time"""
        return float(z[-1])
    
    def terminal_constraints(self,
                             z: np.ndarray,
                             initial_kep_state: np.ndarray,
                             initial_mass: float,
                             initial_utc_str: str,
                             sma_grid: np.ndarray) -> np.ndarray:
        """
        Generates terminal constraints
        """

        num_costate_nodes = len(sma_grid)

        # Extract decision variables
        a_lambda_a_grid = z[0:num_costate_nodes]
        lambda_e_grid = z[num_costate_nodes:2*num_costate_nodes]
        lambda_i_grid = z[2*num_costate_nodes:3*num_costate_nodes]
        tf = z[-1]

        final_x_bar = self.orbit_averaged_propagation(initial_kep_state,
                                                      initial_mass,
                                                      initial_utc_str,
                                                      sma_grid,
                                                      a_lambda_a_grid,
                                                      lambda_e_grid,
                                                      lambda_i_grid,
                                                      tf)

        final_sma = final_x_bar[0]
        return np.array([final_sma - sma_grid[-1]])
    
    def orbit_averaged_propagation(self,
                                   init_kep_state: np.ndarray,
                                   initial_mass: float,
                                   initial_utc_string: str,
                                   sma_grid: np.ndarray,
                                   a_lambda_a_grid: np.ndarray,
                                   lambda_e_grid: np.ndarray,
                                   lambda_i_grid: np.ndarray,
                                   tf: float):
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
        """
        
        if not (len(sma_grid) == len(a_lambda_a_grid) == len(lambda_e_grid) == len(lambda_i_grid)):
            raise ValueError("Costate/SMA grid lengths have mismatched lengths.")

        # Initialize propagation
        initial_et = spice.utc2et(initial_utc_string)
        t_curr = 0.0 # seconds since start time
        x_bar_curr = np.array([
            init_kep_state[0], #sma
            init_kep_state[1], #ecc
            init_kep_state[2], #inc
            init_kep_state[3], #raan
            init_kep_state[4], #aop
            initial_mass])

        # Enter propagation loop
        num_steps = 40 #TODO arbitrary
        delta_t = tf/num_steps
        for step_ind in range(num_steps):

            curr_sma = x_bar_curr[0]
            
            # Interpolate costate values
            a_lambda_a = np.interp(curr_sma, sma_grid, a_lambda_a_grid)
            lambda_e = np.interp(curr_sma, sma_grid, lambda_e_grid)
            lambda_i = np.interp(curr_sma, sma_grid, lambda_i_grid)
            lambda_a = a_lambda_a / curr_sma
            
            orbit_averaged_derivative_handle = lambda t, x: self.orbit_averaged_derivative(t,
                                                                                           x, 
                                                                                           lambda_a, 
                                                                                           lambda_e, 
                                                                                           lambda_i,
                                                                                           initial_et)

            x_bar_next = timestepper_utils.heun_step(orbit_averaged_derivative_handle,
                                                    t_curr,
                                                    x_bar_curr,
                                                    delta_t)
            
            # Prepare for next iteration (k -> k+1)
            x_bar_curr = x_bar_next
            t_curr = t_curr + delta_t

        return x_bar_curr
    
    def orbit_averaged_derivative(self, 
                                  t: float,
                                  x_bar: np.ndarray,
                                  lambda_a: float,
                                  lambda_e: float,
                                  lambda_i: float,
                                  initial_et: float) -> np.ndarray:
        """
        Computes each element's mean time rate of change by calculating the 
        incremental change in an orbital element over a single revolution 
        and dividing by its orbital period.

        Arguments:
            - t: time
            - x_bar: mean state vector, consisting of slow elements and mass only [sma ecc inc raan aop m]
            - lambda_a
            
        Returns:
            -x_bar_dot: mean state derivative
        """

        # Extract elements
        sma = x_bar[0]
        ecc = x_bar[1]
        inc = x_bar[2]
        # raan = x_bar[3]
        # aop = x_bar[4]
        # m = x_bar[5]
        
        # Generate integration limits
        # NOTE: Need to be careful here, SPICE defines angles as [0,2pi]
        curr_et = initial_et + t
        E_en = np.pi # TODO actually compute it for now just trying to see if solver works for eclipseless case tho
        E_ex = -np.pi
        E_grid = np.linspace(E_ex, E_en, 100)
        #TODO assume thruster power and hence thurst is zero when sc in in shadow. If no shadowing conditions exist
        # for a prtcly osculating orbit, then limits are from E_ex=-pi to E_en=pi

        integrand_grid = np.zeros((len(E_grid),6))
        for ind in range(len(E_grid)):
            E_i = E_grid[ind]
            integrand_grid[ind,:] = self.orbit_integrand(E_i, x_bar, lambda_a, lambda_e, lambda_i)


        # Compute integral term, representing the total change in orbital elements over the
        # thrust arc in a single revolution
        # integrates vertically
        integral_value = np.trapezoid(integrand_grid, E_grid, axis=0)

        # Compute mean rate of change of orbital elements from thrust and 2-body point mass effects
        Tp = astro_utils.get_orbit_period(sma)
        x_bar_dot = (1/Tp) * integral_value

        # Compute averaged rates of change for RAAN and AOP from oblateness effect
        mean_motion = astro_utils.compute_mean_motion(sma)
        num_j2_term = 3*mean_motion*(Constants.R_EARTH**2)*Constants.J2
        den_j2_term = (sma**2)*(1-ecc**2)**2
        mean_raan_dot_j2_term = ((-num_j2_term)/(2*den_j2_term)) * np.cos(inc)
        mean_aop_dot_j2_term = ((num_j2_term)/(4*den_j2_term)) * (4 - 5 * np.sin(inc)**2)
        x_bar_j2_dot = np.array([0,0,0,mean_raan_dot_j2_term,mean_aop_dot_j2_term,0])

        return x_bar_dot + x_bar_j2_dot

    def orbit_integrand(self, 
                        E: float, 
                        x_bar: np.ndarray,
                        lambda_a: float,
                        lambda_e: float,
                        lambda_i: float) -> np.ndarray:
        """
        Helper callback function for computing the orbit integrand term.

        Arguments:
            - E: Eccentric anomaly (rad)
            - x_bar: Averaged incomplete state [sma ecc inc raan aop m]

        References:
            - Eq 5.11, Spacecraft Trajectory Optimization
        """

        # Extract elements
        sma = x_bar[0]
        ecc = x_bar[1]
        inc = x_bar[2]
        raan = x_bar[3]
        aop = x_bar[4]
        m = x_bar[5]

        # Compute intermediate terms
        mean_motion = astro_utils.compute_mean_motion(sma)
        ta = astro_utils.eccentric2true(E, x_bar[1])
        r = astro_utils.compute_radius(sma, ecc, ta)
        v = astro_utils.vis_viva(sma, r)

        # Compute thrust acceleration components
        alpha = self.compute_thrust_angle(sma,ecc,r,v,ta,
                                          lambda_a,lambda_e,lambda_i)
        beta = 0 #TODO currently yaw control not supported
        A_mag = 0.1 #TODO hardcoded rn, fix it
        A_mag_kN = A_mag * 1e-3
        a_thrust = self.compute_nth_thrust_components(A_mag_kN, alpha, beta)
        
        # Construct full state
        full_state = np.array([sma,ecc,inc,raan,aop,ta,m])
        
        # Conmpute instantaneous state derivative
        full_state_dot = self.keplerian_diff_eq(full_state, a_thrust)

        # Extract out incomplete state
        x_dot = np.array([full_state_dot[0], #sma_dot
                            full_state_dot[1], #ecc_dot
                            full_state_dot[2], #inc_dot
                            full_state_dot[3], #raan_dot
                            full_state_dot[4], #aop_dot
                            full_state_dot[6]]) #mass_dot

        # Change of variable term dt/dE
        dt_dE = r/(mean_motion*sma)

        return x_dot * dt_dE


    def keplerian_diff_eq(self,
                          full_state: np.ndarray,
                          a_pert: np.ndarray) -> np.ndarray:
        """
        Computes instantaneous keplerian state and mass derivatives using 
        Gauss form of Lagrange's planetary equations and mass flow rate equation.

        Arguments:
            - t: time
            - full_state: keplerian state and mass [sma ecc inc raan aop ta m]
            - a_pert: perturbing accelerations in NTH frame [a_n a_t a_h]
        
        Returns:
            - np.ndarray: full state derivative [sma ecc inc raan aop ta m]
        """
        
        # Extract keplerian elements and mass
        sma = full_state[0]
        ecc = full_state[1]
        inc = full_state[2]
        # raan = full_state[3]
        aop = full_state[4]
        ta = full_state[5]
        # m = full_state[6]

        # Extract thrust components in NTH frame
        a_n = a_pert[0]
        a_t = a_pert[1]
        a_h = a_pert[2]

        r = astro_utils.compute_radius(sma, ecc, ta)
        v = astro_utils.vis_viva(sma, r)
        h = astro_utils.compute_ang_mom_norm(sma, ecc)

        sma_dot = (2*(sma**2)*v)/(Constants.EARTH_MU) * a_t
        ecc_dot = (1/v) * ( 2*(ecc + np.cos(ta))*a_t + (r/sma)*a_n*np.sin(ta) )
        inc_dot = (r/h) * a_h * np.cos(aop + ta)
        raan_dot = (r/(h*np.sin(inc))) * a_h * np.sin(aop + ta)
        aop_dot = (1/(ecc*v)) * ( 2*a_t*np.sin(ta) - (2*ecc+(r/sma)*np.cos(ta))*a_n ) - (r/(h*np.sin(inc))) * a_h * np.sin(aop + ta) * np.cos(inc)
        ta_dot = (h/(r**2)) - (1/(ecc*v)) * ( 2*a_t*np.sin(ta) - (2*ecc+(r/sma)*np.cos(ta))*a_n )
        # m_dot = (-2*thruster_efficiency*input_power)/((Constants.G0*Isp)**2) #TODO uncomment
        m_dot = 0
        
        return np.array([sma_dot, ecc_dot, inc_dot, raan_dot, aop_dot, ta_dot, m_dot])

    def compute_thrust_angle(self, 
                             sma: float,
                             ecc: float,
                             r: float,
                             v: float,
                             nu: float,
                             lambda_a: float, 
                             lambda_e: float,
                             lambda_i: float) -> float:
        """
        Compute optimal feeback thrust angle given current state and costate.

        Arguments:
            - sma: semi-major axis (km)
            - ecc: eccentricity
            - r: orbit radius (km)
            - v: orbit velocity (km)
            - nu: true anomaly (rad)
            - lambda_a: sma costate grid
            - lambda_e: ecc costate grid
            - lambda_i: inc costate grid
        """

        sin_alpha_num = -lambda_e * (r/sma) * np.sin(nu)
        cos_alpha_num = -2 * ( lambda_a * ((sma**2)*(v**2))/(Constants.EARTH_MU) + lambda_e * (ecc + np.cos(nu)) )
        den1 = lambda_a*((sma**2 * v**2)/Constants.EARTH_MU) + lambda_e*(ecc+np.cos(nu))
        den2 = lambda_e * (r/sma) * np.sin(nu)
        denominator = np.sqrt( 4*den1**2 + den2**2 )

        if denominator <= 1e-12:
            return 0.0
        
        sin_alpha = sin_alpha_num / denominator
        cos_alpha = cos_alpha_num / denominator

        return np.arcsin(sin_alpha)
    
    def compute_nth_thrust_components(self,
                                      A_mag_kN: float,
                                      alpha: float,
                                      beta: float) -> np.ndarray:
        return A_mag_kN * np.array([np.sin(alpha)*np.cos(beta),
                                    np.cos(alpha)*np.cos(beta),
                                    np.sin(beta)]) # [a_n a_t a_h]
    