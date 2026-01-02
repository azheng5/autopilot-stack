import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

sys.path.append(str(Path(__file__).parent))
from flight_dynamics import astro_utils
from flight_dynamics import Constants

spice.furnsh("meta_kernel.tm")

"""
Astrodynamics tool for determining eclipse statuses of Earth-orbiting spacecraft.

References:
    - Vallado, Fundamentals of Astrodynamics and Applications 4th Edition, Ch 5.3.2
    - Gonzalez, https://www.youtube.com/watch?v=LeJpWfyXJPw
    - Longo and Rickman, https://ntrs.nasa.gov/api/citations/19950023025/downloads/19950023025.pdf
"""

def check_eclipse(r_sc: np.ndarray, curr_utc_str: str) -> int:
    """
    Check if spacecraft location is in eclipse at a given moment in time
    
    Arguments:
        - r_sc: Spacecraft position vector in Earth-centered J2000 (km)
        - curr_utc_str: Current UTC time

    Returns:
        - bool: 0 (no eclipse), 1 (penumbra), 2 (umbra)
    """

    # Position of earth relative to sun in J2000 frame
    r_earth_sun = get_earth_sun_vector(curr_utc_str)
    r_earth_sun_norm = np.linalg.norm(r_earth_sun)
    r_earth_sun_hat = r_earth_sun / r_earth_sun_norm

    # Compute projection and rejection vectors of satellite position
    r_proj_scalar = np.dot(r_sc, r_earth_sun_hat)
    # Spacecraft is on terminator plane facing the sun, which never has eclipse
    if r_proj_scalar <= 0:
        return 0
    
    r_proj = r_proj_scalar * r_earth_sun_hat
    r_proj_norm = np.linalg.norm(r_proj)
    r_rej = r_sc - r_proj
    r_rej_norm = np.linalg.norm(r_rej)

    # Check if satellite in umbra
    Ki_u = (Constants.D_EARTH * r_earth_sun_norm) / (Constants.D_SUN - Constants.D_EARTH)
    alpha_u = np.arcsin(Constants.D_EARTH / (2* Ki_u))
    zeta =  (Ki_u - r_proj_norm) * np.tan(alpha_u)
    if r_rej_norm <= zeta:
        return 2

    # Check if satellite in penumbra
    Ki_p = (Constants.D_EARTH * r_earth_sun_norm) / (Constants.D_SUN + Constants.D_EARTH)
    alpha_p = np.arcsin(Constants.D_EARTH / (2 * Ki_p))
    kappa = (Ki_p + r_proj_norm) * np.tan(alpha_p)
    if r_rej_norm >= zeta and r_rej_norm <= kappa:
        return 1
    
    return 0

def brute_force_eclipse_angles(kep_state: np.ndarray,
                           curr_utc_str: str) -> Tuple[float|None, float|None]:
    """
    Compute eclipse entry and exit points in an orbit using brute force.

    NOTE: Returns eclipse angles in [0,2pi]
    """
    
    if kep_state[1] == 0:
        raise ValueError("Function doesn't support circular orbits.")
    
    ta_grid = np.linspace(0, 2*np.pi, 720)

    curr_et = spice.utc2et(curr_utc_str)
    entry_found = False
    exit_found = False

    # Initialize previous eclipse status
    # ta = ta_grid[-1]
    # new_kep_state = np.concatenate((kep_state[0:5], [ta]))
    # cart_state = astro_utils.kep2cart(kep_state, curr_et)
    # prev_eclipse_status = check_eclipse(cart_state[0:3], curr_utc_str)
    prev_eclipse_status = None
    
    for ta in ta_grid:

        new_kep_state = np.concatenate((kep_state[0:5], [ta]))
        cart_state = astro_utils.kep2cart(new_kep_state, curr_et)
        eclipse_status = check_eclipse(cart_state[0:3], curr_utc_str)

        # Detect eclipse entry
        if prev_eclipse_status is not None:
            if prev_eclipse_status == 0 and eclipse_status > 0:
                if entry_found:
                    raise ValueError("Second eclipse entry detected.")
                entry_found = True
                ta_entry = ta
            elif prev_eclipse_status > 0 and eclipse_status == 0:
                if exit_found:
                    raise ValueError("Second eclipse exit detected.")
                exit_found = True
                ta_exit = ta

        prev_eclipse_status = eclipse_status

    # No eclipse entry/exit points were detected
    if entry_found and exit_found:
        return ta_entry, ta_exit
    elif not entry_found and not exit_found:
        return None, None
    else:
        raise ValueError("Both entry and exit angle must be defined or both must be not defined.")

def compute_eclipse_angles(kep_state: np.ndarray,
                           curr_utc_str: str) -> Tuple[float, float]:
    """
    Compute true anomalies of eclipse entry and exit locations using a 
    cylindrical shadow assumption.

    1-2 orders of magnitude faster than `brute_force_eclipse_angles` but 
    lower resolution because it relies on a simplified cylindrical shadow assumption.

    Not thoroughly tested in this codebase, but Lundberg (1995) suggests that
    variations from the penumbra-umbra corrections from the cylindrical shadow 
    assumption are about one minute.

    NOTE: (for future ref) In case oblateness is to be consider, refer to this paper
    (https://www.sciencedirect.com/science/article/pii/S0094576523003582)
    
    Arguments:
        - kep_state: Current keplerian state [sma ecc inc raan aop ta]
        - curr_utc_str: Current UTC string (formatted as Constants.UTC_FORMAT)

    Returns:
        - Tuple[float, float]: Eclipse entry and exit angles

    References
        - Vallado, Fundamentals of Astrodynamics and Applications, Chapter 5.3.2
        - Neta and Vallado, On Satellite Umbra/Penumbra Entry and Exit Positions
    """


    ta_entry = None
    ta_exit = None

    curr_et = spice.utc2et(curr_utc_str)
    sma = kep_state[0]
    ecc = kep_state[1]
    inc = kep_state[2]
    raan = kep_state[3]
    aop = kep_state[4]

    # Near eclipse boundary
    if ecc >= 0.04:
        print("WARNING: Function poorly for ecc greater than ~0.04.")
    if inc >= 80*(np.pi/180):
        raise ValueError("WARNING: Function performs poorly for inc greater than ~80 degrees.")

    # Sun position relative to earth in J2000 frame
    r_sun_earth = -1 * get_earth_sun_vector(curr_utc_str)
    r_sun_earth_norm = np.linalg.norm(r_sun_earth)

    # Get cartesian state
    ma = astro_utils.true2mean(kep_state[5], ecc)
    kep_state_with_ma = np.concat((kep_state[0:5],[ma]))
    cart_state = astro_utils.kep2cart(kep_state_with_ma, curr_et)
    r_sc = cart_state[0:3]
    v_sc = cart_state[3:6]

    # Compute perifocal unit vectors in Earth-centered J2000 frame
    ecc_vec = astro_utils.compute_eccentricity_vector(r_sc,v_sc)
    ang_mom_hat = np.cross(r_sc,v_sc)/np.linalg.norm(np.cross(r_sc,v_sc))
    if ecc != 0:
        P_hat = ecc_vec / np.linalg.norm(ecc_vec)
    else:
        print("WARNING: Arbitrarily setting P_hat to [1,0,0]")
        P_hat = np.array([1,0,0])
    Q_hat = np.cross(ang_mom_hat, P_hat)

    # Compute beta1 and beta2
    beta1 = np.dot(r_sun_earth, P_hat) / r_sun_earth_norm
    beta2 = np.dot(r_sun_earth, Q_hat) / r_sun_earth_norm

    # if beta1**2 < 1 - (Constants.R_EARTH/(sma*(1-ecc)))**2:
    #     raise ValueError("No eclipse intersections found")
    # if beta1**2 > 1 - (Constants.R_EARTH/(sma*(1+ecc)))**2:
    #     raise ValueError("No eclipse intersections found")

    # Shadow function (x = cos(ta))
    slr = astro_utils.compute_semi_latus_rectum(sma,ecc)

    alpha = Constants.R_EARTH / slr
    alpha1 = (
        (alpha**4)*(ecc**4) - 
        2*(alpha**2)*(beta2**2-beta1**2)*(ecc**2) + 
        (beta1**2+beta2**2)**2
    )
    alpha2 = (
        4*(alpha**4)*(ecc**3) - 
        4*(alpha**2)*(beta2**2-beta1**2)*ecc
    )
    alpha3 = (
        6*(alpha**4)*(ecc**2) - 
        2*(alpha**2)*(beta2**2-beta1**2) - 
        2*(alpha**2)*(1-beta2**2)*(ecc**2) +
        2*(beta2**2-beta1**2)*(1-beta2**2) - 
        4*(beta2**2)*(beta1**2)
    )
    alpha4 = (
        4*(alpha**4)*ecc - 
        4*(alpha**2)*(1-beta2**2)*ecc
    )
    alpha5 = (
        alpha**4 - 2*(alpha**2)*(1-beta2**2) + (1-beta2**2)**2
    )


    coeffs = np.array([alpha1, alpha2, alpha3, alpha4, alpha5])
    shadow_roots = np.sort(np.roots(coeffs)) # sort in ascending order

    shadow_function = lambda ta: (
        (Constants.R_EARTH**2)*((1+ecc*np.cos(ta))**2) +
        (slr**2)*(beta1*np.cos(ta) + beta2*np.sin(ta))**2 - 
        slr**2
    )

    # x = np.linspace(-1.0, 1.0, 5000)
    # ta_grid = np.linspace(0,2*np.pi,5000)
    # y = np.polyval(coeffs, x)
    # plt.figure()
    # plt.plot(x, y)
    # plt.xlabel("x = cos(true anomaly)")
    # plt.ylabel("Shadow polynomial")
    # plt.grid(True)

    # plt.figure()
    # plt.plot(ta_grid, shadow_function(ta_grid))
    # plt.grid(True)

    # plt.show()

    # Filter out imaginary roots or roots outside of bounds
    valid_mask = ((np.abs(shadow_roots.imag)) < 1e-1) & (shadow_roots.real> -1) & (shadow_roots.real< 1)
    shadow_roots = shadow_roots[valid_mask].real

    # for ind in range(len(shadow_roots)):
    #     if shadow_roots[ind] > 1.0:
    #         shadow_roots[ind] = 1
    #     if shadow_roots[ind] < -1.0:
    #         shadow_roots[ind] = -1

    # Filter out duplicate double roots
    # Take the average of the two double roots to estimate the 
    # true double root
    double_shadow_roots = [None,None]
    if len(shadow_roots) == 4:
        if abs(shadow_roots[0] - shadow_roots[1]) < 1e-6:
            double_shadow_roots[0] = (shadow_roots[0] + shadow_roots[1])/2
        if abs(shadow_roots[2] - shadow_roots[3]) < 1e-6:
            double_shadow_roots[1] = (shadow_roots[2] + shadow_roots[3])/2

        if double_shadow_roots[0] is not None and double_shadow_roots[1] is not None:
            shadow_roots = np.array(double_shadow_roots)
        elif double_shadow_roots[0] is None and double_shadow_roots[1] is not None:
            shadow_roots = np.concatenate((shadow_roots[0:2],[double_shadow_roots[1]]))
        elif double_shadow_roots[0] is not None and double_shadow_roots[1] is None:
            shadow_roots = np.concatenate(([double_shadow_roots[0]],shadow_roots[2:]))

    # Filter out spurious roots
    shadow_angles = []
    for cos_ta in shadow_roots:
        # S_dot = 4*alpha1*cos_ta**3 + 3*alpha2*cos_ta**2 + 2*alpha3*cos_ta + alpha4

        ta_candidates = [
                np.acos(cos_ta),
                2*np.pi-np.acos(cos_ta)
        ]

        for ta in ta_candidates:     
            if beta1*np.cos(ta) + beta2*np.sin(ta) < 0:
                if abs(shadow_function(ta)) < 1e-1:
                    shadow_angles.append(ta)

    if len(shadow_angles) < 2:
        # ta_en, ta_ex = brute_force_eclipse_angles(kep_state, curr_utc_str)
        print("No eclipse angles found.")
        return 0.0,0.0
    elif len(shadow_angles) > 2:
        raise ValueError("Too many shadow angles.")
    
    ta_entry, ta_exit = sort_eclipse_angles(shadow_angles[0], shadow_angles[1])

    return ta_entry, ta_exit

def sort_eclipse_angles(a: float, b: float) -> Tuple[float,float]:
    """
    Given two angles from [0,2pi], determine which is entry and which is exit.

    Arguments:
        - a: an anomaly angle (rad)
        - b: an anomaly angle (rad)

    Returns:
        - Tuple[float,float]: (ta_entry, ta_exit) (angles are from [0,2pi])
    """

    # Get smaller (x) and larger (y) angles
    if a < b:
        x = a
        y = b
    elif b < a:
        x = b
        y = a
    else:
        return a,b
    
    delta1 = y-x
    delta2 = 2*np.pi - delta1

    # Determine which path is shorter. That path is the eclipse arc.
    if delta1 < delta2:
        return x,y
    elif delta2 < delta1:
        return y,x
    # Both paths are equal
    else:
        return x,y
    
def get_earth_sun_vector(curr_utc_str: str) -> np.ndarray:
    """
    Get position vector of earth relative to sun in J2000 frame from SPICE.

    Arguments:
        - curr_utc_str: Current UTC string (formatted as Constants.UTC_FORMAT)

    Returns:
        - np.ndarray: Position vector of Earth
    """

    curr_et = spice.utc2et(curr_utc_str)

    # Position of earth relative to sun in J2000 frame
    cart_state_earth_sun, _ = spice.spkezr("EARTH", curr_et, "J2000", "NONE", "SUN")
    r_earth_sun = cart_state_earth_sun[0:3]
    
    return r_earth_sun