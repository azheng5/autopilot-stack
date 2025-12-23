import sys
from pathlib import Path

import numpy as np
import spiceypy as spice
from scipy.optimize import newton

sys.path.append(str(Path(__file__).parent.parent))
from flight_dynamics import Constants

np.seterr(invalid='raise')

"""
Astrodynamics utility functions for circular/elliptical LEO.

References
    - SpiceyPy documentation: https://spiceypy.readthedocs.io/en/main/documentation.html
"""

def kep2cart(kep_state: np.ndarray, et: float) -> np.ndarray:
    """
    Convert Keplerian elements to Cartesian state using SPICE.

    Arguments:
        kep_state: Keplerian state [sma (km), ecc, inc (rad), raan (rad), aop (rad), ma (rad)]
        et: Ephemeris time (seconds past J2000)

    Returns:
        Cartesian state [rx, ry, rz, vx, vy, vz] in km and km/s
    """

    sma = kep_state[0]
    ecc = kep_state[1]
    if ecc < 0:
        raise ValueError("Eccentricity is negative.")
    inc = kep_state[2]
    raan = kep_state[3]
    aop = kep_state[4]
    ma = kep_state[5]

    # Periapsis distance
    rp = sma * (1 - ecc)

    elts = spice.conics(np.array([rp, ecc, inc, raan, aop, ma, et, Constants.EARTH_MU]), et)
    cart_state = np.array([elts[0], elts[1], elts[2], elts[3], elts[4], elts[5]])
    return cart_state

def cart2kep(cart_state: np.ndarray, et: float) -> np.ndarray:
    """
    Convert Cartesian state to Keplerian elements using SPICE.

    Arguments:
        cart_state: Cartesian state [rx, ry, rz, vx, vy, vz] in km and km/s
        et: Ephemeris time (seconds past J2000)

    Returns:
        Keplerian state [sma (km), ecc, inc (rad), raan (rad), aop (rad), ma (rad)]
    """

    #NOTE: returns LAN instead of RAAN (possible problem?)
    elts = spice.oscelt(cart_state, et, Constants.EARTH_MU)
    rp = elts[0]
    sma = rp / (1 - elts[1])
    ecc = elts[1]


    # inc = wrap_angle(elts[2])
    # raan = wrap_angle(elts[3])
    # aop = wrap_angle(elts[4])
    # ma = wrap_angle(elts[5])
    inc = elts[2]
    raan = elts[3]
    aop = elts[4]
    ma = elts[5]

    kep_state = np.array([sma, ecc, inc, raan, aop, ma])
    return kep_state

def wrap_angle(angle) -> float:

    ang = np.mod(angle, 2*np.pi)
    # if np.isclose(ang, 2*np.pi, atol=1e-12):
    #     return 0.0

    # if ang >= np.pi:
    #     ang = ang - 2*np.pi

    return ang

def compute_sso_inc(sma: float, ecc: float):
    """
    Compute sun-synchronous orbit inclination

    Arguments:
        sma: Semi-major axis (km)
        ecc: Eccentricity

    Returns:
        inc: Inclination (rad)
    """

    if ecc < 0:
        raise ValueError("Eccentricity is negative.")

    raan_dot = 2*np.pi / Constants.SIDEREAL_YEAR_SEC

    n = np.sqrt(Constants.EARTH_MU / sma**3)

    cos_inc = (-2/3) * raan_dot * (1/Constants.J2) * ((1-ecc**2)**2/n) * (sma/Constants.R_EARTH)**2
    return np.acos(cos_inc)

def compute_j2_drift_effect(sma, ecc, inc):

    n = np.sqrt(Constants.EARTH_MU / sma**3)

    raan_dot = -1.5 * Constants.J2 * (n/(1-ecc**2)**2) * (Constants.R_EARTH/sma)**2 * np.cos(inc)
    aop_dot = 0.75 * Constants.J2 * (n/(1-ecc**2))**2 * (Constants.R_EARTH/sma)**2 * (5 * np.cos(inc)**2  - 1)

    return raan_dot, aop_dot

def construct_rotation_matrix(x_hat: np.ndarray,
                              y_hat: np.ndarray,
                              z_hat: np.ndarray) -> np.ndarray:
    """
    Generates a rotation matrix $R_A^B$ from frame A to frame B using 
    3 unit vectors.

    Arguments:
        - x_hat: x basis vector for I frame expressed in B frame ($\mathbf{\hat{x}}^B_I$)
        - y_hat: y basis vector for I frame expressed in B frame ($\mathbf{\hat{y}}^B_I$)
        - z_hat: z basis vector for I frame expressed in B frame ($\mathbf{\hat{z}}^B_I$)
    """

    # Construct rotation matrix R_BA from A to B
    return np.column_stack((x_hat, y_hat, z_hat))

def lvlh_to_eci_matrix(pos: np.ndarray,
                       vel: np.ndarray):
    """
    Generate a rotation matrix from LVLH to ECI frame. The ECI frame is considered as 
    the Earth-centered J2000 frame as defined by SPICE. To rotate a vector from the
    ECI to LVLH frame, use the transpose of this function's return.

    Arguments:
        - pos: Position of spacecraft in ECI frame
        - vel: Velocity of spacecraft in ECI frame

    Returns
        - np.ndarray: Rotation matrix from LVLH to ECI
    """

    x_hat = pos / np.linalg.norm(pos)
    h = np.cross(pos,vel)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)

    # Rotation matrix from LVLH to ECI
    R_ECI_LVLH = construct_rotation_matrix(x_hat, y_hat, z_hat)

    return R_ECI_LVLH

def get_orbit_period(sma: float):
    return (2*np.pi) / np.sqrt(Constants.EARTH_MU/(sma**3))

def mean2true(M: float, 
            ecc: float,
            tol: float,
            max_iter: int = 50) -> float:
    """Converts mean to true anomaly"""

    if ecc < 0:
        raise ValueError("Eccentricity is negative.")
    if ecc == 0:
        return M

    # Initial guess
    E0 = M

    # Solve for eccentric anomaly
    g = lambda E: E - ecc*np.sin(E) - M
    g_dot = lambda E: 1 - ecc*np.cos(E)
    E = newton(g, E0, fprime=g_dot, tol=tol, maxiter=max_iter)

    # Find true anomaly from eccentric anomaly
    return eccentric2true(E, ecc)

def eccentric2true(E: float, ecc: float) -> float:

    if ecc < 0:
        raise ValueError("Eccentricity is negative.")
    if ecc == 0:
        return E
    
    return 2 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(E/2),
        np.sqrt(1 - ecc) * np.cos(E/2)
    )

def true2mean():
    pass

def true2eccentric(ta: float, ecc: float) -> float:

    if ecc < 0:
        raise ValueError("Eccentricity is negative.")
    if ecc == 0:
        return ta
    
    return 2 * np.arctan2(
        np.sqrt(1 - ecc) * np.sin(ta/2),
        np.sqrt(1 + ecc) * np.cos(ta/2)
    )

def mean_anomaly_to_time(M_final: float, M_init: float, sma: float) -> float:
    """Convert change in mean anomaly to change in time (seconds)"""
    mean_motion = np.sqrt(Constants.EARTH_MU / sma**3)
    delta_t = ((M_final - M_init)%(2*np.pi)) / mean_motion
    return delta_t

def compute_mean_motion(sma: float) -> float:
    return np.sqrt(Constants.EARTH_MU / sma**3)

def compute_radius(sma: float, ecc: float, ta: float) -> float:

    if ecc < 0:
        raise ValueError("Eccentricity is negative.")
    if ecc == 0:
        return sma
    
    return (sma*(1 - ecc**2)) / (1 + ecc*np.cos(ta))
    
def vis_viva(sma: float, r: float) -> float:
    return np.sqrt(Constants.EARTH_MU * (2/r - 1/sma))

def compute_ang_mom_norm(sma: float, ecc: float) -> float:

    if ecc < 0:
        raise ValueError("Eccentricity is negative.")
    
    return np.sqrt(Constants.EARTH_MU * sma * (1 - ecc**2))

def classical_to_equinoctial(kep_state: np.ndarray) -> np.ndarray:
    """
    Converts classical keplerian elements to nonsingular equinoctial elements.
    Inverse of equinoctial_to_classical().

    NOTE: If eccentricity is zero, information about E will be lost and it is
    not possible to obtain it with an inverse conversion.
    
    Arguments
        - kep_state: [sma ecc inc raan aop E]

    Returns
        - np.ndarray: [sma h k p q F]
    """

    if len(kep_state) != 6:
        raise ValueError("Input state is not correct length.")

    if kep_state[1] == 0:
        print("NOTE: e=0: Information about E will be lost when converting from classical to equinoctial.")
    
    sma = kep_state[0]
    ecc = kep_state[1]
    inc = kep_state[2]
    raan = kep_state[3]
    aop = kep_state[4]
    E = kep_state[5]

    if sma < 0:
        raise ValueError(f"SMA is negative: {sma}") 
        # we never expect parabolic/hyperbolic in this codebase
    
    if ecc < 0: 
        raise ValueError(f"ECC is negative: {ecc}")

    h = ecc * np.sin(aop + raan)
    k = ecc * np.cos(aop + raan)
    p = np.tan(inc/2) * np.sin(raan)
    q = np.tan(inc/2) * np.cos(raan)
    F = raan + aop + E

    return np.array([sma,h,k,p,q,F])

def equinoctial_to_classical(equin_state: np.ndarray) -> np.ndarray:
    """
    Converts equinoctial elements to classical keplerian elements.
    Inverse of classical_to_equinoctial().

    #TODO does not handle polar orbit singularity

    Arguments:
        - equin_state: [sma h k p q F]

    Returns:
        - np.ndarray: [sma ecc inc raan aop E]
    """

    if len(equin_state) != 6:
        raise ValueError("Input state is not correct length.")

    sma = equin_state[0]
    h = equin_state[1]
    k = equin_state[2]
    p = equin_state[3]
    q = equin_state[4]
    F = equin_state[5]

    # these never happen as long as orbit eccentricity in [0,1)
    if h >= 1:
        raise ValueError(f"h is greater than or equal 1: {h}")
    if h <= -1:
        raise ValueError(f"h is less than or equal -1: {h}")
    if k >= 1:
        raise ValueError(f"k is greater than or equal 1: {k}")
    if k <= -1:
        raise ValueError(f"k is less than or equal -1: {k}")

    ecc = np.sqrt(h**2 + k**2)

    lop = np.arctan2(h,k)
    E = F - lop

    inc = 2 * np.arctan2(np.sqrt(p**2+q**2),1)
    raan = np.arctan2(p,q)
    aop = lop - raan

    if ecc == 0:
        print("NOTE: Converted to e=0 since h=0 and k=0: Info about F will be lost, and defaulting to AOP=0.")
        aop = 0
    else:
        aop = lop - raan

    return np.array([sma,ecc,inc,raan,aop,E])

#TODO:
def lla_to_eci():
    pass

#TODO
def eci_to_lla():
    pass