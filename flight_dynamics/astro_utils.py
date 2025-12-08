import sys
from pathlib import Path

import numpy as np
import spiceypy as spice

sys.path.append(str(Path(__file__).parent.resolve()))
from flight_dynamics import Constants

"""
Standalone astrodynamics utility functions.

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
    inc = wrap_angle(elts[2])
    raan = wrap_angle(elts[3])
    aop = wrap_angle(elts[4])
    ma = wrap_angle(elts[5])

    kep_state = np.array([sma, ecc, inc, raan, aop, ma])
    return kep_state

def wrap_angle(angle) -> float:

    ang = np.mod(angle, 2*np.pi)
    # if np.isclose(ang, 2*np.pi, atol=1e-12):
    #     return 0.0

    if ang >= np.pi:
        ang = ang - 2*np.pi

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

    raan_dot = 2*np.pi / Constants.SIDEREAL_YEAR_SEC

    n = np.sqrt(Constants.EARTH_MU / sma**3)

    cos_inc = (-2/3) * raan_dot * (1/Constants.J2) * ((1-ecc**2)**2/n) * (sma/Constants.R_EARTH)**2
    return np.acos(cos_inc)

def compute_j2_drift_effect(sma, ecc, inc) -> float:

    n = np.sqrt(Constants.EARTH_MU / sma**3)

    raan_dot = -1.5 * Constants.J2 * (n/(1-ecc**2)**2) * (Constants.R_EARTH/sma)**2 * np.cos(inc)
    aop_dot = 0.75 * Constants.J2 * (n/(1-ecc**2))**2 * (Constants.R_EARTH/sma)**2 * (5 * np.cos(inc)**2  - 1)

    return raan_dot, aop_dot