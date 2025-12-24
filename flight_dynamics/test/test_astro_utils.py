import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils as astro_utils
from flight_dynamics import Constants

def test_conversions():

    sma = Constants.R_EARTH + 500
    ecc = 0.001
    inc = 10 * (np.pi / 180)
    raan = 20 * (np.pi / 180)
    aop = 30 * (np.pi / 180)
    ma = 0 * (np.pi / 180)

    expected_kep_state = np.array([sma, ecc, inc, raan, aop, ma])
    et = 0.0

    cart_state = astro_utils.kep2cart(expected_kep_state, et)

    kep_state = astro_utils.cart2kep(cart_state, et)

    print(f"Expected {expected_kep_state}, got {kep_state}")
    assert np.allclose(kep_state, expected_kep_state, atol=1e-9)

    # Test zero eccentricity edge case

def test_classical_to_equinoctial():

    sma = Constants.R_EARTH + 500
    ecc = 0.01
    inc = 10 * (np.pi / 180)
    raan = 20 * (np.pi / 180)
    aop = 30 * (np.pi / 180)

    # Validate inverse for E=0
    kep_state = np.array([sma, ecc, inc, raan, aop, 0])
    res_equin_state = astro_utils.classical_to_equinoctial(kep_state)
    res_kep_state = astro_utils.equinoctial_to_classical(res_equin_state)
    assert np.allclose(kep_state, res_kep_state, atol=1e-12)

    # Validate inverse for E!=0
    kep_state = np.array([sma, ecc, inc, raan, aop, 0.1])
    res_equin_state = astro_utils.classical_to_equinoctial(kep_state)
    res_kep_state = astro_utils.equinoctial_to_classical(res_equin_state)
    res_res_equin_state = astro_utils.classical_to_equinoctial(res_kep_state)
    assert np.allclose(kep_state, res_kep_state, atol=1e-12)

    # Validate inverse for E<0
    kep_state = np.array([sma, ecc, inc, raan, aop, -np.pi])
    res_equin_state = astro_utils.classical_to_equinoctial(kep_state)
    res_kep_state = astro_utils.equinoctial_to_classical(res_equin_state)
    res_res_equin_state = astro_utils.classical_to_equinoctial(res_kep_state)
    assert np.allclose(kep_state, res_kep_state, atol=1e-12)

    # NOTE: So far I've been able to validate the conversion is perfectly inverse 
    # with up 1e-12 accuracy

def test_equinoctial_to_classical():

    sma = Constants.R_EARTH + 500
    h = 0.000001 # getting as close to the e=0 boundary as possible
    k = 0.000001
    p = 0.1
    q = -0.1

    equin_state = np.array([sma, h, k, p, q, 0.0])
    res_kep_state = astro_utils.equinoctial_to_classical(equin_state)
    res_equin_state = astro_utils.classical_to_equinoctial(res_kep_state)
    assert np.allclose(equin_state, res_equin_state, atol=1e-12)

    equin_state = np.array([sma, h, k, p, q, 1.0])
    res_kep_state = astro_utils.equinoctial_to_classical(equin_state)
    res_equin_state = astro_utils.classical_to_equinoctial(res_kep_state)
    assert np.allclose(equin_state, res_equin_state, atol=1e-12)

    equin_state = np.array([sma, h, k, p, q, -np.pi])
    res_kep_state = astro_utils.equinoctial_to_classical(equin_state)
    res_equin_state = astro_utils.classical_to_equinoctial(res_kep_state)
    assert np.allclose(equin_state, res_equin_state, atol=1e-12)

    # NOTE: So far I've been able to validate the conversion is perfectly inverse 
    # with up 1e-12 accuracy


# Debugging mode
if __name__ == "__main__":
    # test_conversions()
    test_classical_to_equinoctial()
    test_equinoctial_to_classical()