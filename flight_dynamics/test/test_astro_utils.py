import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flight_dynamics import astro_utils as astro_utils
from flight_dynamics import Constants

def test_conversions():
    """"""

    sma = Constants.R_EARTH + 500
    ecc = 0.0001
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

# Debugging mode
if __name__ == "__main__":
    test_conversions()