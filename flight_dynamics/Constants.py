from pathlib import Path


"""
Planetary and solar constants

Reference:
    - Vallado, Fundamentals of Astrodynamics and Applications 4th Edition, Appendix D
"""

# Radius of sun (km)
R_SUN = 696000
D_SUN = R_SUN*2

# Earth equatorial radius (km)
R_EARTH = 6378.1363
D_EARTH = R_EARTH*2

# Earth gravitational parameter (km^3/s^2)
EARTH_MU =  3.986004415e5

# Earth mass (kg)
EARTH_MASS = 5.9742e24

# Earth's zonal harmonic coefficients
J2 = 0.0010826269
J3 = -0.0000025323
J4 = -0.0000016204

G0 = 0.00980665

SIDEREAL_YEAR_SEC = 365.25 * 24 * 3600

"""
Universal output data path
"""
OUT_PATH = Path("data")
OUT_PATH.mkdir(exist_ok=True)

"""
Universal UTC string format across codebase
ISO 8601 UTC timestamp with microseconds
"""
UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

"""
Directory containing SPICE kernels and ephemerides
#NOTE: Do not change this path, `meta_kernel.tm` assumes this path
"""
KERNEL_DIR = "/opt/spice_kernels"