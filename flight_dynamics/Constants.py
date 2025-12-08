from pathlib import Path

EARTH_MU = 3.986004418e5

EARTH_MASS = 5.97219e24

J2 = 0.0010827

R_EARTH = 6378.1370

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
UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"