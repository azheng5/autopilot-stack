import sys
from pathlib import Path

import numpy as np
import spiceypy as spice

sys.path.append(str(Path(__file__).parent))
from flight_dynamics import Constants

spice.furnsh("meta_kernel.tm")

class Eclipse:
    """
    Astrodynamics tool for determining eclipse statuses of Earth-orbiting spacecraft.
    
    References:
        - Vallado, Fundamentals of Astrodynamics and Applications 4th Edition, Ch 5.3.2
        - Gonzalez, https://www.youtube.com/watch?v=LeJpWfyXJPw
        - Longo and Rickman, https://ntrs.nasa.gov/api/citations/19950023025/downloads/19950023025.pdf
    """

    @staticmethod
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
        r_earth_sun = Eclipse.get_earth_sun_vector(curr_utc_str)
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
        zeta =  (Ki_u - r_proj_norm) * np.tan(alpha_u) # IS IT r_proj_norm OR r_proj_scalar???????
        if r_rej_norm <= zeta:
            return 2

        # Check if satellite in penumbra
        Ki_p = (Constants.D_EARTH * r_earth_sun_norm) / (Constants.D_SUN + Constants.D_EARTH)
        alpha_p = np.arcsin(Constants.D_EARTH / (2 * Ki_p))
        kappa = (Ki_p + r_proj_norm) * np.tan(alpha_p) # IS IT r_proj_norm OR r_proj_scalar???????
        if r_rej_norm >= zeta and r_rej_norm <= kappa:
            return 1
        
        return 0

    @staticmethod
    def compute_shadow_angle() -> float:
        """
        TODO: reference this paper: https://www.sciencedirect.com/science/article/pii/S0094576523003582
        or can ref Vallado 5.3.2
        
        Compute shadow arc angle of a Keplerian orbit. If 0.0, then the given
        orbit doesn't have an eclipse region.
        """
        pass

    @staticmethod
    def get_earth_sun_vector(curr_utc_str: str):

        curr_et = spice.utc2et(curr_utc_str)

        # Position of earth relative to sun in J2000 frame
        cart_state_earth_sun, _ = spice.spkezr("EARTH", curr_et, "J2000", "NONE", "SUN")
        r_earth_sun = cart_state_earth_sun[0:3]
        
        return r_earth_sun