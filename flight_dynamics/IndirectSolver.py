import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad

class IndirectSolver:
    """
    Indirect thrust solver for coplanar circle-to-circle low thrust transfers.

    Takes an an initial Cartesian state and a target semi-major axis, and generates
    an optimal thrust profile to achieve the transfer.

    References:
        - Colasurdo and Casalino, https://arc.aiaa.org/doi/epdf/10.2514/6.2004-5087
    
    """

    def __init__(self,
                 initial_kep_state: np.ndarray,
                 final_sma: float) -> None:
        self.initial_kep_state = initial_kep_state
        self.final_sma = final_sma

    def single_rev_program(self, shadow_angle: float) -> np.ndarray:
        """
        Compute thrust profile for a single revolution coplanar transfer. Seeks to 
        maximize change in semi-major axis while maintaining zero eccentricity at end of revolution.

        Arguments:
            - shadow_angle: Amplitude of shadow arc angle for the current revolution (delta_s) (radians)
        """

        # First, compute start and end points of the thrusting arc. For a single revolution, 
        # the thrusting arc begins at an angle of nu=-delta and ends at an angle of 
        # nu=+delta, where the angular position is defined wrt the Earth-Sun line.
        delta = -0.5 * (shadow_angle - 2*np.pi)

        # Define mean anomaly nu over thrusting arc
        nu = np.linspace(-delta, delta, 1000)

        # Solve for K2 integration constant
        K2 = newton(lambda K2: self.K2_root_fn(K2, delta), x0=0.0)

        # Obtain optimal thrust direction alpha with mean anomaly nu as independent variable
        tan_alpha = (K2 * np.sin(nu)) / ( 2*(1 + K2 * np.cos(nu)) )
        alpha = np.arctan(tan_alpha)

        # Represent alpha as function of time

        

        # Compute 3d thrust components
        
        return alpha
        
    def K2_root_fn(self, K2: float, delta: float) -> float:
        """Helper function for K2 root finding."""
        value, _ = quad(lambda nu: self.K2_integrand(K2,nu), -delta, delta)
        return value

    def K2_integrand(self, K2: float, nu: float) -> float:
        """Helper function for K2 root finding."""
        num = K2 * np.sin(nu)**2 + 4 * (1 + K2 * np.cos(nu)) * np.cos(nu)
        den = np.sqrt( ( K2 * np.sin(nu) )**2 + 4*( 1 + K2 * np.cos(nu) )**2 )
        return num / den

    # def multi_rev_program()

    # def compute_eclipse
        # use spice to get earth position, and use that to compute shadow angle