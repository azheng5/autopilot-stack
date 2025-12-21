import numpy as np

def rk4_step(fn,
            t: float,
            x: np.ndarray,
            u: np.ndarray,
            delta_t: float) -> np.ndarray:
    
    #TODO: add u_next as input to improve timestep accuracy?
        
    k1 = fn(t,x,u)
    k2 = fn(t + delta_t/2, x + (delta_t/2)*k1, u)
    k3 = fn(t + delta_t/2, x + (delta_t/2)*k2, u)
    k4 = fn(t + delta_t, x + delta_t*k3, u)

    return x + (delta_t/6) * (k1 + 2*k2 + 2*k3 + k4)
    
def heun_step(fn,
              t: float,
              x: np.ndarray,
              delta_t: float,
              u: np.ndarray|None = None) -> np.ndarray:
    if u is not None:
        k1 = fn(t, x, u)
        k2 = fn(t + delta_t, x + delta_t*k1, u)
    else:
        k1 = fn(t, x)
        k2 = fn(t + delta_t, x + delta_t*k1)
    return  x + 0.5 * delta_t * (k1 + k2)