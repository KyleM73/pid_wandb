import numpy as np

def rk4(
        f: callable,
        y0: np.ndarray,
        t: float,
        dt: float,
        u: float,
) -> np.ndarray:
    """
    Runge-Kutta Explicit Solver
    
    Arguments:
        f: System dynamics
        y0: Initial state [x, dx]
        t: Current time
        dt: Time step
        u: Control input

    Returns:
        y_next: State at t + dt
    """
    k1 = f(t, y0, u)
    k2 = f(t + dt / 2, y0 + dt * k1 / 2, u)
    k3 = f(t + dt / 2, y0 + dt * k2 / 2, u)
    k4 = f(t + dt, y0 + dt * k3, u)

    y_next = y0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_next