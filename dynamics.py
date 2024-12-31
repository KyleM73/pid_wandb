import numpy as np

def ode2(
        t: float,
        y: np.ndarray,
        u: float,
        m: float = 1.0,
        b: float = 0.5,
        k: float = 2.0,
) -> np.ndarray:
    """
    Second-order dynamical system:
    m*x'' + b*x' + k*x = u

    Arguments:
        y: [x, dx]
        u: Control input
        m: Mass
        b: Damping
        k: Stiffness

    Returns:
        dy: Derivative [dx, ddx].
    """
    x, dx = y
    ddx = (u - b * dx - k * x) / m
    return np.array([dx, ddx], dtype=np.float64)