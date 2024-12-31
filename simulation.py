import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics import ode2
from pid import pid
from rk4 import rk4

def simulate(kp: float, ki: float, kd: float) -> dict:
    """
    Simulate the system with given PID gains and return performance metrics

    Arguments:
        kp, ki, kd: PID gains

    Returns:
        Dictionary with RMSE and settling time
    """
    # Simulation parameters
    T: float = 10.0  # simulation time
    dt: float = 0.01  # time step
    steps: int = int(T / dt)
    target: float = 1.0  # desired position

    # ODE parameters
    m: float = 1.0
    b: float = 0.5
    k: float = 2.0

    # Initialize state and variables
    x0: float = 0.0  # initial position
    v0: float = 0.0  # initial velocity
    y: np.ndarray = np.array([x0, v0], dtype=np.float64)
    times: np.ndarray = np.linspace(0, T, steps)
    x_vals, v_vals, u_vals = [], [], []
    integral: float = 0.0

    # Early stopping parameters
    convergence_threshold = 0.01
    settling_window = int(0.5 / dt)  # Check for settling over 0.5 seconds

    # Simulation loop
    for i, t in enumerate(times):
        x, dx = y
        u, integral = pid(x, dx, target, kp, ki, kd, integral, dt)
        y = rk4(lambda t, y, u: ode2(t, y, u, m=m, b=b, k=k), y, t, dt, u)

        # Store values for plotting
        x_vals.append(y[0])
        v_vals.append(y[1])
        u_vals.append(u)

        # Early stopping check
        if i >= settling_window:
            recent_errors = np.array([abs(target - x_vals[j]) for j in range(i - settling_window, i)])
            if np.all(recent_errors < convergence_threshold):
                break

    # Calculate performance metrics
    error: np.ndarray = np.array([target - x for x in x_vals], dtype=np.float64)
    rmse: float = np.sqrt(np.mean(error**2))
    settling_time = times[i] if i < len(times) else float("inf")

    # Plot the step response
    plt.figure(figsize=(10, 6))
    plt.plot(times[:len(x_vals)], x_vals, label="Position (x)")
    plt.axhline(target, color="r", linestyle="--", label="Target")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title("Step Response")
    plt.legend()
    plt.grid(True)
    plot_path = f"plots/step_response_kp_{kp:.2f}_ki_{ki:.2f}_kd_{kd:.2f}.png"
    plt.savefig(plot_path)
    plt.close()

    return {
        "rmse": rmse,
        "settling_time": settling_time,
        "step_response_plot": plot_path
    }