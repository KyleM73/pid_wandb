def pid(x: float, dx: float, target: float, kp: float, ki: float, kd: float, integral: float, dt: float) -> tuple[float, float]:
    """
    PID controller

    Arguments:
        x: Current position
        dx: Current velocity
        target: Target position
        kp, ki, kd: PID gains
        integral: Accumulated integral error
        dt: Time step

    Returns:
        control signal u, updated integral error.
    """
    error = target - x
    integral += error * dt
    derivative = -dx

    u = kp * error + ki * integral + kd * derivative
    return u, integral