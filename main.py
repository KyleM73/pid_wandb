import numpy as np
import wandb

from simulation import simulate


def sweep() -> None:
    """
    Perform a WandB sweep to optimize PID gains.
    """
    wandb.init()

    # Retrieve hyperparameters from sweep config
    kp = wandb.config.kp
    ki = wandb.config.ki
    kd = wandb.config.kd

    # Simulate the system
    metrics = simulate(kp, ki, kd)

    # Log metrics to WandB
    wandb.log({"rmse": metrics["rmse"], "settling_time": metrics["settling_time"]})

    # Log the step response plot
    wandb.log({"step_response": wandb.Image(metrics["step_response_plot"])})

if __name__ == "__main__":
    np.random.seed(42)  # Fixed random seed for reproducibility

    # Define the sweep configuration
    sweep_config = {
        "method": "bayes",
        "metric": {
            "name": "rmse",
            "goal": "minimize"
        },
        "parameters": {
            "kp": {"min": 0.0, "max": 100.0},
            "ki": {"min": 0.0, "max": 100.0},
            "kd": {"min": 0.0, "max": 100.0},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="pid-optimization")
    wandb.agent(sweep_id, function=sweep, count=1000)
