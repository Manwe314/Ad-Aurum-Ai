import os
import copy
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from bbb.globals import WW_VARS_TRAINING
from .experiments import run_delta_parallel
from skopt.plots import plot_convergence
from .plotter import generate_gp_1d_frames, make_1d_gp_gif
import matplotlib.pyplot as plt
import json

# === CONFIG ===
N_INITIAL_POINTS = 45
N_CALLS = 100
BOUNDS = {
    key: (0.0, 2.0) for key in WW_VARS_TRAINING.keys()
}

# === DEFINE SEARCH SPACE ===
dimensions = [
    Real(low, high, name=key)
    for key, (low, high) in BOUNDS.items()
]
param_names = list(BOUNDS.keys())

# === OBJECTIVE FUNCTION ===
@use_named_args(dimensions=dimensions)
def objective(**params):
    # Update global weights
    WW_VARS_TRAINING.update(params)

    # Run the game simulation with these weights
    result = run_delta_parallel()

    # Our custom score: win_rate - lambda * std_dev
    score = result.score  # assume this is already adjusted with std_penalty

    #print(f"Score={score:.4f} with params={params}")
    return -score  # BO minimizes, we want to maximize score

# === RUN OPTIMIZATION ===
def run_bo():
    print("Starting Bayesian Optimization with Gaussian Process + EI")
    os.makedirs("plots", exist_ok=True)

    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        acq_func="EI",            # Expected Improvement
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL_POINTS,
        random_state=42,
        verbose=True
    )

    plot_convergence(result)
    plt.savefig("src/Training/plots/convergence_curve.png")
    plt.clf()

    # Display best result
    #print("\n=== BEST FOUND CONFIGURATION ===")
    best_score = -result.fun
    best_params = dict(zip(param_names, result.x))
    save_best_weights(best_params, best_score)
    # for k, v in best_params.items():
    #     print(f"{k}: {v:.4f}")
    #print(f"Best score: {best_score:.4f}")

    return best_params, best_score

def save_best_weights(best_params, score, path="src/Training/bo_results/best_weights.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {
        "score": score,
        "weights": best_params
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Saved best weights to: {path}")







if __name__ == "__main__":
    run_bo()
