import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from skopt.space import Real

def generate_gp_posterior_frames(result, param_x, param_y, folder="src/Training/plots/gp_2d_frames", n_points=50):
    os.makedirs(folder, exist_ok=True)

    space = result.space
    names = [dim.name for dim in space.dimensions]
    
    idx_x = names.index(param_x)
    idx_y = names.index(param_y)

    # Build grid for 2D surface
    dim_x = space.dimensions[idx_x]
    dim_y = space.dimensions[idx_y]
    x_vals = np.linspace(dim_x.low, dim_x.high, n_points)
    y_vals = np.linspace(dim_y.low, dim_y.high, n_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    XY_grid = np.array([[x, y] for x in x_vals for y in y_vals])

    for i, model in enumerate(result.models):
        # Create full input vectors for GP prediction
        grid_full = []
        for x, y in XY_grid:
            point = []
            for j, dim in enumerate(space.dimensions):
                if j == idx_x:
                    point.append(x)
                elif j == idx_y:
                    point.append(y)
                else:
                    # Use midpoint for other dims
                    point.append((dim.low + dim.high) / 2)
            grid_full.append(point)

        mu, std = model.predict(grid_full, return_std=True)
        mu = -mu  # because we minimized
        mu = mu.reshape(n_points, n_points)
        std = std.reshape(n_points, n_points)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        cp = ax.contourf(X, Y, mu, levels=30, cmap='viridis')
        cs = ax.contour(X, Y, std, levels=10, cmap='cool', alpha=0.5)

        # Add sampled points up to iteration i
        sampled_points = np.array(result.x_iters[:i+1])
        xs = sampled_points[:, idx_x]
        ys = sampled_points[:, idx_y]
        ax.scatter(xs, ys, c='white', s=20, edgecolors='black')

        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_title(f"GP Posterior at Iteration {i+1}")
        fig.colorbar(cp, ax=ax, label="Predicted Score (mean)")
        plt.savefig(f"{folder}/frame_{i:03d}.png")
        plt.close()

def generate_gp_1d_frames(result, param_name, folder="src/Training/plots/gp_1d_frames", n_points=100):
    """
    Generate a frame per iteration showing GP posterior over 1 weight.
    Includes sampled points and GP uncertainty.
    """
    os.makedirs(folder, exist_ok=True)
    space = result.space
    names = [dim.name for dim in space.dimensions]

    # Get index of the parameter to vary
    idx = names.index(param_name)
    dim = space.dimensions[idx]
    x_vals = np.linspace(dim.low, dim.high, n_points)

    for step, model in enumerate(result.models):
        # ✅ Use the best sample so far as the "fixed" anchor point
        if step == 0:
            best_index = 0
        else:
            best_index = np.argmin(result.func_vals[:step+1])

        fixed_sample = result.x_iters[best_index]
        fixed_vals = {name: val for name, val in zip(names, fixed_sample)}

        # 🔁 Build input matrix where only one variable varies
        X = []
        for val in x_vals:
            point = []
            for i, d in enumerate(space.dimensions):
                if i == idx:
                    point.append(val)
                else:
                    point.append(fixed_vals[d.name])
            X.append(point)

        mu, std = model.predict(X, return_std=True)
        mu = -mu  # because skopt minimizes

        # 🧪 Plot
        plt.figure(figsize=(8, 4))
        plt.plot(x_vals, mu, label="Predicted Score", color="blue")
        plt.fill_between(x_vals, mu - std, mu + std, alpha=0.3, label="±1 Std Dev", color="blue")

        # 🟥 Add sampled points (X values along this dimension)
        xs = [x[idx] for x in result.x_iters[:step+1]]
        ys = [-result.func_vals[i] for i in range(step+1)]  # Convert back from minimized value
        plt.scatter(xs, ys, c='red', s=30, label='Sampled Points')

        # 📊 Final touches
        plt.title(f"GP Posterior for '{param_name}' — Iteration {step+1}")
        plt.xlabel(param_name)
        plt.ylabel("Predicted Score")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save frame
        plt.savefig(f"{folder}/frame_{step:03d}.png")
        plt.close()


def make_gif_from_frames(folder="src/Training/plots/gp_2d_frames", output="src/Training/plots/gp_posterior_evolution.gif", fps=2):
    images = []
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    for filename in files:
        path = os.path.join(folder, filename)
        images.append(imageio.imread(path))
    imageio.mimsave(output, images, fps=fps)
    print(f"✅ Saved GIF to {output}")

def make_1d_gp_gif(folder="src/Training/plots/gp_1d_frames", output="src/Training/plots/gp_1d_evolution.gif", fps=2):
    images = []
    files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    for fname in files:
        path = os.path.join(folder, fname)
        images.append(imageio.imread(path))
    imageio.mimsave(output, images, fps=fps)
    print(f"✅ Saved 1D GP evolution GIF to: {output}")
