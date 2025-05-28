"""Shallow water gravity wave example using the consolidated solver."""

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os

from wave_sim import ShallowWaterGravityWave
from wave_sim.initial_conditions import gaussian_1d


def h0(x):
    L = x[-1] + (x[1] - x[0])
    return 1.0 + 0.1 * gaussian_1d(x, center=L/4, sigma=L/20)


def u0(x):
    return np.zeros_like(x)

DEFAULT_OUTPUT_DIR = "output_1d_animations_individual"


def generate_animation(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    out_name: str = "shallow_water_gravity_wave.mp4",
    steps: int | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)

    L_domain = 10.0
    Nx_points = 200
    T_duration = 2.0
    g_val = 9.81
    dx_val = L_domain / Nx_points
    dt_val = 0.5 * dx_val / np.sqrt(g_val)

    sim = ShallowWaterGravityWave(g=g_val, L=L_domain, Nx=Nx_points, dt=dt_val, T=T_duration)
    sim.initial_conditions(h0, u0)

    writer = imageio.get_writer(out_path, fps=30)
    fig, ax = plt.subplots(figsize=(8, 4))

    nsteps = sim.nt if steps is None else min(steps, sim.nt)
    for i in range(nsteps):
        sim.step()
        ax.clear()
        ax.plot(sim.x, sim.Q[0, :])
        ax.set_ylim(0.8, 1.2)
        ax.set_xlabel("x")
        ax.set_ylabel("Water height")
        ax.set_title(f"Shallow Water Gravity Wave, Time: {i*sim.dt:.3f}s")
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(img)

    writer.close()
    plt.close(fig)
    print(f"Generated 1D animation: {out_path}")
    return out_path


if __name__ == "__main__":
    path = generate_animation()
