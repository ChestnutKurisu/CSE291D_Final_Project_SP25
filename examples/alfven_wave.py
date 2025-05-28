"""Alfv\u00e9n wave example using the consolidated solver."""

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os

from wave_sim import AlfvenWave


def v0(x):
    L = x[-1]
    return np.sin(2 * np.pi * x / L)


DEFAULT_OUTPUT_DIR = "output_1d_animations_individual"


def generate_animation(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    out_name: str = "alfven_wave.mp4",
    steps: int | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)

    sim_L = 2.0
    sim_Nx = 800
    sim_T = 2.0
    dx = sim_L / sim_Nx
    vA_sim = 1.0
    dt_val = 0.5 * dx / vA_sim
    sim = AlfvenWave(L=sim_L, Nx=sim_Nx, T=sim_T, dt=dt_val)
    sim.initial_conditions(v0)
    writer = imageio.get_writer(out_path, fps=30)
    fig, ax = plt.subplots(figsize=(8, 4))
    nsteps = sim.nt if steps is None else min(steps, sim.nt)
    for _ in range(nsteps):
        sim.step()
        ax.clear()
        ax.plot(sim.x, sim.v)
        ax.set_ylim(-1.2, 1.2)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(img)
    writer.close()
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = generate_animation()
    print(f"Wrote {path}")

