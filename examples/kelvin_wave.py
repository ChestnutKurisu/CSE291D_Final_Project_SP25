"""Rotating shallow water Kelvin wave using the unified solver."""

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os

from wave_sim import KelvinWave


def eta0(y):
    return np.exp(-((y - 2.5) / 0.5) ** 2)


DEFAULT_OUTPUT_DIR = "output_1d_animations_individual"


def generate_animation(output_dir=DEFAULT_OUTPUT_DIR, out_name="kelvin_wave.mp4"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)

    sim_L = 10.0
    sim_Ny = 800
    sim_T = 10.0
    dy = sim_L / (sim_Ny - 1)
    c = np.sqrt(9.81 * 1.0)
    dt_val = 0.4 * dy / c
    sim = KelvinWave(L=sim_L, Ny=sim_Ny, T=sim_T, dt=dt_val)
    sim.initial_conditions(eta0)
    writer = imageio.get_writer(out_path, fps=30)
    fig, ax = plt.subplots(figsize=(8, 4))
    for _ in range(sim.nt):
        sim.step()
        ax.clear()
        ax.plot(sim.y, sim.eta)
        ax.set_ylim(-1.1, 1.1)
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

