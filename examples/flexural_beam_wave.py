"""Flexural beam wave solved with the consolidated solver."""

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os

from wave_sim import FlexuralBeamWave


def w0(x):
    L = x[-1]
    return np.exp(-100 * (x - L / 2) ** 2)


DEFAULT_OUTPUT_DIR = "output_1d_animations_individual"


def generate_animation(output_dir=DEFAULT_OUTPUT_DIR, out_name="flexural_beam_wave.mp4"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)

    sim_L = 2.0
    sim_Nx = 801
    sim_T = 5.0
    dx = sim_L / (sim_Nx - 1)
    D_val = 0.01
    dt_val = 0.2 * dx ** 2 / np.sqrt(D_val)
    sim = FlexuralBeamWave(D=D_val, L=sim_L, Nx=sim_Nx, T=sim_T, dt=dt_val)
    sim.initial_conditions(w0)
    writer = imageio.get_writer(out_path, fps=30)
    fig, ax = plt.subplots(figsize=(8, 4))
    for _ in range(sim.nt):
        sim.step()
        ax.clear()
        ax.plot(sim.x, sim.w)
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

