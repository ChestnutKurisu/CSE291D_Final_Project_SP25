"""Rotating shallow water Kelvin wave using the unified solver."""

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os

from wave_sim import KelvinWave


def eta0(y):
    return np.exp(-((y - 2.5) / 0.5) ** 2)


OUTPUT_DIR = "output_1d_animations"


def generate_animation(out_name="kelvin_wave.mp4"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_name)
    sim = KelvinWave(L=10.0, Ny=800, T=10.0)
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

