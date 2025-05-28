"""Alfv\u00e9n wave example using the consolidated solver."""

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os

from wave_sim import AlfvenWave


def v0(x):
    L = x[-1]
    return np.sin(2 * np.pi * x / L)


OUTPUT_DIR = "output_1d_animations"


def generate_animation(out_name="alfven_wave.mp4"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_name)
    sim = AlfvenWave(L=2.0, Nx=800, T=2.0)
    sim.initial_conditions(v0)
    writer = imageio.get_writer(out_path, fps=30)
    fig, ax = plt.subplots(figsize=(8, 4))
    for _ in range(sim.nt):
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

