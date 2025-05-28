"""Linear Rossby planetary wave solved spectrally."""

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os

from wave_sim import RossbyPlanetaryWave


def psi0(X, Y):
    Lx = X.max()
    Ly = Y.max()
    return np.exp(-((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2) / 0.2)


OUTPUT_DIR = "output_1d_animations"


def generate_animation(out_name="rossby_planetary_wave.mp4"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_name)
    sim = RossbyPlanetaryWave(Nx=256, Ny=256, T=3.0)
    sim.initial_conditions(psi0)
    X, Y = np.meshgrid(
        np.linspace(0, sim.Lx, sim.Nx, endpoint=False),
        np.linspace(0, sim.Ly, sim.Ny, endpoint=False),
        indexing="ij",
    )
    writer = imageio.get_writer(out_path, fps=30)
    fig, ax = plt.subplots(figsize=(6, 5))
    for _ in range(sim.nt):
        sim.step()
        ax.clear()
        cf = ax.contourf(X, Y, sim.psi, levels=20, cmap="RdBu_r")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
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

