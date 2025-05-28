"""Plane acoustic wave example using the consolidated solver."""

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os

from wave_sim import PlaneAcousticWave
from wave_sim.initial_conditions import gaussian_1d


def p0_initial(x):
    L = x[-1] if x.size > 0 else 1.0
    return gaussian_1d(x, center=L/4, sigma=L/20)

def dp0_dt_initial(x):
    return np.zeros_like(x)

DEFAULT_OUTPUT_DIR = "output_1d_animations_individual"


def generate_animation(output_dir=DEFAULT_OUTPUT_DIR, out_name="plane_acoustic_wave.mp4"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)

    L_domain = 1.0
    Nx_points = 400
    T_duration = 2.0
    c_speed = 1.0
    dx_val = L_domain / Nx_points
    dt_val = 0.5 * dx_val / c_speed

    sim = PlaneAcousticWave(c=c_speed, L=L_domain, Nx=Nx_points, dt=dt_val, T=T_duration)
    sim.initial_conditions(p0_initial, dp_init_func=dp0_dt_initial)

    writer = imageio.get_writer(out_path, fps=30)
    fig, ax = plt.subplots(figsize=(8, 4))

    for i in range(sim.nt):
        sim.step()
        ax.clear()
        ax.plot(sim.x, sim.p_now)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("x")
        ax.set_ylabel("Pressure p(x,t)")
        ax.set_title(f"Plane Acoustic Wave, Time: {i*sim.dt:.3f}s")
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
