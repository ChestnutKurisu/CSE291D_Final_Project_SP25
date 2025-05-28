"""Spherical acoustic wave example using the consolidated solver."""

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os

from wave_sim import SphericalAcousticWave
from wave_sim.initial_conditions import gaussian_1d


def p0_initial(r):
    R = r[-1] if r.size > 0 else 2.0
    return gaussian_1d(r, center=R/4, sigma=R/20)


def dp0_dt_initial(r):
    return np.zeros_like(r)

DEFAULT_OUTPUT_DIR = "output_1d_animations_individual"


def generate_animation(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    out_name: str = "spherical_acoustic_wave.mp4",
    steps: int | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)

    R_domain = 2.0
    Nr_points = 400
    T_duration = 2.0
    c_speed = 1.0
    dr_val = R_domain / Nr_points
    dt_val = 0.5 * dr_val / c_speed

    sim = SphericalAcousticWave(c=c_speed, R=R_domain, Nr=Nr_points, dt=dt_val, T=T_duration)
    sim.initial_conditions(p0_initial, dp_init_func=dp0_dt_initial)

    writer = imageio.get_writer(out_path, fps=30)
    fig, ax = plt.subplots(figsize=(8, 4))

    nsteps = sim.nt if steps is None else min(steps, sim.nt)
    for i in range(nsteps):
        sim.step()
        ax.clear()
        ax.plot(sim.r, sim.p_now)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("r")
        ax.set_ylabel("Pressure p(r,t)")
        ax.set_title(f"Spherical Acoustic Wave, Time: {i*sim.dt:.3f}s")
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
