"""Shallow water gravity wave example using the consolidated solver."""

import numpy as np

from wave_sim import ShallowWaterGravityWave
from wave_sim.animation_utils import generate_1d_animation, DEFAULT_OUTPUT_DIR_1D
from wave_sim.initial_conditions import gaussian_1d


def h0(x):
    L = x[-1] + (x[1] - x[0])
    return 1.0 + 0.1 * gaussian_1d(x, center=L/4, sigma=L/20)


def u0(x):
    return np.zeros_like(x)

DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR_1D


def generate_animation(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    out_name: str = "shallow_water_gravity_wave.mp4",
    steps: int | None = None,
) -> str:
    L_domain = 10.0
    Nx_points = 200
    T_duration = 2.0
    g_val = 9.81
    dx_val = L_domain / Nx_points
    dt_val = 0.5 * dx_val / np.sqrt(g_val)

    sim = ShallowWaterGravityWave(g=g_val, L=L_domain, Nx=Nx_points, dt=dt_val, T=T_duration)
    sim.initial_conditions(h0, u0)

    return generate_1d_animation(
        solver=sim,
        out_name=out_name,
        plot_variable_name="Q",
        title_prefix="Shallow Water Gravity Wave",
        y_label="Water height",
        output_dir=output_dir,
        y_lims=(0.8, 1.2),
        total_steps=steps,
    )


if __name__ == "__main__":
    path = generate_animation()
