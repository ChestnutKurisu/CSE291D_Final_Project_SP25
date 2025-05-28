"""Rotating shallow water Kelvin wave using the unified solver."""

import numpy as np

from wave_sim import KelvinWave
from wave_sim.animation_utils import generate_1d_animation, DEFAULT_OUTPUT_DIR_1D


def eta0(y):
    return np.exp(-((y - 2.5) / 0.5) ** 2)


DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR_1D


def generate_animation(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    out_name: str = "kelvin_wave.mp4",
    steps: int | None = None,
) -> str:
    sim_L = 10.0
    sim_Ny = 800
    sim_T = 10.0
    dy = sim_L / (sim_Ny - 1)
    c = np.sqrt(9.81 * 1.0)
    dt_val = 0.4 * dy / c
    sim = KelvinWave(L=sim_L, Ny=sim_Ny, T=sim_T, dt=dt_val)
    sim.initial_conditions(eta0)

    return generate_1d_animation(
        solver=sim,
        out_name=out_name,
        plot_variable_name="eta",
        title_prefix="Kelvin Wave",
        y_label="Surface displacement",
        output_dir=output_dir,
        y_lims=(-1.1, 1.1),
        x_label="y",
        total_steps=steps,
    )


if __name__ == "__main__":
    path = generate_animation()
    print(f"Wrote {path}")

