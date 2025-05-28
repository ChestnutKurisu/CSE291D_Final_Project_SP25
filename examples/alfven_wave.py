"""Alfv\u00e9n wave example using the consolidated solver."""

import numpy as np

from wave_sim import AlfvenWave
from wave_sim.animation_utils import generate_1d_animation, DEFAULT_OUTPUT_DIR_1D


def v0(x):
    L = x[-1]
    return np.sin(2 * np.pi * x / L)


DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR_1D


def generate_animation(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    out_name: str = "alfven_wave.mp4",
    steps: int | None = None,
) -> str:
    sim_L = 2.0
    sim_Nx = 800
    sim_T = 2.0
    dx = sim_L / sim_Nx
    vA_sim = 1.0
    dt_val = 0.5 * dx / vA_sim
    sim = AlfvenWave(L=sim_L, Nx=sim_Nx, T=sim_T, dt=dt_val)
    sim.initial_conditions(v0)

    return generate_1d_animation(
        solver=sim,
        out_name=out_name,
        plot_variable_name="v",
        title_prefix="Alfvén Wave",
        y_label="Velocity v(x,t)",
        output_dir=output_dir,
        y_lims=(-1.2, 1.2),
        total_steps=steps,
    )


if __name__ == "__main__":
    path = generate_animation()
    print(f"Wrote {path}")

