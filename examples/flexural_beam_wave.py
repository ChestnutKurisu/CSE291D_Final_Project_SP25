"""Flexural beam wave solved with the consolidated solver."""

import numpy as np

from wave_sim import FlexuralBeamWave
from wave_sim.animation_utils import generate_1d_animation, DEFAULT_OUTPUT_DIR_1D


def w0(x):
    L = x[-1]
    return np.exp(-100 * (x - L / 2) ** 2)


DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR_1D


def generate_animation(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    out_name: str = "flexural_beam_wave.mp4",
    steps: int | None = None,
) -> str:
    sim_L = 2.0
    sim_Nx = 801
    sim_T = 5.0
    dx = sim_L / (sim_Nx - 1)
    D_val = 0.01
    dt_val = 0.2 * dx ** 2 / np.sqrt(D_val)
    sim = FlexuralBeamWave(D=D_val, L=sim_L, Nx=sim_Nx, T=sim_T, dt=dt_val)
    sim.initial_conditions(w0)

    return generate_1d_animation(
        solver=sim,
        out_name=out_name,
        plot_variable_name="w",
        title_prefix="Flexural Beam Wave",
        y_label="Displacement w(x,t)",
        output_dir=output_dir,
        y_lims=(-1.1, 1.1),
        total_steps=steps,
    )


if __name__ == "__main__":
    path = generate_animation()
    print(f"Wrote {path}")

