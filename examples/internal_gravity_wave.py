"""1-D internal gravity wave using the consolidated solver."""

import numpy as np

from wave_sim import InternalGravityWave
from wave_sim.animation_utils import generate_1d_animation, DEFAULT_OUTPUT_DIR_1D


def psi0(x):
    return np.exp(-100 * (x - 1.0) ** 2)


DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR_1D


def generate_animation(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    out_name: str = "internal_gravity_wave.mp4",
    steps: int | None = None,
) -> str:
    sim_L = 2.0
    sim_Nx = 800
    sim_T = 3.0
    dx = sim_L / sim_Nx
    N_val = 1.0
    dt_val = 0.8 * dx / N_val
    sim = InternalGravityWave(L=sim_L, Nx=sim_Nx, T=sim_T, dt=dt_val)
    sim.initial_conditions(psi0)

    return generate_1d_animation(
        solver=sim,
        out_name=out_name,
        plot_variable_name="psi",
        title_prefix="Internal Gravity Wave",
        y_label="Streamfunction ψ(x,t)",
        output_dir=output_dir,
        y_lims=(-1.1, 1.1),
        total_steps=steps,
    )


if __name__ == "__main__":
    path = generate_animation()
    print(f"Wrote {path}")

