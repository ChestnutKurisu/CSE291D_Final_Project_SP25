"""Plane acoustic wave example using the consolidated solver."""

import numpy as np

from wave_sim import PlaneAcousticWave
from wave_sim.animation_utils import generate_1d_animation, DEFAULT_OUTPUT_DIR_1D
from wave_sim.initial_conditions import gaussian_1d


def p0_initial(x):
    L = x[-1] if x.size > 0 else 1.0
    return gaussian_1d(x, center=L/4, sigma=L/20)

def dp0_dt_initial(x):
    return np.zeros_like(x)

DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR_1D


def generate_animation(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    out_name: str = "plane_acoustic_wave.mp4",
    steps: int | None = None,
) -> str:
    L_domain = 1.0
    Nx_points = 400
    T_duration = 2.0
    c_speed = 1.0
    dx_val = L_domain / Nx_points
    dt_val = 0.5 * dx_val / c_speed

    sim = PlaneAcousticWave(c=c_speed, L=L_domain, Nx=Nx_points, dt=dt_val, T=T_duration)
    sim.initial_conditions(p0_initial, dp_init_func=dp0_dt_initial)

    return generate_1d_animation(
        solver=sim,
        out_name=out_name,
        plot_variable_name="p_now",
        title_prefix="Plane Acoustic Wave",
        y_label="Pressure p(x,t)",
        output_dir=output_dir,
        y_lims=(-1.2, 1.2),
        total_steps=steps,
    )


if __name__ == "__main__":
    path = generate_animation()
