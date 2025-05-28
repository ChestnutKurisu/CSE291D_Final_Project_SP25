"""Spherical acoustic wave example using the consolidated solver."""

import numpy as np

from wave_sim import SphericalAcousticWave
from wave_sim.animation_utils import generate_1d_animation, DEFAULT_OUTPUT_DIR_1D
from wave_sim.initial_conditions import gaussian_1d


def p0_initial(r):
    R = r[-1] if r.size > 0 else 2.0
    return gaussian_1d(r, center=R/4, sigma=R/20)


def dp0_dt_initial(r):
    return np.zeros_like(r)

DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR_1D


def generate_animation(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    out_name: str = "spherical_acoustic_wave.mp4",
    steps: int | None = None,
) -> str:
    R_domain = 2.0
    Nr_points = 400
    T_duration = 2.0
    c_speed = 1.0
    dr_val = R_domain / Nr_points
    dt_val = 0.5 * dr_val / c_speed

    sim = SphericalAcousticWave(c=c_speed, R=R_domain, Nr=Nr_points, dt=dt_val, T=T_duration)
    sim.initial_conditions(p0_initial, dp_init_func=dp0_dt_initial)

    return generate_1d_animation(
        solver=sim,
        out_name=out_name,
        plot_variable_name="p_now",
        title_prefix="Spherical Acoustic Wave",
        y_label="Pressure p(r,t)",
        output_dir=output_dir,
        y_lims=(-1.2, 1.2),
        x_label="r",
        total_steps=steps,
    )


if __name__ == "__main__":
    path = generate_animation()
